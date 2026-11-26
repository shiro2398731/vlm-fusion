# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC 下载脚本 — asf_search 认证版 v3

用途: SAR/InSAR 源（源5）——地震/滑坡/沉降/火山形变预测训练数据
产品: IW SLC（干涉用）
数据源: ASF（Earthdata 账号认证）

⚠️ 关键: ASF datapool 下载不能直接 curl（token/密码都会 401），
        必须用 asf_search 库（内部自动从 ASF 认证端点换临时凭证）。

前置依赖（首次先装）:
  C:\\anaconda\\python.exe -m pip install asf_search

认证: 用 Earthdata 用户名 + 密码（auth_with_creds）。
      密码不写死，通过 --pass 传入，或环境变量 EARTHDATA_PASS。

用法:
  # 先 dry-run 看清单
  python scripts\\download_sentinel1.py --disaster ground_subsidence --start 2026-01-01 --end 2026-08-31 --orbit 47 --pass 你的密码 --dry-run
  # 真下
  python scripts\\download_sentinel1.py --disaster ground_subsidence --start 2026-01-01 --end 2026-08-31 --orbit 47 --pass 你的密码
"""
import os
from config_local import cred as _cred
import sys
import argparse
from datetime import datetime, timedelta

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import asf_search
except ImportError:
    print("❌ 缺少 asf_search，先安装:")
    print("   C:\\anaconda\\python.exe -m pip install asf_search")
    sys.exit(1)

OUT_ROOT = r"C:\vlmdisaster\data\raw\05_sar_insar\slc"
OUT_ROOT_GRD = r"C:\vlmdisaster\data\raw\05_sar_insar\grd"

# 按灾害分类（对应 DISASTER_SAR_PLAN.md）
# product: SLC=干涉形变类；GRD=水体/海冰类（强度影像，不干涉）
DISASTERS = {
    # --- 形变类（SLC + InSAR）---
    "ground_subsidence": {"bbox": (115.0, 37.5, 117.0, 39.0), "name": "地面沉降", "product": "SLC"},
    "earthquake_deformation": {"bbox": (36.0, 37.0, 37.0, 38.0), "name": "地震形变", "product": "SLC"},
    "landslide_deformation": {"bbox": (103.0, 35.9, 103.7, 36.3), "name": "滑坡形变", "product": "SLC"},
    "ground_collapse": {"bbox": None, "name": "地面塌陷", "product": "SLC"},
    "volcano_deformation": {"bbox": None, "name": "火山形变", "product": "SLC"},
    # --- 水体/海冰类（GRD + 后向散射阈值）---
    "flood_water": {"bbox": (112.0, 29.0, 116.5, 31.5), "name": "洪水水体", "product": "GRD"},
    "waterlogging": {"bbox": (114.5, 34.5, 118.5, 37.5), "name": "涝灾渍涝", "product": "GRD"},
    "sea_ice": {"bbox": (117.5, 37.0, 122.5, 41.0), "name": "海冰凌汛", "product": "GRD"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disaster", help="灾害名（见 DISASTERS）")
    ap.add_argument("--bbox", help="西,南,东,北（与 --disaster 二选一）")
    ap.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    ap.add_argument("--orbit", type=int, default=None, help="只下指定相对轨道")
    ap.add_argument("--limit", type=int, default=2, help="只下最新的 N 景（默认 2，够组 1 对；0=全部）")
    ap.add_argument("--vv-only", action="store_true", help="只保留 VV+VV 单极化的景（去重共轭）")
    ap.add_argument("--product", choices=["SLC", "GRD"], default=None,
                    help="产品类型（默认按灾害定义；裸 bbox 默认 SLC）")
    ap.add_argument("--user", default=_cred("EARTHDATA_USER"), help="Earthdata 用户名")
    ap.add_argument("--password", default=None, help="Earthdata 密码（或环境变量 EARTHDATA_PASS）")
    ap.add_argument("--dry-run", action="store_true", help="只搜不下")
    args = ap.parse_args()

    pwd = args.password or _cred("EARTHDATA_PASS")
    if not pwd:
        print("✗ 需要密码：--pass 你的密码 或设置环境变量 EARTHDATA_PASS")
        sys.exit(1)

    # 认证
    session = asf_search.ASFSession()
    try:
        session.auth_with_creds(args.user, pwd)
    except Exception as e:
        print(f"❌ 认证失败: {e}")
        sys.exit(1)
    print("✅ ASF 认证成功")

    # AOI
    if args.disaster:
        if args.disaster not in DISASTERS:
            print("✗ 未知灾害:", args.disaster, "可用:", list(DISASTERS))
            sys.exit(1)
        info = DISASTERS[args.disaster]
        if info["bbox"] is None:
            print(f"✗ '{args.disaster}' 的 AOI 未定，先在 DISASTER_SAR_PLAN.md 填")
            sys.exit(1)
        w, s, e, n = info["bbox"]
        product = args.product or info.get("product", "SLC")
        root = OUT_ROOT_GRD if product == "GRD" else OUT_ROOT
        out_dir = os.path.join(root, args.disaster)
    else:
        w, s, e, n = [float(x) for x in args.bbox.split(",")]
        product = args.product or "SLC"
        out_dir = OUT_ROOT_GRD if product == "GRD" else OUT_ROOT

    aoi = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    print("=" * 60)
    print(f"  Sentinel-1 {product} 下载 (asf_search v3)")
    print("=" * 60)
    print(f"  产品: {product}")
    print(f"  AOI: {aoi}")
    print(f"  时间: {start.date()} ~ {end.date()}")
    print(f"  轨道: {args.orbit or '全部'}")
    print(f"  输出: {out_dir}")
    print()

    # asf_search 的 processingLevel 枚举：GRD 分 GRD_HD / GRD_MD，没有裸 'GRD'
    proc_level = 'GRD_HD' if product == 'GRD' else product
    results = asf_search.search(
        platform=['SENTINEL-1A', 'SENTINEL-1B', 'SENTINEL-1C', 'SENTINEL-1D'],
        processingLevel=proc_level,
        start=start, end=end,
        intersectsWith=aoi,
    )
    if args.orbit is not None:
        results = [r for r in results if r.properties.get("pathNumber") == args.orbit]

    # 去重：同一天、同轨、同飞向的只留一景（避免相邻 burst 重复）
    seen = {}
    for r in results:
        p = r.properties
        key = (p.get("pathNumber"), p.get("flightDirection"), str(p.get("startTime", ""))[:10])
        if key not in seen:
            seen[key] = r
    results = list(seen.values())

    # 优先保留同一轨道的景（InSAR 配对需同轨）；并只取同一颗卫星（S1A/S1B 不能混对）
    if results:
        from collections import Counter
        orbits = Counter(r.properties.get("pathNumber") for r in results)
        main_orbit = orbits.most_common(1)[0][0]
        results = [r for r in results if r.properties.get("pathNumber") == main_orbit]
        print(f"  选定主轨道 {main_orbit}")
        # 按卫星（platform）分组，取景最多的那颗
        plats = Counter(r.properties.get("platform") for r in results)
        if len(plats) > 1:
            main_plat = plats.most_common(1)[0][0]
            results = [r for r in results if r.properties.get("platform") == main_plat]
            print(f"  选定卫星 {main_plat}（排除混对的另一颗）")

    # 只下最新 N 景
    if args.limit and args.limit > 0 and len(results) > args.limit:
        results = results[:args.limit]

    print(f"  搜到 {len(results)} 景")
    total = sum(r.properties.get("bytes", 0) for r in results)
    print(f"  合计约 {total/1e9:.1f} GB")
    print()

    if args.dry_run:
        for r in results:
            print(f"    · {r.properties.get('fileName')} ({r.properties.get('bytes',0)/1e9:.1f}GB)")
        print("\n  [DRY-RUN] 未下载。")
        return

    os.makedirs(out_dir, exist_ok=True)
    print("  开始下载（curl 断点续传模式，大文件耐心）...")
    import subprocess
    for r in results:
        url = r.properties.get("url")
        fn = r.properties.get("fileName")
        if not url or not fn:
            continue
        save = os.path.join(out_dir, fn)
        # 已完整（≥4GB）的跳过
        min_ok = 4_000_000_000 if product == "SLC" else 300_000_000
        if os.path.exists(save) and os.path.getsize(save) > min_ok:
            print(f"    · 已有 {fn}"); continue
        # 残缺文件删除
        if os.path.exists(save):
            print(f"    ! {fn} 残缺({os.path.getsize(save)/1e6:.0f}MB)，删除重下")
            os.remove(save)
        print(f"    下载 {fn} ...", flush=True)
        # 用 asf_search 自带下载器（内部处理认证/签名）
        try:
            r.download(path=out_dir, session=session)
            got = os.path.getsize(save) if os.path.exists(save) else 0
            print(f"    ✓ size={got/1e6:.0f}MB", flush=True)
        except Exception as e:
            print(f"    ✗ {fn}: {e}", flush=True)
    print("  ✅ 下载完成")


if __name__ == "__main__":
    main()
