# -*- coding: utf-8 -*-
"""
InSAR 自动化处理脚本（SNAP gpt · 三段式 v4）

为什么是三段：
  SNAP 14 的相位解缠插件提供的是一个"外部工具适配器"算子，名字是
  `Snaphu-unwrapping`（连字符），而且它吃的是"已经导出到磁盘的相位文件"，
  不能像老写法那样直接接在滤波算子后面。所以按 SNAP 官方流程拆成三段：

  Pass 1  gpt        ：轨道校正 → 配准 → 干涉 → 去平 → 滤波
                        ├─ Write        → <out>/filtered.dim      （滤波后干涉图，留给 Pass 3）
                        └─ SnaphuExport → <out>/snaphu/            （导出相位 + snaphu.conf）
  Pass 2  snaphu.exe ：按 snaphu.conf 里的命令行做相位解缠
  Pass 3  gpt        ：SnaphuImport → PhaseToDisplacement → Terrain-Correction
                        └─ Write        → <out>/deformation_map.dim （LOS 形变图）

用法：
  python src/insar/run_insar.py --master <主影像 .SAFE/.zip> --slave <从影像> --out <输出目录>
  # 只生成处理图、不执行（用于检查配置）：
  python src/insar/run_insar.py --master ... --slave ... --out ... --dry-run

依赖：
  SNAP 14（gpt）        : C:\\Program Files\\esa-snap\\bin\\gpt
  SNAPHU 2.0.4（插件包）: C:\\vlmdisaster\\tools\\snaphu\\snaphu-v2.0.4_win64\\bin\\snaphu.exe
"""
import os
import sys
import glob
import time
import argparse
import subprocess
import xml.etree.ElementTree as ET

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GPT = r"C:\Program Files\esa-snap\bin\gpt"
SNAPHU_EXE = r"C:\vlmdisaster\tools\snaphu\snaphu-v2.0.4_win64\bin\snaphu.exe"
# snaphu 分块时的临时目录：故意放短路径，避免嵌在深层目录里触到 Windows MAX_PATH
SNAPHU_TILEDIR = r"C:\vlmdisaster\snaphu_tmp"
DEM_NAME = "SRTM 3Sec"
# S1 IW SLC 必须先切子条带（Back-Geocoding 明确要求 split 产品）。
# 本对影像覆盖 AOI 中心(116.0E, 38.25N)的子条带是 IW2 —— 由 SAFE 的
# annotation 地理定位网格算出：IW1=116.06~117.46E / IW2=115.06~116.48E
SUBSWATH = "IW2"
POLARISATION = "VV"

# 精轨校正开关。
# 关闭原因（2026-09-29 实测）：SNAP 读取本地 .EOF 精轨文件时，遇到中文用户名
# 路径含中文/非 ASCII 时可能抛 "Cannot parse xml file path : unknown protocol"；
# 而在线下载精轨的主机 step.esa.int 在本机被墙。关掉后直接使用 SLC 自带的
# 轨道状态矢量，整条流程可完全离线跑通。日后若换到纯英文用户名的机器，
# 把它改回 True 即可恢复精轨校正。
USE_ORBIT_CORRECTION = False
NS = "http://snap.esa.int/schema/GRAPH_PROCESSING"


def _new_graph():
    ET.register_namespace("", NS)
    root = ET.Element(f"{{{NS}}}graph", id="Graph")
    ET.SubElement(root, f"{{{NS}}}version").text = "1.0"
    return root


def _node(root, node_id, operator, params=None, sources=None):
    """加一个节点。sources 是 [(标签, 上游节点id)]，标签如 sourceProduct / sourceProduct.1"""
    node = ET.SubElement(root, f"{{{NS}}}node", id=node_id)
    ET.SubElement(node, f"{{{NS}}}operator").text = operator
    src = ET.SubElement(node, f"{{{NS}}}sources")
    for tag, refid in (sources or []):
        ET.SubElement(src, f"{{{NS}}}{tag}", refid=refid)
    if params:
        p = ET.SubElement(node, f"{{{NS}}}parameters")
        for k, v in params.items():
            ET.SubElement(p, f"{{{NS}}}{k}").text = str(v)
    return node


def _write_graph(root, path):
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def pick_subswath(master, lon, lat):
    """根据 AOI 中心点自动判断落在 IW1/IW2/IW3 哪个子条带。

    读 SAFE annotation 的 geolocationGridPoint（VV 极化优先），取每个
    子条带的经纬范围，返回包含 (lon, lat) 的那个。测不出来时回退到默认。
    """
    import re as _re
    ann_dir = os.path.join(master, "annotation")
    if not os.path.isdir(ann_dir):
        return None
    best = None
    for f in sorted(os.listdir(ann_dir)):
        if not f.endswith(".xml"):
            continue
        parts = f.split("-")
        if len(parts) < 4:
            continue
        sw, pol = parts[1].lower(), parts[3].lower()
        if sw not in ("iw1", "iw2", "iw3") or pol != "vv":
            continue
        try:
            txt = open(os.path.join(ann_dir, f), encoding="utf-8", errors="replace").read()
            la = [float(x) for x in _re.findall(r"<latitude>([-\d.eE+]+)</latitude>", txt)]
            lo = [float(x) for x in _re.findall(r"<longitude>([-\d.eE+]+)</longitude>", txt)]
        except Exception:
            continue
        if not la or not lo:
            continue
        if min(lo) <= lon <= max(lo) and min(la) <= lat <= max(la):
            best = sw.upper()
            print("          AOI 落在 %s（lon %.2f~%.2f, lat %.2f~%.2f）"
                  % (best, min(lo), max(lo), min(la), max(la)))
            break
    return best


def build_graph1(master, slave, out_dir, aoi=None, subswath=None, dem=None):
    """Pass 1：轨道校正 → 配准 → 干涉 → 去平 →（可选 Subset 切 AOI）→ 滤波

    aoi = (lon, lat, size_deg)：以 (lon,lat) 为中心、边长 size_deg 度的正方形窗口。
    切窗口后，filtered.dim / snaphu 导出 / Pass 3 全部随之变小，避免全场景卡死。
    subswath：子条带（默认 IW2），不同 AOI 可能落在 IW1/IW3。
    dem：DEM 名（默认 SRTM 3Sec，只覆盖 60°N 内；高纬如冰岛要用 ASTER 1sec GDEM）。
    """
    sw = subswath or SUBSWATH
    dem_name = dem or DEM_NAME
    root = _new_graph()
    _node(root, "read-master", "Read", {"file": "${master}"})
    _node(root, "read-slave", "Read", {"file": "${slave}"})

    if USE_ORBIT_CORRECTION:
        orbit = {"orbitType": "Sentinel Precise (Auto Download)", "polyDegree": 3,
                 "continueOnFail": "false"}
        _node(root, "orbit-master", "Apply-Orbit-File", orbit, [("sourceProduct", "read-master")])
        _node(root, "orbit-slave", "Apply-Orbit-File", orbit, [("sourceProduct", "read-slave")])
        master_src, slave_src = "orbit-master", "orbit-slave"
    else:
        master_src, slave_src = "read-master", "read-slave"

    split = {"subswath": sw, "selectedPolarisations": POLARISATION}
    _node(root, "split-master", "TOPSAR-Split", split, [("sourceProduct", master_src)])
    _node(root, "split-slave", "TOPSAR-Split", split, [("sourceProduct", slave_src)])

    _node(root, "coreg", "Back-Geocoding",
          {"demName": dem_name, "demResamplingMethod": "BILINEAR_INTERPOLATION"},
          [("sourceProduct", "split-master"), ("sourceProduct.1", "split-slave")])
    _node(root, "interf", "Interferogram", None, [("sourceProduct", "coreg")])
    _node(root, "deburst", "TOPSAR-Deburst", None, [("sourceProduct", "interf")])
    _node(root, "topo", "TopoPhaseRemoval", {"demName": dem_name},
          [("sourceProduct", "deburst")])

    filter_src = "topo"
    if aoi is not None:
        lon, lat, size = aoi
        h = size / 2.0
        # WKT 多边形（经纬度顺序：lon lat）
        wkt = (f"POLYGON (({lon-h} {lat-h}, {lon+h} {lat-h}, "
               f"{lon+h} {lat+h}, {lon-h} {lat+h}, {lon-h} {lat-h}))")
        _node(root, "subset", "Subset", {"geoRegion": wkt},
              [("sourceProduct", "topo")])
        filter_src = "subset"

    _node(root, "filter", "GoldsteinPhaseFiltering", None, [("sourceProduct", filter_src)])

    # 两个汇点：一个把滤波结果存盘（Pass 3 要用），一个导出相位给 snaphu
    _node(root, "write-filtered", "Write",
          {"file": os.path.join(out_dir, "filtered.dim"), "formatName": "BEAM-DIMAP"},
          [("sourceProduct", "filter")])
    _node(root, "snaphu-export", "SnaphuExport",
          {"targetFolder": os.path.join(out_dir, "snaphu"),
           "statCostMode": "DEFO", "initMethod": "MCF"},
          [("sourceProduct", "filter")])

    return _write_graph(root, os.path.join(out_dir, "graph1_insar.xml"))


def build_graph3(wrapped, unwrapped, out_dir):
    """Pass 3：导入解缠结果 → 相位转形变 → 地理编码 → 写形变图

    两个实测坑：
    1. SnaphuImport 的第二个输入是 snaphu 输出的“通用二进制”(ENVI) 文件，
       读节点必须显式指定 formatName=ENVI，否则 SNAP 拿不到栅格几何（tile=0），
       初始化时会在 XToTileX 处抛 “ArithmeticException: / by zero”。
    2. file 必须指向 .hdr（不是 .img）。SNAP 的 EnviProductReader 用
       FileReader（默认字符集）直接 parse 传入文件；若传 .img 二进制，
       会把二进制当文本读 → MalformedInputException: Input length = 1。
       传 .hdr 才能正常解析。
    3. Terrain-Correction 不要指定 demName/pixelSpacingInMeter/mapProjection
       （用默认）。指定 SRTM 3Sec 时会触发下载本地没有的 tile，建不出地图
       网格，初始化时同样抛 “ArithmeticException: / by zero”；用默认参数
       时它自己就能建立 WGS84(DD) 地图网格。
    """
    root = _new_graph()
    _node(root, "read-wrapped", "Read", {"file": wrapped})
    unwrapped_hdr = unwrapped
    if not unwrapped_hdr.lower().endswith(".hdr"):
        unwrapped_hdr = os.path.splitext(unwrapped_hdr)[0] + ".hdr"
    _node(root, "read-unwrapped", "Read",
          {"file": unwrapped_hdr, "formatName": "ENVI"})
    _node(root, "import", "SnaphuImport", {"doNotKeepWrapped": "false"},
          [("sourceProduct", "read-wrapped"), ("sourceProduct.1", "read-unwrapped")])
    _node(root, "displacement", "PhaseToDisplacement", None,
          [("sourceProduct", "import")])
    _node(root, "terrain", "Terrain-Correction", None,
          [("sourceProduct", "displacement")])
    _node(root, "write", "Write",
          {"file": os.path.join(out_dir, "deformation_map.dim"), "formatName": "BEAM-DIMAP"},
          [("sourceProduct", "terrain")])

    return _write_graph(root, os.path.join(out_dir, "graph3_import_geocode.xml"))


def _run_gpt(graph, extra_args):
    cmd = [GPT, "-J-Xmx12G", graph] + extra_args
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "")
    if out:
        print(out[-3000:])
    # 把 gpt 的完整输出落一份到文件（下次报错直接读文件，不再被截断）
    try:
        logp = os.path.join(os.path.dirname(graph), "gpt_last.log")
        with open(logp, "w", encoding="utf-8", errors="replace") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n" + "=" * 60 + "\n")
            lf.write("--- STDOUT ---\n" + (r.stdout or "") + "\n")
            lf.write("--- STDERR ---\n" + (r.stderr or "") + "\n")
    except Exception:
        pass
    if r.returncode != 0:
        print("  STDERR (头 2500 字):")
        print((r.stderr or "")[:2500])
        print("  STDERR (尾 1500 字):")
        print((r.stderr or "")[-1500:])
        print(f"  （完整输出已存至 {os.path.join(os.path.dirname(graph), 'gpt_last.log')}）")
        return False
    return True


def _find_snaphu_conf(snaphu_dir):
    """snaphu.conf 可能直接在 <out>/snaphu 下，也可能在 <out>/snaphu/<产品名>/ 子目录里
    （SNAP 收尾时会把导出文件移进产品同名子目录）。"""
    direct = os.path.join(snaphu_dir, "snaphu.conf")
    if os.path.isfile(direct):
        return direct
    hits = glob.glob(os.path.join(snaphu_dir, "*", "snaphu.conf"))
    return hits[0] if hits else None


def _read_snaphu_conf(snaphu_dir):
    conf = _find_snaphu_conf(snaphu_dir)
    if conf is None:
        return None, None, None, None
    exec_line = infile = outfile = None
    with open(conf, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if "snaphu -f snaphu.conf" in s:
                exec_line = s
            parts = s.split()
            if len(parts) >= 2:
                if parts[0] == "INFILE":
                    infile = parts[1]
                elif parts[0] == "OUTFILE":
                    outfile = parts[1]
    return conf, exec_line, infile, outfile


def _write_snaphu_local_conf(src_conf, work):
    """基于 SNAP 导出的 snaphu.conf 写一份本地配置，修掉两个实测坑：

    1. SNAP 写的 CORRFILE 是个空名字（值字面上就是 ".snaphu.img"），而导出目录里
       并没有相干性文件 → snaphu 会去开 ".snaphu.img" 然后 Abort。
       这里把它注释掉，让 snaphu 使用内置默认相干性。
    2. Windows 上 snaphu（MSYS2 构建）多进程分块会崩
       （"Unexpected or abnormal exit of child process"）→ 改单进程。
       同时把分块临时目录放到短路径，避免深层目录触到 MAX_PATH。
    """
    out_dir = os.path.dirname(os.path.dirname(work))   # <out>\snaphu\<产品> → <out>
    coh = glob.glob(os.path.join(out_dir, "filtered.data", "coh_*.img"))
    local_conf = os.path.join(work, "snaphu_local.conf")
    with open(src_conf, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # 读 wrapped phase 的尺寸（就是分块解缠时每块的期望尺寸来源，这里取全图尺寸用于比）
    def _read_img_size(img_path):
        hdr = os.path.splitext(img_path)[0] + ".hdr"
        if not os.path.exists(hdr):
            return None, None
        samples = lines_n = None
        with open(hdr, "r", encoding="utf-8", errors="replace") as hf:
            for hln in hf:
                low = hln.strip().lower()
                if low.startswith("samples"):
                    samples = int(low.split("=")[1].strip())
                elif low.startswith("lines"):
                    lines_n = int(low.split("=")[1].strip())
        return samples, lines_n

    # 是否分块：分块解缠在 Windows 上到“组装”那一步极易崩
    # （secondary arcs 超限）。设环境变量 SNAPHU_NO_TILE=1 改为整图一次解。
    no_tile = os.environ.get("SNAPHU_NO_TILE", "0") == "1"
    use_tiles = not no_tile

    with open(local_conf, "w", encoding="utf-8") as f:
        for ln in lines:
            if ln.strip().upper().startswith("CORRFILE"):
                # 一律禁用 CORRFILE：SNAP 带的相干性尺寸与相位不一致，
                # 无论分块与否都会 Abort；统一用 uniform weights。
                # 注意：写进 conf 的注释必须是纯 ASCII —— SNAP 的 ENVI reader
                #  会扫描同目录附属文件，遇到非 UTF-8 字节会抛 MalformedInputException。
                f.write("# [run_insar.py] CORRFILE disabled (use uniform weights)\n")
                continue
            if ln.strip().upper().startswith(("NTILEROW", "NTILECOL")):
                continue   # 丢弃原 conf 里的分块参数，避免重复定义
            if ln.strip().upper().startswith("NPROC"):
                continue   # 同理：NPROC 也会重复（SNAP 写一个，我们再加一个）
            f.write(ln)
        f.write("\n# ---- overrides appended by run_insar.py ----\n")
        if use_tiles:
            n_tile = os.environ.get("SNAPHU_TILES", "8")
            overrides = ("NTILEROW\t" + n_tile, "NTILECOL\t" + n_tile, "NPROC\t1",
                         "ROWOVRLP\t200", "COLOVRLP\t200",
                         "TILECOSTTHRESH\t100", "MINREGIONSIZE\t100",
                         "TILEDIR\t" + SNAPHU_TILEDIR)
        else:
            overrides = ("NTILEROW\t1", "NTILECOL\t1", "NPROC\t1",
                         "TILECOSTTHRESH\t100", "MINREGIONSIZE\t100")
        for line in overrides:
            f.write(line + "\n")
    if use_tiles:
        print("  分块解缠：禁用 CORRFILE（改用 uniform weights）")
    if not use_tiles:
        print("  整图解缠：不分块（NTILEROW/COL=1），跳过组装环节")
    return local_conf


def run_snaphu(snaphu_dir):
    """Pass 2：按 snaphu.conf 里的命令行调 snaphu.exe，返回解缠结果 .hdr 路径"""
    conf, exec_line, infile, outfile = _read_snaphu_conf(snaphu_dir)
    if exec_line is None:
        print(f"  ✗ 读不到 {conf}，或里面没有 'snaphu -f snaphu.conf' 命令行")
        return None

    work = os.path.dirname(conf)
    # snaphu.conf 里那行命令可能带前导 '#' 或绝对路径，不能简单 split()[1:]；
    # 先取 "snaphu -f " 之后的参数，再去掉第一个（原 conf 名），只留 infile/linelength。
    key = "snaphu -f "
    positional = exec_line[exec_line.find(key) + len(key):].split()[1:]

    _write_snaphu_local_conf(conf, work)
    args = ["-f", "snaphu_local.conf"] + positional
    print(f"  命令行: {SNAPHU_EXE} {' '.join(args)}")
    print(f"  工作目录: {work}")
    print("  ---- snaphu 实时输出（带时间戳，判断是否真卡） ----")

    # 不能用 subprocess.run(capture_output=True)：那样会把 snaphu 的全部输出
    # 吞进管道，界面上长时间无任何动静，看起来就像卡死。
    # 改成 Popen + 逐行实时打印，同时写日志文件备查。
    t0 = time.time()
    log_path = os.path.join(work, "snaphu_run.log")
    rc = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            [SNAPHU_EXE] + args, cwd=work,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            print("  [%6.1fs] %s" % (time.time() - t0, line), flush=True)
            lf.write(line + "\n")
            lf.flush()
        rc = proc.wait()

    print("  ---- snaphu 结束，耗时 %.1f s，返回码 %s ----" % (time.time() - t0, rc))
    if rc != 0:
        print("  ✗ snaphu 非零退出，日志: %s" % log_path)
        return None

    # 找解缠结果 UnwPhase_*.hdr。
    # 坑：SNAP 导出目录里同时有 Phase_ifg_*.snaphu.hdr（包装相位）和
    # UnwPhase_*.snaphu.hdr（解缠结果），两者 mtime 只差一两秒，靠
    # "挑最新 hdr" 会误选到 Phase_ifg，导致 Pass 3 的 SnaphuImport 拿到
    # 错误的输入、gpt 一启动就退出。这里改成显式只认 UnwPhase。
    cand = _find_unwrapped(snaphu_dir)
    if cand is None:
        print(f"  ✗ 解缠已执行，但找不到输出文件。目录: {snaphu_dir}")
    return cand


def _find_unwrapped(snaphu_dir):
    """在 snaphu 导出目录里找解缠结果 UnwPhase*.hdr（排除包装相位 Phase_ifg*.hdr）。

    坑：SNAP 导出后，Phase_ifg_*.snaphu.hdr（包装相位）与
    UnwPhase_*.snaphu.hdr（解缠结果）的 mtime 只差零点几秒，靠“挑最新 hdr”
    会误选到 Phase_ifg，导致 Pass 3 的 SnaphuImport 拿到错输入、gpt 启动即退。
    所以这里显式只认 UnwPhase，且只查 snaphu_dir 下的子目录。
    """
    pats = [os.path.join(snaphu_dir, "UnwPhase*.hdr"),
            os.path.join(snaphu_dir, "*", "UnwPhase*.hdr")]
    hits = []
    for p in pats:
        hits.extend(glob.glob(p))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="SNAP InSAR 三段式自动化处理")
    ap.add_argument("--master", required=True, help="主影像（.SAFE 目录或 .zip）")
    ap.add_argument("--slave", required=True, help="从影像（.SAFE 目录或 .zip）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--aoi", nargs=3, type=float, default=None,
                    metavar=("LON", "LAT", "SIZE"),
                    help="AOI 中心经度/纬度 + 边长(度)，如 --aoi 116.0 38.25 0.5")
    ap.add_argument("--dry-run", action="store_true", help="只生成处理图，不执行")
    ap.add_argument("--skip-pass1", action="store_true",
                    help="跳过 Pass 1，直接复用已存在的 filtered.dim 和 snaphu/ 导出结果")
    ap.add_argument("--only-pass3", action="store_true",
                    help="只跑 Pass 3：复用已存在的 filtered.dim 与 UnwPhase，跳过 Pass 1/2")
    ap.add_argument("--subswath", default=None, help="子条带 IW1/IW2/IW3（默认 IW2）")
    ap.add_argument("--dem", default=None, help="DEM 名（默认 SRTM 3Sec；高纬用 'ASTER 1sec GDEM'）")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    # 未显式指定子条带且有 AOI 时，按 AOI 位置自动判定
    if args.subswath is None and args.aoi is not None:
        auto = pick_subswath(args.master, args.aoi[0], args.aoi[1])
        if auto:
            args.subswath = auto
    snaphu_dir = os.path.join(out, "snaphu")
    filtered = os.path.join(out, "filtered.dim")

    print("=" * 60)
    print("  SNAP InSAR 三段式处理 ♪")
    print("=" * 60)
    print(f"  主影像  : {args.master}")
    print(f"  从影像  : {args.slave}")
    print(f"  输出目录: {out}")
    print()

    g1 = build_graph1(args.master, args.slave, out, aoi=args.aoi, subswath=args.subswath, dem=args.dem)
    print(f"[Pass 1] 处理图: {g1}")
    if args.aoi:
        print(f"          AOI: 中心({args.aoi[0]},{args.aoi[1]}) 边长{args.aoi[2]}度")

    if args.dry_run:
        g3 = build_graph3(filtered, "${unwrapped}", out)
        print(f"[Pass 3] 处理图: {g3}")
        print("\n[DRY-RUN] 只生成处理图，未执行。")
        return

    # ---- 只跑 Pass 3：跳过 Pass 1 和 Pass 2 ----
    if args.only_pass3:
        if not os.path.isfile(filtered):
            print(f"\n❌ --only-pass3 需要已存在的 filtered.dim：{filtered}")
            return
        unwrapped = _find_unwrapped(snaphu_dir)
        if not unwrapped:
            print(f"\n❌ --only-pass3 未在 snaphu/ 下找到 UnwPhase*.hdr，无法导入")
            return
        print("\n---------- Pass 1 / Pass 2: 跳过（复用已有 filtered.dim 与 UnwPhase） ----------")
        print(f"  解缠输入: {unwrapped}")
        print("\n---------- Pass 3: 导入 → 相位转形变 → 地理编码 ----------")
        g3 = build_graph3(filtered, unwrapped, out)
        print(f"  处理图: {g3}")
        if not _run_gpt(g3, []):
            print("\n❌ Pass 3 失败")
            return
        print("\n" + "=" * 60)
        print("  ✅ InSAR 处理完成！")
        print("=" * 60)
        print(f"  形变图: {os.path.join(out, 'deformation_map.dim')}")
        return

    # ---- Pass 1 ----
    if args.skip_pass1:
        conf = _find_snaphu_conf(snaphu_dir)
        missing = []
        if not os.path.isfile(filtered):
            missing.append(filtered)
        if conf is None:
            missing.append(os.path.join(snaphu_dir, "snaphu.conf"))
        if missing:
            print("\n❌ --skip-pass1 需要以下文件已存在，但缺少:")
            for m in missing:
                print("   " + m)
            return
        print("\n---------- Pass 1: 跳过（复用已有的 filtered.dim 与 snaphu 导出） ----------")
    else:
        print("\n---------- Pass 1: 轨道校正 → 配准 → 干涉 → 去平 → 滤波 → 导出相位 ----------")
        print("  （要先下精轨和 DEM，第一次会慢，别关窗口）")
        if not _run_gpt(g1, ["-Pmaster=" + args.master, "-Pslave=" + args.slave]):
            print("\n❌ Pass 1 失败")
            return
        print("  ✅ Pass 1 完成")

    # ---- Pass 2 ----
    print("\n---------- Pass 2: SNAPHU 相位解缠 ----------")
    unwrapped = run_snaphu(snaphu_dir)
    if not unwrapped:
        print(f"\n❌ Pass 2 失败（导出目录: {snaphu_dir}）")
        return
    print(f"  ✅ Pass 2 完成 → {unwrapped}")

    # ---- Pass 3 ----
    print("\n---------- Pass 3: 导入 → 相位转形变 → 地理编码 ----------")
    g3 = build_graph3(filtered, unwrapped, out)
    print(f"  处理图: {g3}")
    if not _run_gpt(g3, []):
        print("\n❌ Pass 3 失败")
        return

    print("\n" + "=" * 60)
    print("  ✅ InSAR 处理完成！形变图:")
    print("     " + os.path.join(out, "deformation_map.dim"))
    print("=" * 60)


if __name__ == "__main__":
    main()
