# -*- coding: utf-8 -*-
r"""
批量把元素转成结构化 JSON 输出（修正版）

## 2026-11-10 修正

第一版的正类集合是凭印象写的，跟真实标签对不上。这一版全部按
`label_stats.py` 跑出来的真实标签分布重写。

## 原则

- 每个元素的判据和物理量都不同，全部从数据集已有的字段里取
- **不凭空生成任何数值**，数值字段原样搬运
- detected 的判定严格按真实标签，不硬凑正负比例
"""

import argparse
import collections
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LF_DATA = r"C:\vlmdisaster\LLaMA-Factory\data"
INFO = os.path.join(LF_DATA, "dataset_info.json")

# 每个元素：
#   key    数据集 key
#   out    输出文件名前缀
#   dtype  灾害类型（进 JSON）
#   src    遥感源
#   pq     核心物理量
#   pos    正类标签集合（按真实标签）
#   nums   数值字段 [(源字段, 输出字段, 单位)]
#   crit   判据（有标准出处的写标准号）
SPECS = [
    dict(key='disaster_01_snow_fsc', out='snow_fsc', dtype='snow_cover',
         src='OPT', pq='fractional_snow_cover',
         pos={'moderate', 'light', 'heavy'},
         nums=[('mean_fsc', 'mean_fsc', '')],
         crit='NDSI-based fractional snow cover; moderate and above is hazardous'),

    dict(key='disaster_01_green_tide', out='green_tide', dtype='green_tide',
         src='OPT', pq='floating_algae_index',
         pos={'green_tide', 'patchy'},
         nums=[('frac_above', 'frac_above', ''), ('threshold', 'threshold', '')],
         crit='FAI above threshold fraction indicates floating algae'),

    dict(key='disaster_01_hurricane', out='hurricane_opt', dtype='tropical_cyclone',
         src='TIR', pq='brightness_temperature',
         pos={'mature', 'developing'},
         nums=[('frac_cold', 'frac_cold', '')],
         crit='fraction of cloud tops at or below 200 K indicates organized convection'),

    dict(key='disaster_03_lst', out='lst', dtype='temperature_hazard',
         src='TIR', pq='land_surface_temperature',
         pos={'heatwave', 'cold_wave', 'frost'},
         nums=[('day_p90', 'day_p90_c', 'degC'), ('night_p10', 'night_p10_c', 'degC')],
         crit='day P90 and night P10 anomalies indicate temperature hazard'),

    dict(key='disaster_05_waterlogging', out='waterlogging', dtype='waterlogging',
         src='SAR', pq='backscatter_dark_fraction',
         pos={'water'}, nums=[('dark_frac', 'dark_frac', '')],
         crit='SAR dark-area fraction above 35 percent indicates water body'),

    dict(key='disaster_05_flood_water', out='flood_water', dtype='flood',
         src='SAR', pq='backscatter_dark_fraction',
         pos={'water'}, nums=[('dark_frac', 'dark_frac', '')],
         crit='SAR dark-area fraction above 35 percent indicates water body'),

    dict(key='disaster_01_volcano', out='volcano_ndvi', dtype='volcanic_eruption',
         src='OPT', pq='ndvi_change',
         pos={'bright_plume', 'ash_deposit'},
         nums=[('d_ndvi', 'd_ndvi', ''), ('d_bright', 'd_bright', '')],
         crit='NDVI decrease and brightness increase after eruption'),

    dict(key='disaster_03_volcano_thermal', out='volcano_thermal', dtype='volcanic_thermal_anomaly',
         src='TIR', pq='brightness_temperature_hotspot',
         pos={'yes'}, nums=[('n_hot', 'n_hot_pixels', 'pixels'),
                            ('n_points', 'n_points', 'pixels')],
         crit='hot pixels above background brightness temperature'),

    # 三种标签都是 CME，detected 恒为 true，分类靠 severity
    dict(key='disaster_12_cme', out='cme', dtype='coronal_mass_ejection',
         src='OPT', pq='cme_speed',
         pos={'normal_cme', 'halo_cme', 'fast_cme'},
         nums=[('speed_kms', 'speed_kms', 'km/s'), ('width', 'width_deg', 'deg')],
         crit='CME speed and angular width from coronagraph; halo CME faces Earth'),

    dict(key='disaster_12_geomagnetic', out='geomagnetic', dtype='geomagnetic_storm',
         src='MW', pq='dst_index',
         pos={'weak_storm', 'moderate_storm', 'severe_storm', 'extreme_storm'},
         nums=[('min_dst', 'min_dst_nt', 'nT')],
         crit='Dst below -30 weak, below -50 moderate, below -100 severe, below -200 extreme'),

    dict(key='disaster_05_earthquake_deformation', out='earthquake_def', dtype='earthquake_deformation',
         src='SAR', pq='los_displacement',
         pos={'obvious_deform', 'slight_deform'},
         nums=[('median_mm', 'median_mm', 'mm')],
         crit='InSAR LOS displacement median indicates surface deformation'),

    dict(key='disaster_05_landslide_deformation', out='landslide_def', dtype='landslide_deformation',
         src='SAR', pq='los_displacement',
         pos={'obvious_deform', 'slight_deform'},
         nums=[('median_mm', 'median_mm', 'mm')],
         crit='InSAR LOS displacement median indicates slope deformation'),

    dict(key='disaster_12_flare', out='flare', dtype='solar_flare',
         src='OPT', pq='xrs_b_flux',
         pos={'c_flare', 'm_flare', 'x_flare'},
         nums=[('xrsb_flux', 'xrsb_flux', 'W/m2')],
         crit='GOES XRS-B peak flux: C 1e-6 to 1e-5, M 1e-5 to 1e-4, X above 1e-4'),

    dict(key='disaster_06_gpm', out='gpm', dtype='heavy_rainfall',
         src='PRECIP', pq='precipitation_accumulation',
         pos={'heavy_rain', 'moderate_rain'},
         nums=[('mean_mm', 'mean_mm', 'mm'), ('max_mm', 'max_mm', 'mm')],
         crit='24-hour accumulation above 50 mm indicates rainstorm'),

    dict(key='disaster_04_smap', out='smap', dtype='soil_moisture_anomaly',
         src='MW', pq='soil_moisture',
         pos={'dry', 'wet'}, nums=[('mean_sm', 'mean_sm', 'm3/m3')],
         crit='soil moisture deviation from climatology'),

    dict(key='disaster_11_haze', out='haze', dtype='haze',
         src='OPT', pq='aerosol_optical_depth',
         pos={'haze', 'severe_haze'}, nums=[('mean_aod', 'mean_aod', '')],
         crit='AOD above 0.5 indicates haze, above 1.0 severe haze'),

    dict(key='disaster_03_enso', out='enso', dtype='enso',
         src='MW', pq='nino34_sst_anomaly',
         pos={'el_nino', 'la_nina'}, nums=[('sst_anomaly_c', 'sst_anomaly_c', 'degC')],
         crit='Nino3.4 SST anomaly: above +0.5 El Nino, below -0.5 La Nina'),

    dict(key='disaster_03_iod', out='iod', dtype='iod',
         src='MW', pq='dipole_mode_index',
         pos={'positive_iod', 'negative_iod'}, nums=[('dmi_c', 'dmi_c', 'degC')],
         crit='Dipole Mode Index: above +0.4 positive, below -0.4 negative'),

    dict(key='disaster_03_marine_heatwave', out='marine_heatwave', dtype='marine_heatwave',
         src='MW', pq='sst_anomaly_area',
         pos={'marine_heatwave'}, nums=[('area_frac', 'area_frac', 'percent')],
         crit='SST anomaly above +1.0 degC over significant area'),

    # GB/T 20482-2017 牧区雪灾等级：light/moderate/severe 是灾，no_disaster 不是
    dict(key='disaster_04_snowstorm', out='snowstorm', dtype='snowstorm',
         src='OPT', pq='snow_depth',
         pos={'severe', 'moderate', 'light'}, nums=[('snow_cm', 'snow_cm', 'cm')],
         crit='GB/T 20482-2017 pastoral snow disaster grading by snow depth'),

    dict(key='disaster_05_groundwater_overuse', out='groundwater_overuse', dtype='groundwater_overuse',
         src='SAR', pq='los_displacement',
         pos={'obvious_subsidence', 'slight_subsidence'},
         nums=[('median_mm', 'median_mm', 'mm')],
         crit='InSAR subsidence median indicates groundwater overuse'),

    dict(key='disaster_06_hurricane_merged', out='hurricane_rain', dtype='tropical_cyclone_rainfall',
         src='PRECIP', pq='daily_precipitation',
         pos={'extreme_rainstorm', 'heavy_rainstorm', 'rainstorm'},
         nums=[('max_mm', 'max_mm', 'mm/day')],
         crit='CMA rainfall level: rainstorm above 50 mm/day'),

    dict(key='disaster_11_sanddust', out='sanddust', dtype='sand_dust',
         src='OPT', pq='aerosol_optical_depth',
         pos={'dust', 'dust_storm'}, nums=[('mean_aod', 'mean_aod', '')],
         crit='AOD above 1.0 indicates sand-dust loading'),

    dict(key='disaster_08_wave', out='wave', dtype='high_wave',
         src='ALT', pq='significant_wave_height',
         pos={'red', 'orange', 'yellow'},
         nums=[('hs_max', 'hs_max_m', 'm'), ('hs_median', 'hs_median_m', 'm')],
         # 专业口径：severity 直接用 GB/T 浪高等级 red/orange/yellow/blue/calm
         crit='GB/T wave level: red>=6.0, orange 4.5-6.0, yellow 2.5-4.5, blue<2.5'),

    dict(key='disaster_01_landslide', out='landslide', dtype='landslide',
         src='OPT', pq='ndvi_change',
         pos={'landslide_scar', 'revegetated'}, nums=[('ndvi_delta', 'ndvi_delta', '')],
         crit='NDVI change after event indicates vegetation loss from landslide'),

    dict(key='disaster_01_sand_dust', out='sand_dust', dtype='sand_dust',
         src='OPT', pq='dust_fraction',
         pos={'dust'}, nums=[('dust_frac', 'dust_frac', '')],
         crit='fraction of window covered by dust plume'),

    # 2026-11-14 新增：光学洪水（与 SAR 的 flood_water 是不同源，不重复）
    dict(key='disaster_01_flood', out='flood_opt', dtype='flood',
         src='OPT', pq='water_fraction',
         pos={'flood', 'waterlogging'}, nums=[('water_frac', 'water_frac', '')],
         crit='optical water index fraction above 20 percent indicates inundation'),
]


def make_question(numfields, severity_levels):
    nf = ', '.join(numfields) if numfields else 'none'
    return ('<image>This is a remote-sensing disaster detection image. Output a JSON '
            'object with fields: disaster_type, detected (true/false), source, '
            'physical_quantity, severity, %s, criterion. '
            'severity must be one of these exact level words: %s.'
            % (nf, severity_levels))


def build(rec, spec):
    label = str(rec.get('label', '')).lower()
    detected = label in spec['pos']

    obj = {
        'disaster_type': spec['dtype'],
        'detected': detected,
        'source': spec['src'],
        'physical_quantity': spec['pq'],
        # severity 一律用该领域的标准等级词（GB/T、GOES、CMA 等），不用通用语义词
        'severity': label,
    }

    numnames = []
    for src_key, out_key, unit in spec['nums']:
        v = rec.get(src_key)
        if v is None:
            continue
        try:
            obj[out_key] = round(float(v), 3)
        except (TypeError, ValueError):
            obj[out_key] = v
        if unit:
            obj[out_key + '_unit'] = unit
        numnames.append(out_key + ('(%s)' % unit if unit else ''))

    obj['criterion'] = spec['crit']
    return obj, numnames


def run(spec, dry_run, val_ratio=0.12):
    info = json.load(open(INFO, encoding='utf-8'))
    key = spec['key']
    if key not in info:
        print('  ✗ %s 未注册' % key)
        return None
    p = os.path.join(LF_DATA, info[key]['file_name'])
    if not os.path.exists(p):
        print('  ✗ %s 文件缺失' % key)
        return None

    recs = json.load(open(p, encoding='utf-8'))
    out = []
    numnames = []
    # 从数据集里收集真实的 severity 取值，写进问题里约束模型
    levels = []
    for r in recs:
        lab = r.get('label')
        if lab is not None and str(lab) not in levels:
            levels.append(str(lab))
    levels_str = ' / '.join(levels[:8])

    for r in recs:
        conv = r.get('conversations') or []
        if len(conv) < 2:
            continue
        obj, nf = build(r, spec)
        numnames = nf
        conv[0]['value'] = make_question(nf, levels_str)
        conv[1]['value'] = json.dumps(obj, ensure_ascii=False)
        r['conversations'] = conv
        out.append(r)

    if not out:
        print('  ✗ %s 无有效记录' % key)
        return None

    n_val = max(20, int(len(out) * val_ratio))
    random.Random(42).shuffle(out)
    val, train = out[:n_val], out[n_val:]

    pos_n = sum(1 for r in out
                if json.loads(r['conversations'][1]['value'])['detected'])
    pct = pos_n / len(out) * 100
    warn = ''
    if pct >= 90 or pct <= 5:
        warn = ' ⚠️近乎单类'
    print('  %-22s %6d 条  正 %5d (%3.0f%%)  %s%s'
          % (spec['out'], len(out), pos_n, pct,
             ','.join(numnames) or '无数值', warn))

    if dry_run:
        print('     样例: %s' % out[0]['conversations'][1]['value'][:160])
        return (spec['out'], len(train), len(val), pos_n)

    for data, suffix in ((train, ''), (val, '_val')):
        fn = 'train_%s_json%s.json' % (spec['out'], suffix)
        json.dump(data, open(os.path.join(LF_DATA, fn), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    return (spec['out'], len(train), len(val), pos_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--only', type=str, default='')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.list:
        print('%-18s %-34s %-26s %s' % ('输出名', '数据集', '灾害类型', '源'))
        print('-' * 88)
        for s in SPECS:
            print('%-18s %-34s %-26s %s'
                  % (s['out'], s['key'], s['dtype'], s['src']))
        print('-' * 88)
        print('共 %d 个元素' % len(SPECS))
        return

    todo = SPECS
    if args.only:
        keys = {x.strip() for x in args.only.split(',')}
        todo = [s for s in SPECS if s['out'] in keys]

    print('=' * 88)
    print('  批量转结构化 JSON：%d 个元素' % len(todo))
    print('  正类集合按真实 label 分布写，数值原样搬运')
    print('=' * 88)

    results = []
    for s in todo:
        try:
            r = run(s, args.dry_run)
            if r:
                results.append(r)
        except Exception as e:
            print('  ✗ %s 失败: %s' % (s['out'], str(e)[:100]))

    print('\n' + '=' * 88)
    print('完成 %d 个元素' % len(results))
    if not args.dry_run:
        print('下一步：注册到 dataset_info.json，再批量训练。')


if __name__ == '__main__':
    main()
