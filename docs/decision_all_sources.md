# 决策融合三方对照

| 元素 | baseline | CLIP 决策 | DINOv2 决策 | 最优 |
|---|--:|--:|--:|--:|
| bloom_chla | 4/6 | 4/6 | 4/6 | baseline (0) |
| coastal_nt | 5/6 | 6/6 | 6/6 | CLIP (+1) |
| coastal_sla | 4/8 | 5/8 | 5/8 | CLIP (+1) |
| coral_dhw | 5/6 | 6/6 | 6/6 | CLIP (+1) |
| drought_grace | 4/4 | 4/4 | 4/4 | baseline (0) |
| drought_ndvi_clim | 4/4 | 4/4 | 4/4 | baseline (0) |
| earthquake_def | 4/6 | 6/6 | 6/6 | CLIP (+2) |
| enso | 6/6 | 6/6 | 6/6 | baseline (0) |
| flare | 2/8 | 4/8 | 6/8 | DINOv2 (+4) |
| flood_opt | 6/6 | 6/6 | 6/6 | baseline (0) |
| flood_precip_global | 2/8 | 7/8 | 7/8 | CLIP (+5) |
| flood_water | 3/4 | 4/4 | 4/4 | CLIP (+1) |
| geomagnetic | 4/7 | 4/7 | 5/7 | DINOv2 (+1) |
| gpm_nt | 3/6 | 6/6 | 6/6 | CLIP (+3) |
| green_tide | 5/6 | 4/6 | 1/6 | baseline (0) |
| ground_subsidence | 4/4 | 4/4 | 4/4 | baseline (0) |
| groundwater_grace | 4/4 | 4/4 | 4/4 | baseline (0) |
| groundwater_overuse | 4/6 | 4/6 | 4/6 | baseline (0) |
| haze | 2/6 | 6/6 | 6/6 | CLIP (+4) |
| heatwave_lst | 3/4 | 4/4 | 4/4 | CLIP (+1) |
| hurricane_opt | 3/6 | 4/6 | 5/6 | DINOv2 (+2) |
| hurricane_rain | 2/8 | 7/8 | 5/8 | CLIP (+5) |
| iod | 6/6 | 6/6 | 3/6 | baseline (0) |
| marine_heatwave | 4/4 | 4/4 | 3/4 | baseline (0) |
| sanddust | 4/6 | 5/6 | 6/6 | DINOv2 (+2) |
| sea_ice_amsr2 | 3/8 | 6/8 | 6/8 | CLIP (+3) |
| sea_ice_s2_arctic | 2/4 | 4/4 | 4/4 | CLIP (+2) |
| snow_fsc | 2/5 | 5/5 | 4/5 | CLIP (+3) |
| snow_swe | 3/8 | 5/8 | 5/8 | CLIP (+2) |
| snowstorm | 7/8 | 7/8 | 7/8 | baseline (0) |
| uvai_dust | 5/8 | 6/8 | 7/8 | DINOv2 (+2) |
| volcano_so2 | 4/8 | 6/8 | 5/8 | CLIP (+2) |
| volcano_thermal | 3/4 | 4/4 | 4/4 | CLIP (+1) |
| waterlogging_pct | 4/6 | 4/6 | 5/6 | DINOv2 (+1) |
| wave | 2/8 | 3/8 | 5/8 | DINOv2 (+3) |
| wildfire | 0/4 | 4/4 | 4/4 | CLIP (+4) |
| wildfire_smoke | 3/8 | 6/8 | 6/8 | CLIP (+3) |
| wind_gale | 3/6 | 4/6 | 5/6 | DINOv2 (+2) |

合计（以最优探针计）：升 26、平 12、降 0，共 38 元素