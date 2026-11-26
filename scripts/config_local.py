# -*- coding: utf-8 -*-
"""本地账号配置（公开仓库里全部留空，用户自行填写）。

用法：
  1. 按 `docs/账号注册与配置.md` 注册对应平台的账号；
  2. 在下面填你自己的用户名 / 密码 / token；
  
"""

import os

# NASA Earthdata（MODIS / GPM / SMAP / OMI 等）
EARTHDATA_USER = ""   # 填 Earthdata 用户名
EARTHDATA_PASS = ""   # 填 Earthdata 密码
EARTHDATA_TOKEN = ""  # 填 Earthdata Login 生成的 token（推荐）

# Copernicus CDSE（Sentinel-1/2/3）
CDSE_USER = ""        # 填 CDSE 注册邮箱
CDSE_PASS = ""        # 填 CDSE 密码

# NSIDC（AMSR2 / 海冰 / 积雪；如需）
NSIDC_USER = ""
NSIDC_PASS = ""

def cred(name):
    """环境变量优先，其次读本文件里的同名常量。"""
    v = os.environ.get(name)
    if v:
        return v
    return globals().get(name, "")
