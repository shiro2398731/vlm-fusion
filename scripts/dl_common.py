# -*- coding: utf-8 -*-
"""
通用下载工具（requests 版）—— 替代本机损坏的 curl(schannel)。

背景：本机所有 curl.exe 均为 schannel 后端，AcquireCredentialsHandle 失败
     (SEC_E_NO_CREDENTIALS)，无法建立 TLS。requests(OpenSSL) 正常。

提供两个与旧脚本同名的替换：
  curl_get(url, timeout, headers=None)        -> bytes（同旧行为）
  http_download(url, save_path, headers=None,  -> "ok"/"exists"/"fail"
                max_attempts=3, min_size=0, magic=None, resume=True)
"""
import os
import time
import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) vlmdisaster-downloader"


def curl_get(url, timeout=60, headers=None):
    """GET 到内存，返回 bytes（失败返回 b''）。headers 为 ['K: V', ...] 列表。"""
    h = {"User-Agent": UA}
    for item in (headers or []):
        if ":" in item:
            k, v = item.split(":", 1)
            h[k.strip()] = v.strip()
    try:
        r = requests.get(url, headers=h, timeout=(20, timeout))
        if r.status_code == 200:
            return r.content
        return b""
    except Exception:
        return b""


def http_download(url, save_path, headers=None, max_attempts=3,
                  min_size=0, magic=None, resume=True, connect=20, read=600):
    """下载到文件，断点续传 + 重试 + 校验。

    magic: bytes 前缀（如 HDF4 魔数 b'\\x0e\\x03\\x13\\x01'），校验失败视为假文件。
    返回 "ok" / "exists" / "bad" / "fail"
    """
    h = {"User-Agent": UA}
    for item in (headers or []):
        if ":" in item:
            k, v = item.split(":", 1)
            h[k.strip()] = v.strip()

    # 已存在且完整
    if os.path.exists(save_path):
        if _valid(save_path, min_size, magic):
            return "exists"
        try:
            os.remove(save_path)
        except OSError:
            pass

    tmp = save_path + ".part"
    # 清理无效的残留 .part
    if os.path.exists(tmp) and not _valid(tmp, max(1000, min_size), magic):
        try:
            os.remove(tmp)
        except OSError:
            pass

    for attempt in range(1, max_attempts + 1):
        try:
            existing = os.path.getsize(tmp) if (resume and os.path.exists(tmp)) else 0
            req_h = dict(h)
            if existing:
                req_h["Range"] = "bytes=%d-" % existing
            r = requests.get(url, headers=req_h, stream=True, timeout=(connect, read))
            if r.status_code == 404:
                return "fail"
            if existing and r.status_code == 200:
                existing = 0  # 服务端不支持续传，从头写
            mode = "ab" if existing else "wb"
            with open(tmp, mode) as f:
                for chunk in r.iter_content(1024 * 512):
                    f.write(chunk)
        except Exception:
            time.sleep(5)
            continue

        if _valid(tmp, min_size, magic):
            os.replace(tmp, save_path)
            return "ok"
        # 假文件（如登录页）：重试无意义
        if magic and os.path.exists(tmp):
            with open(tmp, "rb") as f:
                if f.read(len(magic)) != magic and os.path.getsize(tmp) > len(magic):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return "bad"
        time.sleep(5)
    return "fail"


def _valid(path, min_size, magic):
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < max(1, min_size):
        return False
    if magic:
        with open(path, "rb") as f:
            if f.read(len(magic)) != magic:
                return False
    return True
