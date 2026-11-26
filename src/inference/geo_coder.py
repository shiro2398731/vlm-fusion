"""
逆地理编码模块 — 经纬度 → 地名

支持 Nominatim（免费）和 高德（需API Key）。
"""
from typing import Optional
import requests


class GeoCoder:
    """经纬度逆地理编码"""

    def __init__(self, provider: str = "nominatim", amap_key: str = ""):
        self.provider = provider
        self.amap_key = amap_key

    def reverse(self, lat: float, lon: float) -> str:
        """
        将经纬度转换为地名。

        Args:
            lat: 纬度
            lon: 经度

        Returns:
            地名描述（如 "四川省绵阳市..."）
        """
        if self.provider == "amap" and self.amap_key:
            return self._amap_reverse(lat, lon)
        return self._nominatim_reverse(lat, lon)

    def _nominatim_reverse(self, lat: float, lon: float) -> str:
        """Nominatim 免费逆地理编码"""
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10,
            "accept-language": "zh-CN",
        }
        headers = {"User-Agent": "DisasterMonitor/1.0 (academic project)"}
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("display_name", "未知位置")
        except Exception as e:
            print(f"[!] Nominatim 逆地理编码失败: {e}")
        
        return f"纬度{lat:.4f}, 经度{lon:.4f}"

    def _amap_reverse(self, lat: float, lon: float) -> str:
        """高德逆地理编码（需API Key）"""
        url = "https://restapi.amap.com/v3/geocode/regeo"
        params = {
            "key": self.amap_key,
            "location": f"{lon},{lat}",
            "extensions": "all",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "1":
                    return data["regeocode"]["formatted_address"]
        except Exception as e:
            print(f"[!] 高德逆地理编码失败: {e}")
        
        return f"纬度{lat:.4f}, 经度{lon:.4f}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()
    
    gc = GeoCoder()
    print(gc.reverse(args.lat, args.lon))

