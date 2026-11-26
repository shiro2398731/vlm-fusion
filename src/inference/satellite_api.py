"""
卫星数据API模块 — 对接吉林一号开放平台

实际使用时需要注册获取 API Key:
  https://developer.jl1.cn/

本模块提供接口框架，支持两种模式：
1. 真实API模式（需API Key）
2. 本地模拟模式（用于测试，读取本地卫星图像）
"""
import os
from typing import Optional, Dict
import requests


class SatelliteAPI:
    """吉林一号卫星数据接口"""

    def __init__(self, api_key: str = "", base_url: str = "https://developer.jl1.cn/api"):
        self.api_key = api_key
        self.base_url = base_url

    def get_latest_image(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10,
    ) -> Optional[Dict]:
        """
        获取指定区域的卫星图像。

        Args:
            lat: 中心纬度
            lon: 中心经度
            radius_km: 搜索半径（公里）

        Returns:
            {
                "image_path": 图像本地路径,
                "lat": 纬度,
                "lon": 经度,
                "timestamp": 拍摄时间
            }
        """
        if not self.api_key:
            return self._mock_image(lat, lon)
        return self._real_api(lat, lon, radius_km)

    def _real_api(self, lat: float, lon: float, radius_km: float) -> Optional[Dict]:
        """真实API调用（需根据吉林一号实际接口调整）"""
        url = f"{self.base_url}/satellite/search"
        params = {
            "key": self.api_key,
            "lat": lat,
            "lon": lon,
            "radius": radius_km,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "image_path": data.get("image_url", ""),
                    "lat": lat,
                    "lon": lon,
                    "timestamp": data.get("timestamp", ""),
                }
        except Exception as e:
            print(f"[!] 卫星API调用失败: {e}")
        return None

    def _mock_image(self, lat: float, lon: float) -> Dict:
        """本地模拟模式（测试用）"""
        return {
            "image_path": "",
            "lat": lat,
            "lon": lon,
            "timestamp": "mock",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()
    
    api = SatelliteAPI()
    result = api.get_latest_image(args.lat, args.lon)
    print(result)

