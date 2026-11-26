"""
主智能体 — 灾害监测自动化框架

串联完整流程：
  卫星图像 → VLM灾害识别 → 逆地理编码 → DeepSeek防灾建议 → 输出
"""
import os
import sys
from typing import List, Optional, Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.vlm_inference import DisasterVLM
from inference.geo_coder import GeoCoder
from inference.satellite_api import SatelliteAPI
from agent.llm_advisor import LLMAdvisor


class DisasterAgent:
    """灾害监测智能体"""

    def __init__(
        self,
        vlm_model_path: str,
        lora_path: str,
        deepseek_api_key: str = "",
        geo_provider: str = "nominatim",
        amap_key: str = "",
        satellite_api_key: str = "",
    ):
        print("=" * 55)
        print("  灾害监测智能体初始化 ♪")
        print("=" * 55)
        
        self.vlm = DisasterVLM(vlm_model_path, lora_path)
        self.geo = GeoCoder(geo_provider, amap_key)
        self.satellite = SatelliteAPI(satellite_api_key)
        
        self.advisor = None
        if deepseek_api_key:
            self.advisor = LLMAdvisor(deepseek_api_key)
        
        print("[✓] 智能体初始化完成")

    def analyze(
        self,
        image_paths: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        with_advice: bool = True,
    ) -> Dict:
        """
        完整灾害分析流程。

        Args:
            image_paths: 卫星图像路径列表
            lat: 纬度（可选）
            lon: 经度（可选）
            with_advice: 是否生成防灾建议

        Returns:
            包含识别结果、位置、建议的完整报告
        """
        result = {}
        
        # Step 1: VLM 灾害识别
        print("\n[1/3] VLM 灾害识别...")
        disaster_info = self.vlm.predict(image_paths)
        result["disaster"] = disaster_info
        print(f"  识别结果: {disaster_info[:100]}...")
        
        # Step 2: 逆地理编码定位
        print("\n[2/3] 地理位置定位...")
        location = "未知位置"
        if lat is not None and lon is not None:
            location = self.geo.reverse(lat, lon)
        result["location"] = location
        print(f"  位置: {location}")
        
        # Step 3: DeepSeek 防灾建议
        if with_advice and self.advisor:
            print("\n[3/3] 生成防灾建议...")
            advice = self.advisor.generate_advice(disaster_info, location)
            result["advice"] = advice
            print(f"  建议: {advice[:100]}...")
        else:
            result["advice"] = "（未启用防灾建议）"
        
        return result

    def analyze_from_satellite(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10,
    ) -> Dict:
        """
        从卫星API获取图像并分析（自动流程）。
        """
        print(f"\n[*] 获取卫星数据 ({lat}, {lon}, 半径{radius_km}km)...")
        sat_data = self.satellite.get_latest_image(lat, lon, radius_km)
        
        if not sat_data or not sat_data.get("image_path"):
            print("[!] 未获取到卫星图像，仅返回位置信息")
            return {
                "disaster": "（无卫星图像）",
                "location": self.geo.reverse(lat, lon),
                "advice": "",
            }
        
        return self.analyze([sat_data["image_path"]], lat, lon)

if __name__ == "__main__":
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="灾害监测智能体")
    parser.add_argument("--image", nargs="+", required=True, help="卫星图像路径")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--deepseek-key", default="")
    parser.add_argument("--no-advice", action="store_true")
    args = parser.parse_args()
    
    agent = DisasterAgent(
        vlm_model_path="C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master",
        lora_path="C:/vlmdisaster/outputs/qwen3vl-disaster",
        deepseek_api_key=args.deepseek_key,
    )
    
    result = agent.analyze(
        args.image,
        lat=args.lat,
        lon=args.lon,
        with_advice=not args.no_advice,
    )
    
    print("\n" + "=" * 55)
    print("  灾害监测完整报告")
    print("=" * 55)
    print(f"\n【灾害识别】\n{result['disaster']}")
    print(f"\n【受灾位置】\n{result['location']}")
    print(f"\n【防灾建议】\n{result['advice']}")
    print("=" * 55)

