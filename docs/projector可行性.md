# projector 嵌入级融合：可行性记录

## 结论（当前）

- 显存：**可行**。8078 服务运行中（约 4.1GB）再加载一份 4bit Qwen3-VL-4B，
  成功且剩余 1.68GB（见 `scripts/projector_memtest.py` 实测）。
- 训练链路：**未实现**。需要把 CLIP/ResNet 向量经 projector 变成额外
  token 拼进 Qwen 输入序列，并做小规模训练；当前仓库只有输入信息融合（文本）。
- 决策：暂不投入。当前决策级融合在部署口径已达到 baseline 76/153→125/153、
  hold 集 12 升 2 平 0 降；projector 属“完全体”而非必要项。

## 若要实现，最小方案

1. 取每样本 CLIP/ResNet 向量（已有探针，向量现成）
2. projector：Linear(d, k*H) → 展成 k 个 token（H 为 Qwen 隐藏维度）
3. 在语言模型 embedding 层前拼接 k 个投影 token（hooks 或 inputs_embeds）
4. 冻结 Qwen 与 LoRA，只训练 projector（每元素几百条样本）
5. 用 teacher forcing 在元素训练 JSON 上做 SFT 式损失
6. 验证：同分布 + hold/外部日期，与决策级融合对比

## 风险

- 需要改 transformers 前向（视觉 token 与文本 token 合并时机）
- 8GB 下 batch 只能 1，训练时间长
- 语义支路文本注入已被证明会拖累，projector 是否真能更好需实验验证

## 实测补充（2026-11-28 深夜）

- 服务运行中再加载一份 4bit 基座：成功，剩余 1.68GB（`scripts/projector_memtest.py`）
- 对该第二份模型做一次前向+反向（文本样本）：峰值显存 5.28GB，
  同时服务在跑时 free 变成 0.00GB，处于临界
- 结论：**projector 训练必须先停 8078 服务**；单独运行时 5.28GB 峰值在 8GB 内可行
- 训练链路仍未实现，预计单元素（coral）原型需要：
  停服 → 加载 4bit → 构造 projector 与 token 拼接 → 数十步 SFT → 恢复服务
