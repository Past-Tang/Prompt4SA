# Prompt4SA - VQA-SA 复现（MARS2 @ ICCV 2025）

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![vLLM 0.8.5.post1](https://img.shields.io/badge/vLLM-0.8.5.post1-ff69b4.svg)](https://docs.vllm.ai/)
[![MS-SWIFT](https://img.shields.io/badge/ms--swift-3.x-success.svg)](https://github.com/modelscope/ms-swift)
[![Platform Linux](https://img.shields.io/badge/Platform-Linux-lightgrey.svg)](https://kernel.org)



本目录用于提交与复现我们在 MARS2 竞赛 VQA-SA 赛道的代码与说明。核心脚本为 `main.py`，使用 MS-SWIFT (`swift.llm`) + vLLM 推理引擎进行多模态视觉问答（空间意识）。


## 目录结构
```
up_github/
  main.py                  # 批量推理与可视化主脚本
  src/
    NotoSansSC-Regular.otf # 中文可视化字体
  model/
    README.md              # 放置本地权重的说明
  data/                    # 请在本地创建（不提交仓库）
    images/                # 图像文件夹（与 JSON 中 image_path 对应）
    VQA-SA-question.json   # 评测问题文件
  requirements.txt         # 运行依赖
  run.sh                   # 运行脚本（Linux）
  .gitignore               # 忽略大文件/私有数据
```

## 环境准备（Linux）
- 建议使用 Conda：
```bash
conda create -n mars2-vqa python=3.10 -y
conda activate mars2-vqa
```
- 安装依赖：
```bash
pip install -r requirements.txt
```
- 关于 PyTorch/CUDA：请根据你的 CUDA/驱动环境安装匹配版本的 PyTorch；`vllm` 需要与 CUDA 版本兼容。若未安装 `torch`，请参考官方指引安装对应版本。

提示：MS-SWIFT 官方推荐 `vllm==0.8.5.post1`，本仓库已固定。

## 数据与模型
- 本地数据放置（请勿提交）：
  - 图像目录：`up_github/data/images/`
  - 问题文件：`up_github/data/VQA-SA-question.json`
- 在 `main.py` 中设置数据路径（任一方式）：
  1) 直接设置常量（推荐）
     ```python
     VQA_DATA_PATH = 'data/VQA-SA-question.json'
     ```
  2) 保持现有 JSON 的 `image_path`，但将其前缀改为 `data/images/...`（若原始相对路径不同）。
- 模型权重：默认路径见 `main.py` 顶部常量 `MODEL_PATH`，默认为：
  - `/home/tang/workshop/model/InternVL3-78B`
  如路径不同，请修改 `main.py` 中的 `MODEL_PATH` 或将权重放入 `up_github/model/InternVL3-78B/` 并按需调整。

## 运行
- 方式一：脚本
```bash
conda activate mars2-vqa
bash run.sh
```
- 方式二：直接执行
```bash
conda activate mars2-vqa
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

运行完成后：
- 聚合结果 JSON：`VQA-SA-results.json`
- 可视化 PNG：`InternVL3-output/` 下按图片名导出

说明：`main.py` 内已启用多线程可视化与中文字体渲染；当无结果可视化时会自动跳过；并保留外部 `CUDA_VISIBLE_DEVICES` 优先级。

## 提交到 EvalAI（可选）
- 安装 CLI：`pip install evalai`
- 非交互示例（请按需修改 ID 与公开/私密标记）：
```bash
printf "n\n" | evalai challenge 2552 phase 5069 submit \
  --file VQA-SA-results.json --large --public
```
或在 `main.py` 中将 `SUBMIT_TO_EVALAI=True` 并设置 `CHALLENGE_ID/PHASE_ID` 后直接自动提交。

## MARS2 / VQA-SA 赛道背景（简述）
- Workshop：Multimodal Reasoning and Slow Thinking in Large Model Era (System 2)
- 赛道：VQA-SA（Visual Question Answering with Spatial Awareness）
- 目标：评测空间/常识/反事实推理能力
- 时间线（参考公告）：提交截止 2025-08-05；优胜公布 2025-10-20

## 复现清单（建议）
- [ ] 在本地创建 `up_github/data/images/` 与 `up_github/data/VQA-SA-question.json`
- [ ] 确保 JSON 中 `image_path` 与本地图片路径匹配（推荐前缀 `data/images/`）
- [ ] 准备本地模型权重并设置 `MODEL_PATH`
- [ ] 创建并激活 Conda 环境（Linux）
- [ ] `pip install -r requirements.txt`
- [ ] 运行 `bash run.sh` 或 `python main.py`
- [ ] 检查 `VQA-SA-results.json` 与可视化 PNG
- [ ]（可选）安装 `evalai` 并提交评测

## 致谢
- 本项目使用 MS-SWIFT（`ms-swift`）与 vLLM 推理引擎。
- 感谢 MARS2 组委会与 ICCV 2025 Workshop 提供数据与赛题。 