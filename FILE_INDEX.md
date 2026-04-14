# 文件索引

## 顶层目录

| 文件 | 说明 |
|------|------|
| `individual project.txt` | 项目原始要求（老师发的） |
| `individual_project_test.py` | **老师的官方评测脚本**，最终打分用这个跑，不要修改 |
| `GPU_TRAINING_GUIDE.md` | 远程服务器训练指引（SSH / tmux / 4090 配置建议） |
| `RUN_GUIDE.md` | 快速上手运行说明（命令速查） |
| `FILE_INDEX.md` | 本文件 |

---

## code/

### 核心训练与评估

| 文件 | 说明 |
|------|------|
| `train.py` | **主训练脚本**。加载 Llama-2-7B，挂 LoRA，训练，保存 adapter。关键修复：只对 answer token 计算 loss（label masking），覆盖全部 7 个线性层 |
| `evaluate.py` | 训练后评估 train + val 准确率，结果写入 `eval_results.json`，与官方评分逻辑一致 |
| `run_official_test.py` | 复刻官方测试脚本的逻辑，路径通过命令行参数配置（官方脚本路径是硬编码的）。教师发放测试集后，用 `--test_data_path` 传入即可 |
| `utils.py` | 两个工具函数：`load_and_split_dataset`（加载数据集并切分 train/val）、`format_prompt`（构造 prompt 字符串）|

### 流程脚本

| 文件 | 说明 |
|------|------|
| `pipeline.sh` | **单实验完整流程脚本**（推荐入口）。一条命令串联：训练 → 内部评估 → 官方测试逻辑 → 生成报告。全程输出 tee 到 `experiment.log`，SSH 断联不丢日志 |
| `sweep.py` | **多实验炼丹器**。按顺序跑 `EXPERIMENTS` 列表里的所有配置，每个实验独立记日志，最后打印准确率排名表 |
| `run.sh` | 快捷启动脚本。自动建 tmux session，调用 `pipeline.sh`，适合第一次跑 |
| `run_resume.sh` | 断点续训。找到最近的实验目录和 checkpoint，续接训练 |

### 辅助工具

| 文件 | 说明 |
|------|------|
| `make_summary.py` | 为单个实验生成 `results.md`，汇总准确率、超参数、loss、错误预测样例 |
| `requirements.txt` | Python 依赖列表（`pip install -r requirements.txt`） |
| `test_run.py` | 快速冒烟测试，验证环境和模型加载是否正常，不用于正式训练 |

---

## 数据与模型

| 路径 | 说明 |
|------|------|
| `data/dataset.json` | 训练数据集，5000 条 QA，格式 `[{"question": "...", "correct_answer": "..."}]` |
| `model/` | 旧的 adapter 存放目录（v1 遗留）。新版训练结果统一放到 `experiments/` |
| `experiments/<name>/` | 每次实验的完整产物（见下） |

### `experiments/<name>/` 里有什么

| 文件 | 说明 |
|------|------|
| `adapter_config.json` | LoRA 结构配置（rank、alpha、target modules 等）|
| `adapter_model.safetensors` | LoRA adapter 权重，**提交作业只需要这两个文件** |
| `config.json` | 本次实验的训练超参数（由 pipeline 自动保存）|
| `eval_results.json` | train + val 准确率详细结果 |
| `official_test_results.json` | 官方测试脚本逻辑跑出的结果（最重要的指标）|
| `loss_logs.json` | 每步 train loss 和每 epoch val loss 的原始数值 |
| `loss_curve.png` | train + val loss 曲线图（报告用）|
| `timing.json` | 训练开始/结束时间和耗时 |
| `experiment.log` | **完整终端输出**，SSH 断联后查这个 |
| `results.md` | 自动生成的实验总结报告 |

---

## 典型使用流程

```
1. 改 sweep.py 里的 EXPERIMENTS 配置超参数
        ↓
2. python sweep.py --model_path ./Llama-2-7b
        ↓  （每个实验自动跑 pipeline，记日志，出 results.md）
3. 看 experiments/results_summary.json 选最好的模型
        ↓
4. 用 experiments/<best>/adapter_model.safetensors 提交
```
