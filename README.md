# Trip-RL

面向多约束旅行规划的长程工具调用 Agent 后训练项目。在 TravelPlanner 环境中，通过城市、航班、住宿、餐饮、景点和距离查询工具完成多轮规划，并用硬约束与常识约束评估最终行程。

## 当前代码

- **数据构造与评估**：`DataSynthesis/` 提供约束采样、可行性检查和问题生成；`Inference/` 提供多轮推理、成功轨迹生成、行程解析与约束评估。
- **强化学习与奖励塑形**：`RL/agentstar_travel_loop.py` 实现分阶段课程奖励、工具交互与失败轨迹处理；`RL/rllm/rewards/` 实现任务评估及严格行程转录。
- **异步策略优化与在线策略蒸馏（OPD）**：`RL/agentstar_async_travel_loop.py` 对接 veRL V1 异步训练，记录策略版本、处理多轮 action-only history 与逐轮教师信号，并对基础设施失败样本进行掩码处理。
- **当前训练入口**：`RL/examples/travel/train_qwen35_curriculum_opd_async.py`。它依赖同目录的同步基础入口，两者均保留。
- **后端修改**：`patches/verl-v0.9.1.patch` 保存实际使用的 veRL 修改，包括多轮训练、蒸馏损失和指标统计。未携带上游旧版内嵌 veRL。

## 运行前准备

本仓库是 A800 上最新代码的筛选快照，保留原有算法和配置。配置及部分检查脚本仍使用服务器上的绝对路径；迁移运行前需替换模型、数据、后端、TransferQueue 依赖与输出目录。不能直接把现有配置当作通用开箱即用环境。

1. 数据与工具数据库参考 [Agent-STAR 数据集](https://huggingface.co/datasets/xxwu/Agent-STAR-TravelDataset) 和 [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)。本仓库不上传训练数据或数据库 CSV。
2. veRL 后端固定为 `1876b06d0a3e4e71e06230be10af14492ca8a75b`，在对应源码根目录执行 `git apply /path/to/Trip-RL/patches/verl-v0.9.1.patch`。不要使用原项目的旧版内嵌后端。
3. 异步流程使用 TransferQueue 0.1.9，源码提交 `c51614308b68c8d7a87c9b3ef62d59e14c69bde2`。`RL/pyproject.toml` 是保留的 rLLM 包声明；实际核心环境版本见 `environment-versions.json`。
4. 修改 `RL/examples/travel/qwen35_curriculum_opd*.yaml` 及训练入口中的本地路径；API 凭据通过环境变量或仓库外的密钥文件配置。`Inference/config.json` 仅含空凭据。
5. 先运行训练入口的 `--prepare-only` 和 `RL/scripts/` 中相应 CPU 检查，再根据实际 GPU 资源启动训练。部分检查依赖真实本地 tokenizer 或数据，不能在缺少这些资源时视作已通过。

旧 Qwen2.5 训练配方、旧双卡独立 SFT 配方、备份、缓存、日志、权重与历史输出未纳入发布。当前 SFT 训练迁移采用独立的 `train_test` 工作流，不将被替代的旧配方冒充最新训练方案。

## 来源与验证范围

项目基于 [Agent-STAR](https://github.com/WxxShirley/Agent-STAR)、[rLLM](https://github.com/rllm-org/rllm) 和 [veRL](https://github.com/volcengine/verl)。原项目说明与论文引用保存在 `UPSTREAM_README.md`；rLLM 许可证见 `RL/LICENSE`，veRL 许可证见 `patches/VERL_LICENSE`。这些许可证分别适用于对应上游组件，本仓库未擅自为全部来源代码重新声明统一许可证。

`SOURCE_MANIFEST.json` 记录上游提交及导出源文件的 SHA-256。发布检查包括 Python 语法、源文件一致性、后端补丁与凭据模式扫描；这些检查不代表重新完成 GPU 训练或基准评测，本仓库不声明未经核验的个人实验成绩。
