# Changelog

本文件记录本项目所有值得注意的变更。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

**变更类型**（类别标题只允许这五个）：`新增` / `变更` / `修复` / `移除` / `安全`。

> 本文件自首次公开发布起记账。公开发布之前的内部开发历史不属于本记录范围。

## [未发布]

### 新增

- 英文版首页 `README.en.md`：与中文版互相链接，内嵌英文调度主链图；两版内嵌图
  各自由对应语言的图源守门（`tests/tooling/test_diagram_sources.py`），避免只同步
  一版导致另一版静默展示旧架构。

## [0.3.0] - 2026-11-14

### 新增

- 首次公开发布。范围是**架构设计的最小可运行集**（MVP），不是完整研究平台：
  - **数值主干**：`primitives → wolff_axioms → grains → moves →
    proof_orchestrator → trace_format`（`src/kakeya_ai/`）；
  - **C++ 加速内核**：pybind11 + C ABI（`cpp/`），缺失时自动回退 NumPy；
  - **AI 推理层**：Ollama 客户端、两段式打分、beam search、surrogate 打分器、
    专家路由（`src/kakeya_ai/ai/`）；
  - **调度主链与芯片预算模型**：`nano/{manifest,model,scheduling,runtime}`，
    架构清单为单一真源，生成调度图源；
  - **证明轨迹格式**：带哈希链的 `.kpt`，附 `kakeya-kpt` 命令行复核工具；
  - **架构图**：`docs/diagrams/*.mmd`（Mermaid 源，中英双语）。

### 移除

- 相对内部开发版本，本次公开**不含**：形式化证明骨架、绘图与可视化子系统、
  notebook 与评测数据集、实验性 NLU 子项目、以及指南/书稿/综述等文档产物。
  保留原则是"架构清单与调度主链能跑通、能验证"，其余另行维护。
