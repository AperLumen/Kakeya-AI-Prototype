# nano/ —— 纳秒内核与芯片研发

本目录把项目里**围绕"C++ 纳秒内核 + 芯片（加速器）预算/调度"的资产**归在一处。
它们此前散落：`chip_model.py` 在主包 `src/kakeya_ai/` 里（**在 src 里显得突兀**，
它不是几何/AI 运行时，而是**未来芯片研发的判据基础**）；微基准又在
`isolated_nano/`。2026-10-07 按项目方意见合并到本目录。

## 目录内容

本目录按**职责**分四个子包（2026-11-06 整理，原为 5 个文件平铺一层）：

| 子包 | 文件 | 作用 |
| --- | --- | --- |
| [model/](model/) | [chip_model.py](model/chip_model.py) | **芯片预算模型**：把几何内核形式化成可计算的算力/带宽瓶颈判据（roofline + 物理合理性门控 + 调度预测 + 规模→单核墙钟推算 + 能耗拆分 pJ/pair） |
| [scheduling/](scheduling/) | [scheduler.py](scheduling/scheduler.py) | **动态调度器**：按提问拆子问题，逐个子问题决定"调哪些计算模块、用什么资源"，并给出**跨子问题的执行顺序**（拓扑分层 + 核预算裁定并行/串行）。单核墙钟三来源优先级：每模块显式 > 全局标量 > **规模模型推算**（`sizes=...` + `chip_model.estimate_single_core_ms`，由"几管 × 几点"推出）。支持多租户：`plan_query(grant=...)` 用上层授权核数替代芯片拓扑 |
| [runtime/](runtime/) | [resource_arbiter.py](runtime/resource_arbiter.py) | **多租户资源仲裁**：节点级账本（`NodeLedger`）+ 资源授权（`ResourceGrant`）+ **准入队列**（`submit`/`drain`，优先级类 + 释放后重平衡）——把"芯片有多少核"换成"这个租户被授予多少核"，是单租户调度器接入云/多租户的 L1 层 |
| | [prompt_injection.py](runtime/prompt_injection.py) | **提示注入过滤器**（2026-10-31）：给进 LLM 的文本做**廉价粗筛**——复用 `nano_jev` 已验证的匹配机制（英文词边界 + 中文子串），但词表/判定是注入攻击专用（指令劫持/数据外带/能力探测三层）；判 `malicious`/`suspicious`/`benign` 并留可审计证据。守门 `tests/tooling/test_prompt_injection.py`（14 条评估集：注入全拦截、善意零误报含历史误报词） |
| [manifest/](manifest/) | [architecture_manifest.py](manifest/architecture_manifest.py) | **架构清单**：芯片/调度/内核架构的**单一真源**——原则（`PRINCIPLES`）、共享物理量（`SHARED_QUANTITIES`）、调度主链 DAG 数据；渲染 `docs/diagrams/scheduler_flow.mmd`（生成物，手改即被守门判红）。改动前必读该清单顶部的 docstring（含消费协议）；守门 `tests/tooling/test_architecture_manifest.py`（清单↔C++↔Python 三方同源） |
| [isolated/](isolated/) | | **隔离微基准**：纯 C++、无 Python 开销的地面真值（单核延迟 + 多核扫描 + 与 chip_model 对齐的 roofline） |

## 与主包的关系（重要）

* **运行时代码仍在 `src/kakeya_ai/`**：`nano_kernel.py` / `fast_kernel.py` /
  `accel.py` / `cpp_loader.py` / `grains_cpp.py` 是**被主包运行时调用**的加速层，
  它们的家就是主包，**不**迁到本目录。
* **本目录是"研发/判据"侧**：`chip_model.py` 不被任何运行时路径 import（只被
  测试与文档引用），它的读者是**做芯片/加速器决策的人**。故放这里、而不是塞进
  主包。
* C++ 内核本身在仓库根的 `cpp/`（构建脚本 `cpp/build_nano.sh` 决定产物位置），
  本目录的微基准**直接 `#include`/复制**其核心函数，保持"无包依赖"以便在任何
  机器上单编。`cpp/` 亦按职责分两层：`cpp/kernel/`（几何内核与并行切分，
  供 pybind11 扩展使用）与 `cpp/abi/`（C ABI 稳定接口，**不依赖 Python**，
  目标是能在没有 Python 的机器上编译）。

## 用法

```bash
# 芯片预算模型（可运行、可审计）
.venv/bin/python -c "from nano.model.chip_model import main; main()"

# 动态调度器：按提问拆子问题、决定调哪些模块（示例）
.venv/bin/python -c "from nano.scheduling.scheduler import main; main()"

# 纯 C++ 微基准（单核延迟 + 多核扫描 + roofline）
cd nano/isolated
clang++ -O3 -mcpu=apple-m4 -ffast-math -std=c++17 -pthread bench_micro.cpp -o bench_micro
./bench_micro
```

**为什么用 `from nano.model.chip_model import ...` 而不是 `kakeya_ai.chip_model`**：
本目录在仓库根、不在 `src/`，运行它时把**仓库根**加进 `sys.path`（pytest 已按
`scripts` 的同样方式在 `pyproject.toml` 的 `pythonpath` 里声明 `"."`）。这正体现
了"它是研发资产、不是主包的一部分"。

## 守门

* `tests/tooling/test_chip_model.py`：模型自洽 / 单调性 / 口径显式 / 函数化物理合理性门控 / 校准。
* `tests/geometry/test_microbench_integration.py`：微基准与 `chip_model` 共享常量一致 + 真跑验证。
* `tests/orchestration/test_scheduler.py`：动态调度器（分解/模块选择/资源分配/规模感知/执行衔接/
  执行顺序）**与多租户仲裁**（`NodeLedger` 准入/释放、`grant` 注入驱动可调度核数）。
* `tests/orchestration/test_schedule_prediction.py`：调度预测模型（用实测曲线回验、参数拟合）。

详见 [isolated/README.md](isolated/README.md)；预算模型的取值与取舍见
架构清单（`nano/manifest/architecture_manifest.py`）。
