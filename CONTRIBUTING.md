# 参与贡献

本项目当前是**单人维护的研究/工程混合体**，欢迎贡献，但请先读完本文件——本仓库有
若干与"常规开源项目"不同的硬性约定。

## 改动前必读

- 改代码行为必须同步改对应文档；注释与 docstring 用中文（标识符、路径、
  命令、模型名、数学符号除外）。
- 改 `cpp/`、`nano/` 或调度相关代码前，先读架构清单
  [`nano/manifest/architecture_manifest.py`](nano/manifest/architecture_manifest.py)
  顶部的 docstring（含消费协议与守门要求）。

> **本机资源约束**：本项目在资源有限的单机上开发——本机**一次最多跑两个进程**、
> **不在本机跑全量测试**、**一次最多跑两个测试文件**。全量验证交给 CI
> （GitHub runner 上没有这条约束）。

## 开发环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# 构建 C++ 加速内核（可选，但强烈建议；缺失时自动回退 NumPy）
bash cpp/build_nano.sh
```

AI 相关功能需要本地 [Ollama](https://ollama.com) 及模型：

```bash
ollama pull qwen3.5:4b
```

**不需要 Ollama 也能跑测试**：所有依赖模型的用例要么 `skip`，要么打桩。

同理，**surrogate 打分器权重不随仓库分发**（`data/surrogate_scorer_weights.json`
是本机的训练产物）。没有它时编排器会静默退回启发式，于是"surrogate 选 Move"相关的
守门用例会 `skip` 并写明原因——那不是缺陷，是前提不成立。

## 测试

> 注意：不要在本机跑全量（`pytest tests/`）——全量一次数分钟且吃满内存。
> 按需跑与改动相关的测试文件，一次最多两个；全量验证由 CI 承担。

```bash
.venv/bin/python -m pytest tests/orchestration/test_trace_format.py tests/orchestration/test_moves.py -q
```

慢测试用 marker 隔离，默认跳过：

```bash
.venv/bin/python -m pytest tests/geometry/test_fast_kernel.py -q -m "not slow"
```

CI 上跑全量是允许的（在 GitHub 的机器上，不占本机资源）。

### 覆盖率

分两个口径，**各有各的用**：

**(1) 全局**——回答"整体达标没有"。这个数字**只能在 CI 得到**（本机禁止全量，
见上文资源约束）：CI 的 `test` job 在 `ubuntu/py3.12` 那条 leg 上带
`--cov-fail-under=78`。门槛依据（实测定值与余量）写在 `pyproject.toml` 的
`[tool.coverage]` 注释里。

**(2) 按模块**——回答"**哪些核心路径**没被测到"：本机对单个模块跑它相关的
测试文件（一次最多两个）并带 `--cov=src/kakeya_ai`，看未覆盖行即可。

## 提交前自查

```bash
ruff check .                                     # 静态检查（全域，应 0 告警）
ruff format --check $(git diff --name-only -- '*.py')   # 格式检查（仅本次改动的 .py）
```

`ruff format` **只对本次改动的文件跑**：存量代码从未统一格式化过（约 140 个 `.py`
文件），对全域跑 `--check` 必然失败，而全量重格式化会制造淹没真实改动的巨大 diff
（违反「改动原则」第一条）。pre-commit 的 `ruff-format` 钩子用的是同一口径
（`pass_filenames: true`），本地自查照抄它即可。

**类型检查（可选，pyright）**——`ruff` 不做类型检查，类型错误交由 pyright：

```bash
uvx pyright                                 # 用仓库根的 pyrightconfig.json
uvx pyright src/kakeya_ai/foo.py --pythonpath .venv/bin/python   # 查单个文件
```

- **`--pythonpath .venv/bin/python` 必须给**，否则解析不到 `numpy` 等依赖。
- 路径与检查模式由仓库根的 **`pyrightconfig.json`** 统一（`extraPaths` 含
  `src` 与 `scripts`，与 `pyproject.toml` 的 `pytest` `pythonpath` 一致）——
  这样命令行与 VS Code（Pylance）**判据相同**。
- **编辑器的红字不可全信**：Pylance 是懒分析，**只分析已打开或被 import 到的
  文件**，未被打开的文件即使有错也报"无问题"。要全库确认必须用上面的命令。
- **CI 会跑它**（`ci.yml` 的 `lint` job，非零即失败），且
  `tests/tooling/test_static_check_gate.py` 守门——**改了 CI 配置把这一步删掉会红**。
  本机跑不了全量（资源约束）不影响：这一步在 CI 的机器上跑。
- **当前基线是 0 errors / 0 warnings**（2026-10-02，从 181/3 清起）。
  改完代码跑一次，**别把报错加回去**。

pre-commit 钩子已配置（`pre-commit install` 装上）：提交时自动跑 lint 与快速检查。

**架构图改动**（改了 `docs/diagrams/*.mmd` 时）：`docs/diagrams/` 是**唯一真源**，
GitHub 与编辑器直接渲染，**没有需要同步的产物**。两点要守：

- **双语成对**：`xxx.mmd` 必须有 `xxx_en.mmd`（`tests/tooling/test_diagram_sources.py`）；
- **调度主链图由清单生成**：`scheduler_flow.mmd` 是
  `nano/manifest/architecture_manifest.py` 的渲染产物，**不要手改**
  （`tests/tooling/test_architecture_manifest.py` 会判红）。

需要 PNG（例如贴进幻灯片）时再跑 `bash scripts/docs_build/render_diagrams.sh`
——它需要 Node + mermaid-cli，属**可选**工具，产物不入库。

## 改动原则

1. **改动最小化**：只动任务相关的文件，不顺手重构、不重格式化无关代码。
2. **AI 协作者不做 git mutation**：AI 助手不要代为执行 `commit` / `push` / `rebase`，
   除非维护者明确要求。提交信息按 `type(scope): 简述`、正文写"为什么"；
   源文件与生成物分开提交。
3. **不提交构建产物**：`.venv/`、`__pycache__/`、`*.so`（已入库的除外）、`.DS_Store`。
4. **可复现**：数值结论必须附生成它的脚本与命令。改代码行为必须同步改文档。

## 许可与贡献授权

本项目采用**双许可**（见 [`LICENSING.md`](LICENSING.md)）：学术/非商业场景适用
AGPL-3.0，商业使用需单独授权。

**提交贡献前请注意**：提交即表示你同意（1）你的贡献以 AGPL-3.0 授权，且
（2）项目方可将它纳入商业授权版本。双许可模式下，项目方必须对全部代码拥有
**再许可权**，否则无法对外提供商业授权——这是该模式的必要条件，不是可选项。

若你不接受第（2）点，请不要提交贡献（可以改为提 issue 描述需求，由维护者实现）。

## 不要破坏的三条不变量

这三条改动前务必先理解：

1. **域不变**：体积估计的默认盒 `[-1,1]³` 是**可复核契约**——已入库 trace 的数值都在它
   下面算过，且 ±1.0 与 2 的幂网格在 float32/float64 下精确，使跨后端**逐位一致**。
   **绝不静默修改默认域**。支持新场景请用显式、写进文件、带版本的域参数。
2. **数值可比**：改默认参数前先问"谁的历史产物依赖它"。
3. **跨后端逐位同**：`numpy` / `fast` / `nano` 三个后端对同一输入必须给出逐位相同的结果。

## 关注点分离（方向层 vs 执行层）

项目定调：**整体方向由 LLM 决策，具体执行与复筛由纯函数完成**。判据是
"确定性可复核 vs 探索性"，不是快慢：

- 所有写进 `.kpt` 哈希链的定量结论**必须出自纯函数**；
- LLM 输出**不得**直接成为 trace 中的数字。

新增功能时请明确自己站在哪一层。

## 提交信息

中文或英文皆可，建议 `type(scope): 简述` 形式（如 `fix(sticky): 修正尺度链假阳性`）。
正文说明**为什么**改，而不只是改了什么。

## 发布流程

包版本有**四个落点**，发版时必须一起改。全部由
`tests/tooling/test_release_consistency.py` 守门（跑一次即知是否漏改）：


| 落点                       | 内容                    | 怎么改                    |
| -------------------------- | ----------------------- | ------------------------- |
| `src/kakeya_ai/version.py` | `__version__`           | **唯一真源**，改这里      |
| `pyproject.toml`           | `dynamic = ["version"]` | 不用改（动态读取真源）    |
| `CITATION.cff`             | `version:`              | 手工同步（守门会查）      |
| `CHANGELOG.md`             | `## [X.Y.Z] - 日期`     | 把`[未发布]` 改名并填日期 |

**发版清单**（照做即可）：

- [ ] 改 `src/kakeya_ai/version.py` 的 `__version__`（唯一真源）
- [ ] 同步 `CITATION.cff` 的 `version:` 字段
- [ ] `CHANGELOG.md`：把 `[未发布]` 改名为 `[X.Y.Z] - 日期`，并在顶部重开 `[未发布]` 节
- [ ] 跑 `tests/tooling/test_release_consistency.py` 确认四处一致
- [ ] 提交后打 tag `vX.Y.Z`（由项目方执行）
- [ ] `python -m build` 构建 sdist/wheel，`tar tzf dist/*.tar.gz` 抽查内容

> **为什么这四者此前会漂**：`CITATION.cff` 自己的注释就写着"发布时同步更新"
> ——**靠人记得**。本项目已多次证明这类约定必然失效（survey 的统计数字、
> "诚实标注"都漂过），每次都是补守门才止住。版本号是同类问题里后果最重的一个：
> 引用元数据的版本错了，别人按它引用的就是错的，且**不会以任何方式报错**。

## 报告问题

- Bug / 功能请求：开 issue，附复现命令与最小输入。
- **安全问题：请勿开公开 issue**，见 `SECURITY.md`。
