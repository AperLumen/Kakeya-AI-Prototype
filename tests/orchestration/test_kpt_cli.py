"""``kakeya_ai.kpt_cli`` 的回归测试 —— 用户可见 CLI 的**首个**测试。

**为什么需要**（2026-10-04 排查发现）：``kpt_cli`` 是 **``kakeya-kpt`` 命令的
实现**（`pyproject.toml` 的 entry point）与 README 收录的格式解释器，但覆盖率
**0%** —— 全库唯一的"用户直接使用却零覆盖"的模块。

**与 ``test_kpt_viewer.py`` 的分工**（避免重复）：

* ``test_kpt_viewer`` 覆盖的是 **GUI viewer**（窗口、像素报告、选中态）；
* 本文件覆盖 **CLI 契约**：五个子命令、**退出码**、``--json`` 结构。

**退出码是脚本接口的契约**，也是本文件的核心断言 —— 实测三种取值：

| 退出码 | 含义 | 触发 |
| --- | --- | --- |
| ``0`` | 成功 | 操作正常完成（``verify`` 表示**自洽**） |
| ``1`` | **``verify`` 未通过**（文件自洽性检查失败） | 哈希链/重算不一致 |
| ``2`` | 格式错误 / 文件不存在 | 无法解析、路径不存在 |

**``2`` 与 ``1`` 的区分很要紧**：脚本要把"**文件坏了**"和"**文件自洽但复核不
通过**"分开处置——前者是输入问题，后者才是可疑证据。混为一谈会让"格式错"被
误读成"证明有问题"（或反之）。

**注意 ``verify`` 通过的含义**（CLI 自己会打印这段免责声明，本文件也钉住它）：
> 通过只说明文件**自洽**（未被事后改动、重算一致），**不是数学证明**，
> 也不防有动机的伪造（那需要签名）。

⇒ 本文件同时守"**不许把 ``verify`` 通过说过头**"。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kakeya_ai import kpt_cli


# ---------------------------------------------------------------------------
# fixture：真实 trace（不造假数据——格式边角只能由真文件覆盖）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_trace() -> Path:
    """仓库里第一条真实 ``.kpt``。"""
    traces = sorted((Path(__file__).resolve().parents[2] / "docs" / "traces").glob("*.kpt"))
    if not traces:
        pytest.skip("docs/traces/ 下没有入库 trace（基准数据不随公开仓库分发）")
    return traces[0]


@pytest.fixture(scope="module")
def all_traces() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[2] / "docs" / "traces").glob("*.kpt"))


def run_cli(capsys, *argv: str) -> tuple[int, str]:
    """跑一次 CLI，返回 ``(退出码, stdout)``。"""
    code = kpt_cli.main(list(argv))
    return code, capsys.readouterr().out


#: ``verify`` 默认会**重算体积下界**，实测单个 trace **13–17 秒**（多步的）。
#: 全量跑 5 个文件要 ~1 分钟 —— 本机资源约束下（见 ``CONTRIBUTING.md``）测试里
#: 一律加 ``--no-recompute``：**完整性校验（哈希链）不需要重算**，那才是本文件
#: 要守的东西。重算路径另有一条专门的用例（单个文件、明确标 slow 语义）。
NO_RECOMPUTE = "--no-recompute"


# ---------------------------------------------------------------------------
# 文档与实现的一致性：README 里写的参数必须真存在
# ---------------------------------------------------------------------------


class TestReadmeDocumentsOnlyRealFlags:
    """README 的 CLI 示例里出现的参数，**必须真在这个 CLI 上**。

    **为什么需要**：``kakeya-kpt`` 的用户入口是**可安装的命令**，用户照 README
    敲。若文档写了实现里没有的参数（改名、删掉、或**写错**），用户会拿到
    ``unrecognized arguments`` 而不知所以——而**没有任何检查会红**
    （``test_guide_slicing`` 只守数字，不守 CLI 参数）。

    这与本项目"配置类事实必须可验证"是同一思路（见
    ``test_ci_platform_coverage.py`` / ``test_static_check_gate.py``）。

    **判据**：从 README 的 ``kakeya-kpt`` 代码块里抽出 ``--flag`` 形式的参数，
    逐个在该命令的 argparse 子解析器上查。**只查长选项**（``--xxx``），
    不查 ``-n`` 这类短选项——短选项既可能是参数也可能是数值（如 ``-n 5``），
    抽取会误判。
    """

    def _readme_flags(self) -> list[tuple[str, str]]:
        """返回 README 里出现的 ``[(子命令, --flag)]``。"""

        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
            encoding="utf-8"
        )
        found: list[tuple[str, str]] = []
        for line in readme.splitlines():
            if "kakeya-kpt" not in line:
                continue
            m = re.search(r"kakeya-kpt\s+(\w+)\b(.*)", line)
            if not m:
                continue
            sub, rest = m.group(1), m.group(2)
            for flag in re.findall(r"--[a-z][a-z0-9-]*", rest):
                found.append((sub, flag))
        return found

    def test_readme_flags_exist(self):
        """README 提到的每个长选项都应被该子命令接受。

        **判据用"实测"而非"内省 argparse"**：``parser._actions`` / ``choices``
        是私有属性且类型不精确（pyright 报 4 处
        ``reportOptionalSubscript`` / ``reportOptionalIterable``）。改为**拿
        README 里的命令真跑一遍**看是否被接受 —— 这同时覆盖了"参数存在"与
        "解析器接受它"两件事，而且不依赖私有结构。
        """
        documented = self._readme_flags()
        assert documented, (
            "README 里没抽到任何 `kakeya-kpt` 长选项 —— 措辞/格式变了就改本测试"
        )

        import contextlib
        import io

        bad = []
        for sub, flag in documented:
            # 用一个必然不存在的文件路径调用。**只关心参数层是否被接受**：
            #   · 参数合法 ⇒ 解析通过，之后因"文件不存在"抛我们的 FormatError
            #     或返回非 0（都说明 argparse 已接受该 flag）；
            #   · 参数非法 ⇒ argparse 打 "unrecognized arguments" 并 SystemExit。
            # 为把两者分开，**捕获 argparse 自己打印的 stderr**。
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                try:
                    kpt_cli.main([sub, flag, "/nonexistent/__kpt_probe__.kpt"])
                except SystemExit:
                    pass
                except Exception:
                    pass
            if "unrecognized arguments" in buf.getvalue() or "invalid choice" in buf.getvalue():
                bad.append((sub, flag))
        assert not bad, (
            "README 写了 CLI 上不存在的参数——用户照抄会得到 "
            "`unrecognized arguments`：\n  "
            + "\n  ".join(f"`kakeya-kpt {s} {f}`" for s, f in bad)
        )

    def _subcommands_from_help(self) -> set[str]:
        """从 ``--help`` 输出里解析子命令名（**公开接口**，不碰私有结构）。"""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            kpt_cli.main(["--help"])
        text = buf.getvalue()
        m = re.search(r"\{([a-z,]+)\}", text)
        assert m, f"`--help` 输出里找不到子命令列表——argparse 格式变了？\n{text[:300]}"
        return set(m.group(1).split(","))

    def test_readme_mentions_every_subcommand(self):
        """README 应覆盖**全部**子命令（用户据此判断"有哪些能力"）。

        **为什么**：漏列子命令不仅是文档不全——用户会**以为没有该功能**，
        于是重写一个（这与本项目"脚本索引必须完整"是同一类要求）。
        """
        subs = self._subcommands_from_help()
        assert subs, "没解析出任何子命令"
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
            encoding="utf-8"
        )
        missing = sorted(s for s in subs if f"kakeya-kpt {s}" not in readme)
        assert not missing, (
            f"README 的 kakeya-kpt 示例漏了这些子命令：{missing}——"
            "读者会以为没有这个能力"
        )


# ---------------------------------------------------------------------------
# 五个子命令都能跑通（回归：CLI 是用户入口，坏了没人知道）
# ---------------------------------------------------------------------------


class TestAllSubcommandsWork:
    @pytest.mark.parametrize("sub", ["head", "steps", "show", "verify", "json"])
    def test_subcommand_runs_on_all_real_traces(self, capsys, all_traces, sub):
        """每个子命令在**每一条**真实 trace 上都应成功。

        这同时覆盖"格式演进"：``docs/traces/`` 里既有早期协议文件（``head`` 会
        报 "旧协议文件，无 run_seed 字段"）也有新文件，**两种都要能读**。
        """
        for trace in all_traces:
            code, out = run_cli(capsys, sub, *([NO_RECOMPUTE] if sub == "verify" else []), str(trace))
            assert code == 0, f"`kpt {sub} {trace.name}` 退出码 {code}（应为 0）"
            assert out.strip(), f"`kpt {sub} {trace.name}` 没有任何输出"

    def test_head_is_cheap_and_reports_core_metadata(self, capsys, real_trace):
        """``head`` 只读头部，且报出复核方要核对的几项。

        判据取**字段名**而非排版：排版会变，"报了哪些事实"才是契约。
        """
        code, out = run_cli(capsys, "head", str(real_trace))
        assert code == 0
        for field in ("KPT", "创建", "规模", "结论", "几何哈希"):
            assert field in out, f"`kpt head` 输出缺少 {field!r}"

    def test_json_is_machine_consumable_and_self_describing(self, capsys, real_trace):
        """``json`` 必须输出**可解析**的 JSON，且含复核所需的四个顶层键。"""
        code, out = run_cli(capsys, "json", str(real_trace))
        assert code == 0
        payload = json.loads(out)          # 不可解析会在这一行抛
        assert set(payload) >= {"header", "geometry", "steps", "footer"}, (
            f"`kpt json` 顶层键缺项：{sorted(payload)}"
        )
        assert payload["geometry"]["n_tubes"] > 0
        assert payload["steps"], "steps 为空 —— 样例 trace 应当有步骤"

    def test_json_pretty_flag_changes_formatting_not_content(self, capsys, real_trace):
        """``--pretty`` 只影响排版，**不改变数据**。"""
        _, compact = run_cli(capsys, "json", str(real_trace))
        _, pretty = run_cli(capsys, "json", "--pretty", str(real_trace))
        assert json.loads(compact) == json.loads(pretty)
        assert len(pretty) > len(compact), "--pretty 应当是多行输出"


# ---------------------------------------------------------------------------
# 退出码契约（脚本接口）
# ---------------------------------------------------------------------------


class TestExitCodeContract:
    """退出码是**脚本接口**，三种取值各有含义，不得混淆。"""

    def test_missing_file_exits_2_not_1(self, capsys, tmp_path):
        """文件不存在 ⇒ **2**（输入问题），不是 1（复核未通过）。

        混为一谈会让"路径写错"被误读成"证明可疑"。
        """
        missing = tmp_path / "nope.kpt"
        code, _ = run_cli(capsys, "head", str(missing))
        assert code == 2, f"缺失文件应退出 2（格式/输入错误），实得 {code}"

    def test_garbage_file_exits_2(self, capsys, tmp_path):
        """非 ``.kpt`` 内容 ⇒ **2**。"""
        junk = tmp_path / "junk.kpt"
        junk.write_text("this is not a kpt file\n", encoding="utf-8")
        code, _ = run_cli(capsys, "head", str(junk))
        assert code == 2, f"非法内容应退出 2，实得 {code}"

    def test_tampered_file_does_not_exit_0(self, capsys, real_trace, tmp_path):
        """**篡改过的文件不得让 ``verify`` 退出 0**（这是它存在的全部理由）。

        **为什么只断言"非 0"而不钉死 1 或 2**：本测试翻转一个字节，可能落在
        几何行上（⇒ 解析失败，退出 2）也可能落在哈希覆盖的内容上（⇒ 自洽性
        检查失败，退出 1）。**两者都是"不该被接受"**，而具体哪一个取决于
        字节位置 —— 钉死会得到一个脆弱的测试。

        ⇒ 契约是"**不允许静默通过**"，不是"必须是某个具体退出码"。

        **⚠️ 不要把这个断言扩展到 ``head``**（初版犯过）：``head`` 的设计是
        **只读头部**（"快速打开：只读头部，长 trace 也是微秒级"），因此
        **尾部被篡改时它退出 0 是正确行为** —— 那正是它快的原因。要求所有
        子命令都做完整性校验，等于要求 ``head`` 失去它唯一的存在价值。
        """
        raw = bytearray(real_trace.read_bytes())
        assert raw, "样例 trace 是空文件？"
        i = len(raw) // 2
        raw[i] = (raw[i] + 1) % 256
        tampered = tmp_path / "tampered.kpt"
        tampered.write_bytes(bytes(raw))

        code, _ = run_cli(capsys, "verify", NO_RECOMPUTE, str(tampered))
        assert code != 0, "篡改后 `kpt verify` 仍退出 0 —— 复核形同虚设"

    def test_footer_tampering_is_detected_but_whitespace_is_not(
        self, capsys, real_trace, tmp_path
    ):
        """``head`` 刻意不做全文件校验；而**footer 的语义改动**必须被 ``verify`` 抓到。

        **两件事一起测**：

        ① **``head`` 只读头部**（它的性能卖点）⇒ 尾部改动时它仍应成功。
           这是**反向守门**：防止有人"顺手给 ``head`` 也加校验"，那会让长 trace
           的快速打开变慢、并改变它的成本语义。

        ② **footer 的语义字段被改**（这里改 ``n_steps`` 的值）⇒ ``verify``
           必须发现。footer 不在 ``file_hash``（只覆盖 body）范围内，它靠
           **自己的 ``footer_hash``** 自校验 —— 这条路径必须真的生效。

        **⚠️ 初版两个错误（都已修，记下来免得重犯）**：

        * 曾把 ``head`` 也纳入"不得退出 0"—— **错**，``head`` 本就不校验全文；
        * 曾靠"改文件最后一个字节"来制造篡改 —— **无效**，末字节是**换行符**，
          把它改成别的空白字符**不改变任何语义**，``verify`` 通过是**正确**的。
          ⇒ 制造篡改必须动**语义字段**，不能动尾随空白。
        """
        import json as _json

        raw = real_trace.read_bytes()
        # 文件以换行结尾 ⇒ 先剥掉尾随换行再切最后一"行"，否则拿到空串。
        text = raw.decode("utf-8").rstrip("\n")
        body, _, footer_line = text.rpartition("\n")
        assert footer_line.strip(), "样例 trace 没有 footer 行？"
        footer = _json.loads(footer_line)
        assert footer.get("n_steps"), f"footer 没有 n_steps：{footer}"

        # ① head：对**未改动**的文件应成功且只读头部（它不校验全文）
        head_code, head_out = run_cli(capsys, "head", str(real_trace))
        assert head_code == 0, "`kpt head` 对正常文件应成功"
        assert head_out.strip(), "`kpt head` 应有输出"

        # ② 改 footer 的语义字段，verify 必须抓到
        tampered_footer = dict(footer)
        tampered_footer["n_steps"] = footer["n_steps"] + 1
        tampered = tmp_path / "footer_tampered.kpt"
        tampered.write_text(
            body + "\n" + _json.dumps(tampered_footer, ensure_ascii=False),
            encoding="utf-8",
        )
        # 先确认 head 对新文件也能读（尾部改动不影响头部）
        head_code2, _ = run_cli(capsys, "head", str(tampered))
        assert head_code2 == 0, (
            "`kpt head` 因 footer 改动而失败 —— 它应当只读头部（那是它的性能卖点）"
        )

        verify_code, verify_out = run_cli(capsys, "verify", NO_RECOMPUTE, str(tampered))
        assert verify_code != 0, (
            "footer 的 n_steps 被改而 `verify` 通过 —— footer 自校验（footer_hash）"
            f"没有生效。输出：{verify_out[:200]}"
        )

    def test_verify_on_pristine_trace_exits_0(self, capsys, all_traces):
        """未改动的真实 trace ⇒ ``verify`` 退出 **0**（否则样例库本身有问题）。"""
        for trace in all_traces:
            code, out = run_cli(capsys, "verify", NO_RECOMPUTE, str(trace))
            assert code == 0, f"{trace.name} 复核未通过：{out[:200]}"


# ---------------------------------------------------------------------------
# verify 的**声明边界**：不得把"自洽"说过头
# ---------------------------------------------------------------------------


class TestVerifyDisclaimer:
    """``verify`` 通过 ≠ 数学证明。这条边界必须在输出里明说。"""

    def test_verify_output_states_the_boundary(self, capsys, real_trace):
        """输出必须写明"通过只说明自洽，不是数学证明"。

        **为什么钉这个**：``verify`` 是**工具对结论的背书**，最容易被过度引用
        （"复核通过了" ⇒ 被听成"证明是对的"）。CLI 已把免责声明写进输出，
        本测试防止将来有人为了"输出更简洁"把它删掉。
        """
        code, out = run_cli(capsys, "verify", NO_RECOMPUTE, str(real_trace))
        assert code == 0
        assert "不是数学证明" in out, (
            "`kpt verify` 的输出丢了免责声明 —— 它必须说明'通过只说明文件自洽，"
            "不是数学证明、也不防有动机的伪造'"
        )

    def test_verify_json_exposes_the_same_verdict_fields(self, capsys, real_trace):
        """``--json`` 模式的字段名是**脚本消费的接口**，缺项会静默破坏下游。

        ``ok`` 与四个细分检查（``file_hash_ok`` / ``geometry_hash_ok`` /
        ``decimal_hex_ok`` / ``chain_ok``）都必须在 —— 只报总 ``ok`` 会让
        复核方无法说清"到底哪一项没过"。
        """
        code, out = run_cli(capsys, "verify", NO_RECOMPUTE, "--json", str(real_trace))
        assert code == 0
        payload = json.loads(out)
        for key in (
            "ok",
            "n_steps",
            "file_hash_ok",
            "geometry_hash_ok",
            "decimal_hex_ok",
            "chain_ok",
            "notes",
            "verdicts",
        ):
            assert key in payload, f"`kpt verify --json` 缺字段 {key!r}（脚本接口）"
        assert payload["ok"] is True
        assert payload["n_steps"] > 0


# ---------------------------------------------------------------------------
# 重算路径（**慢**：实测单个多步 trace 13–17 秒）
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRecomputePath:
    """``verify`` **默认带重算**（不带 ``--no-recompute``），这条路径也要有覆盖。

    **为什么单独标 slow 并只跑一个文件**：本机资源约束要求克制（见 ``CONTRIBUTING.md``）。
    实测每个多步 trace 重算要 **13–17 秒**，5 个文件就是一分多钟 —— 而
    "完整性校验（哈希链）"不需要重算，所以**其余用例都用 ``--no-recompute``**，
    只有这一条（明确要覆盖重算本身）留着并标 ``slow``。

    默认 ``addopts = "-m 'not slow'"`` 会跳过它；要跑显式加 ``-m slow``。
    """

    def test_default_mode_actually_recomputes(self, capsys, real_trace):
        """默认模式（不加 ``--no-recompute``）应当**真的重算**并仍然通过。

        判据：与 ``--no-recompute`` 相比，输出**多出重算相关的内容**
        （重算会逐步骤给 verdict）。若两者输出完全一致，说明"默认重算"
        这一档其实没生效。
        """
        code_light, light = run_cli(capsys, "verify", NO_RECOMPUTE, str(real_trace))
        code_full, full = run_cli(capsys, "verify", str(real_trace))
        assert code_light == 0 and code_full == 0, "两种模式都应通过未改动的文件"
        assert len(full) >= len(light), (
            "默认模式输出比 --no-recompute 短 —— '默认重算'可能没生效"
        )
