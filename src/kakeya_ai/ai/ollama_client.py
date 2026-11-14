"""AI 推理层的最小 Ollama 客户端。

使用 Ollama HTTP API（localhost:11434），只依赖 Python 标准库（`urllib`、`json`）。

故障转移（P2#9）：把 ``OLLAMA_HOSTS`` 环境变量设为逗号分隔的 base URL 列表
（``http://host:11434`` 或完整的 ``.../api/generate`` 端点都接受；缺少时会追加
``/api/generate``），即可把负载分散到多个 Ollama 实例。客户端维护一个轮询
游标：每次调用从游标所在的 host 开始，在该 host 上耗尽 ``max_retries`` 次瞬时
失败重试后，按顺序移到下一个 host；成功后游标前移，使下次调用从下一个 host
开始。当所有 host 都失败时抛 ``RuntimeError``。若 ``OLLAMA_HOSTS`` 未设置，
行为与单 ``url`` 的历史行为完全一致。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, Optional, Union

from .observability import default_observability, get_logger, observe_llm_call
from .token_usage import TokenUsage
from ..errors import InputValidationError
from ..resource_budget import DEFAULT_BUDGET


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# 默认值统一取自 ``kakeya_ai.resource_budget``（超时/重试的**单一真源**）。
#
# **为什么保留这两个常量名**：它们被测试与外部代码引用（``DEFAULT_MAX_RETRIES``
# 等），删除会破坏向后兼容。数值改为动态取自预算对象，因此环境变量
# （``KAKEYA_MAX_RETRIES`` / ``KAKEYA_RETRY_BACKOFF``）现在能真的影响到这里——
# 此前它们只是硬编码字面量，改不了。
DEFAULT_MAX_RETRIES = DEFAULT_BUDGET.max_retries
DEFAULT_RETRY_BACKOFF = DEFAULT_BUDGET.retry_backoff

#: 单次 HTTP 请求超时（秒）。取自资源预算，环境变量 ``KAKEYA_HTTP_TIMEOUT``
#: 可覆盖。模型推理可能很慢（deep 模型本地跑可到数十秒），所以默认给得很宽。
DEFAULT_HTTP_TIMEOUT = DEFAULT_BUDGET.http_timeout

#: ``is_server_ready`` 的**探活**超时（秒）。
#:
#: **刻意不跟主超时走**：探活只问"服务在不在"，一个 TCP 往返就够；
#: 用 300 s 会让"Ollama 没起"这种最常见的情况干等 5 分钟才报错。
#: 这是个固定的小值，不是预算项（预算管的是"任务能花多久"，不是"探测多快"）。
PROBE_TIMEOUT = 5.0

# 逗号分隔的 Ollama base URL；未设置 = 单个 DEFAULT_OLLAMA_URL。
OLLAMA_HOSTS_ENV = "OLLAMA_HOSTS"

# 值得重试的异常（瞬时网络/服务端问题）。
_RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionResetError, BrokenPipeError)

#: 默认上下文窗口上限。实测本机 Ollama（mlx runner）对 4B 模型默认给
#: ``num_ctx=262144``，KV cache 随上限预分配，是内存占用大头（M 系芯片统一
#: 内存，模型权重 + KV 全算系统 RAM；16 GB 机器上轮询 deep 时会被打爆）。
#: 本项目 prompt 只有几百 token，4096 绰绰有余。调用方在 ``options`` 里显式
#: 传 ``num_ctx`` 可覆盖。
DEFAULT_NUM_CTX = 4096


def _with_default_num_ctx(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """合并默认 ``num_ctx``；调用方显式给出的值优先。"""
    merged: Dict[str, Any] = {"num_ctx": DEFAULT_NUM_CTX}
    merged.update(options or {})
    return merged


def _should_retry(exc: BaseException) -> bool:
    """若该异常看起来是瞬时的、重试可能有帮助，则返回 True。"""
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        # 没有 reason 的 URLError（例如 HTTP 层失败）可重试；
        # 有 reason 时按其类型判断。
        return reason is None or isinstance(reason, _RETRYABLE_EXCEPTIONS)
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)

_logger = get_logger(__name__)
_obs = default_observability()

# OLLAMA_HOSTS 列表上的轮询游标。本模块按设计是单进程客户端；游标非线程安全，
# 与模块其余部分保持一致。
_host_cursor = 0


def _resolve_hosts(url: str) -> list:
    """返回要尝试的有序 generate 端点 URL 列表。

    环境变量 ``OLLAMA_HOSTS``（逗号分隔的 base URL 或完整 ``/api/generate``
    端点）会替换默认值；每个条目都会归一化为以 ``/api/generate`` 结尾。未设置
    时原样返回 ``[url]``。读取经由 ``kakeya_ai.config``，使该变量的类型/默认值
    只在一处声明。
    """
    from ..config import as_csv

    parts = as_csv(OLLAMA_HOSTS_ENV)
    if not parts:
        return [url]
    hosts = []
    for part in parts:
        part = part.rstrip("/")
        if part.endswith("/api/generate"):
            hosts.append(part)
        else:
            hosts.append(part + "/api/generate")
    return hosts or [url]


def _advance_host_cursor(idx: int, n: int) -> None:
    """把轮询游标移到 ``idx`` 之后的那个 host。"""
    global _host_cursor
    _host_cursor = (idx + 1) % n


def _connection_error(exc: BaseException, url: str, max_retries: int) -> BaseException:
    """把放弃时的异常转换为历史上的 RuntimeError 消息。"""
    if _should_retry(exc):
        return RuntimeError(
            f"Could not reach Ollama API at {url} after "
            f"{max_retries} retries. Is the server running? ({exc})"
        )
    if isinstance(exc, urllib.error.URLError):
        return RuntimeError(
            f"Could not reach Ollama API at {url}. Is the server running? ({exc})"
        )
    return exc


def _extract_ollama_tokens(result: Any) -> tuple:
    """从 Ollama 响应 dict 中取出 (prompt_tokens, completion_tokens)。"""
    if isinstance(result, dict):
        return (
            int(result.get("prompt_eval_count") or 0),
            int(result.get("eval_count") or 0),
        )
    return (0, 0)


def _validate_format(format: Optional[Union[str, Dict[str, Any]]]) -> None:
    """把**后端根本不吃**的 ``format`` 值挡在发请求之前。

    为什么值得一个专门的校验：Ollama 对不认识的 ``format`` 回 **HTTP 500**
    （``expected "json" or a valid JSON Schema object``）。500 是"请求本身非法"，
    但对调用方它和"模型不可用"长得一样——本项目的解码层又有 ``except Exception``
    兜底，于是**一次文法写错 = 整条路静默退化成"模型没选"**（实测踩过：传 GBNF
    导致每次生成都 500，表现却是"未选定操作"）。在这里当场抛错，故障就**有声**。

    ``dict`` 一律放行（交给服务端判 JSON Schema 合法性）；``str`` 只放行 ``"json"``。
    """
    if format is None or isinstance(format, dict):
        return
    if isinstance(format, str) and format.strip() == "json":
        return
    kind = type(format).__name__
    raise InputValidationError(
        f"format={format!r}（{kind}）不是 Ollama 支持的形式：只接受 JSON Schema "
        '`dict` 或字符串 "json"。**Ollama 不支持 GBNF 文法**（实测 0.34.4 返回 '
        "HTTP 500）——要写 GBNF 请换 llama.cpp server 并改 url。本模块曾把 GBNF "
        "当 format 传，导致的 500 被调用方的 except 吞掉，表现成\"模型没选\"。",
        field="format",
    )


def _warn_if_no_headroom(caller: str) -> None:
    """Warn (do not block) when free memory is *sustainedly* below the reserve.

    项目策略（内部设计文档，第 III 篇《GPU 与内存使用策略》）：
    GPU 只服务 LLM 推理，任何触碰 GPU 的批量任务都必须留出 ``RESERVE_BYTES``
    （2 GB）空闲内存。

    "持续"很重要：单次读数偏低是正常的——内核会在毫秒级回收非活跃页——所以
    这个守卫会多次采样后才判定。瞬时下跌绝不能产生警告，否则每次级联调用都会
    刷一条。

    这里只警告：一个已经驻留内存（在 ``keep_alive`` 内）的模型是可以调用的，
    在此处拒绝会破坏级联调用方。
    """
    try:
        # 直连 resource 子包（memory_guard 只是对外兼容壳，内部不走它）。
        from ..resource import describe, is_sustained_low, reserve_bytes
    except ImportError:  # pragma: no cover - 历史布局
        return
    try:
        sustained, samples = is_sustained_low(reserve_bytes())
        if sustained:
            warnings.warn(
                f"{caller}: 空闲内存持续低于 2 GB 预留线"
                f"（连续 {samples} 次探测；{describe()}），"
                f"Ollama 加载模型可能触发 swap。",
                stacklevel=3,
            )
    except Exception:  # pragma: no cover - 绝不因为一次探测而弄断调用
        pass


def generate(
    model: str,
    prompt: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    usage: Optional[TokenUsage] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> str:
    """从本地 Ollama 模型生成响应。

    Parameters
    ----------
    model : str
        模型标签，例如 "qwen3.5:4b"。
    prompt : str
        Prompt 文本。
    url : str
        Ollama generate 端点。当设置了 ``OLLAMA_HOSTS`` 环境变量时，它会被其取代
        而忽略。
    timeout : float
        HTTP 超时（秒）。
    keep_alive : str
        Ollama 在本次调用后应把模型保留加载多久（例如 "5m"、"30m"、"1h"）。
        较长的值可避免级联调用中反复加载模型的延迟。
    options : dict, optional
        额外的 Ollama 生成选项，例如 {"num_predict": 512, "temperature": 0.6}。
        未显式给出 ``num_ctx`` 时自动补默认值 :data:`DEFAULT_NUM_CTX`
        （4096）——模型自带的默认上限（可达 262144）会让 KV cache 吃掉数 GB
        统一内存。
    usage : TokenUsage, optional
        若提供，本次调用的 token 数会累加到这个累加器。
    max_retries : int
        瞬时失败（URLError / 超时 / 连接重置 / broken pipe）后的重试次数。
        每次尝试之间的退避为 ``retry_backoff * 2**attempt`` 秒。重试是**按 host**
        的：某个 host 持续失败则在 ``max_retries`` 次后被放弃，转而尝试下一个
        ``OLLAMA_HOSTS`` 条目。
    retry_backoff : float
        首次重试的基础退避（秒）。
    think : bool, optional
        Ollama ``think`` 标志。``None``（默认）**省略该字段**，故行为不变；
        ``False`` 请求模型跳过推理（仅部分模型支持——例如 qwen3；``deepseek-r1``
        会忽略它）。
    format : dict or str, optional
        **约束解码**（constrained decoding）：``None``（默认）**省略字段**，行为
        不变。传 JSON Schema ``dict``（Ollama structured outputs）即让服务端
        **只在合法输出空间里采样**——这比"先生成、再严格解析、失败就退回兜底"更强：
        非法 token 从概率上就取不到。本项目的用途见
        :func:`kakeya_ai.nlu.intent.make_llm_semantic_classifier`（把分类输出限制成
        受控词表的 JSON）。

        **``str`` 只接受 ``"json"``**：Ollama **不支持** GBNF 文法（实测 0.34.4：
        传 GBNF 得到 ``HTTP 500 invalid format: ...; expected "json" or a valid JSON
        Schema object``）。要写 GBNF 需换 llama.cpp server 并改 ``url``。传别的字符串
        会**当场**抛 :class:`~kakeya_ai.errors.InputValidationError`（而不是让 500
        在调用方被吞掉——本项目实测踩过这个坑：HTTP 500 被 ``except Exception`` 吃掉，
        表现成"模型没选"，故障完全静默）。

    Returns
    -------
    response_text : str

    Raises
    ------
    RuntimeError
        若 API 不可达或模型不可用。
    """
    _validate_format(format)
    options = _with_default_num_ctx(options)
    # 模型行为策略统一拦截（think:false 兼容性、预算下限——2026-11-12 起）：
    # 某些模型（xian-reason）传 think:false 会被 renderer 剥掉训练预填，
    # 输出退化；详见 ai/model_policies.py。
    from .model_policies import apply_generate_policy, prepare_generate_prompt

    think, options = apply_generate_policy(model, think, options)
    # 微调模型的训练格式包装：/api/generate 不走 Modelfile TEMPLATE，
    # 裸 prompt 会让微调模型出训练分布（见 model_policies 的字段注释）。
    prompt = prepare_generate_prompt(model, prompt)
    result = _generate_response(
        model,
        prompt,
        url,
        timeout=timeout,
        keep_alive=keep_alive,
        options=options,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        think=think,
        format=format,
    )
    if "error" in result:
        raise RuntimeError(f"Ollama API error: {result['error']}")
    if usage is not None:
        usage.add_from_ollama_response(result)
    text = str(result.get("response", ""))
    if not text.strip():
        # 空回答与"模型真的没话可说"在 response 层无法区分；think 型模型的典型
        # 成因是思维链吃光 num_predict（done_reason='length'）。留 warning 供排查。
        _logger.warning(
            "Ollama 空回答: model=%s done_reason=%s eval_count=%s think=%s",
            model,
            result.get("done_reason"),
            result.get("eval_count"),
            think,
        )
    return text


@observe_llm_call(observability=_obs, extract_tokens=_extract_ollama_tokens)
def _generate_response(
    model: str,
    prompt: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """带按 host 重试与 host 故障转移地调用 generate API。

    故障转移语义：从轮询游标所在的 host 开始，在其上耗尽 ``max_retries`` 次
    瞬时失败重试，然后按 ``OLLAMA_HOSTS`` 顺序移到下一个 host；成功后游标前移，
    使下次调用从下一个 host 开始。当所有 host 都失败时，抛出最后一个异常
    （转换为历史上的 RuntimeError 消息）。
    """
    _warn_if_no_headroom("generate")
    hosts = _resolve_hosts(url)
    last_error: Optional[BaseException] = None
    for offset in range(len(hosts)):
        idx = (_host_cursor + offset) % len(hosts)
        host_url = hosts[idx]
        try:
            result = _generate_with_retries(
                model,
                prompt,
                host_url,
                timeout=timeout,
                keep_alive=keep_alive,
                options=options,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                think=think,
                format=format,
            )
        except Exception as e:
            last_error = e
            if _should_retry(e) and offset < len(hosts) - 1:
                _logger.warning(
                    "Ollama host %s unreachable (%s); trying next host (%d/%d)",
                    host_url, e, offset + 1, len(hosts) - 1,
                )
                continue
            raise _connection_error(e, host_url, max_retries) from e
        _advance_host_cursor(idx, len(hosts))
        return result
    # 实践中不可达：循环要么返回要么抛错。保留它是为了类型检查器与防御式编程。
    raise RuntimeError(f"Could not reach any Ollama host ({last_error})")


def _generate_with_retries(
    model: str,
    prompt: str,
    url: str,
    timeout: float,
    keep_alive: str,
    options: Optional[Dict[str, Any]],
    max_retries: int,
    retry_backoff: float,
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """在单个 host 上重试请求；最后一个异常会被重新抛出。"""
    for attempt in range(max_retries + 1):
        try:
            return _ollama_api_request(
                model,
                prompt,
                url,
                timeout=timeout,
                keep_alive=keep_alive,
                options=options,
                max_retries=0,
                retry_backoff=retry_backoff,
                think=think,
                format=format,
            )
        except Exception as e:
            if attempt < max_retries and _should_retry(e):
                delay = retry_backoff * (2 ** attempt)
                _logger.warning(
                    "Ollama request failed (%s); retrying in %.1fs (attempt %d/%d)",
                    e, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            raise
    # 循环体必定 return 或 raise，故此处理论上不可达。但显式写出而不是
    # 依赖它：``max_retries < 0`` 时 ``range()`` 为空，循环一次都不执行，
    # 函数会返回 None——调用方拿着 None 当 dict 用，报错位置离根因很远。
    # 宁可在此处给出明确的错误，也不要留下一条返回 None 的隐式路径。
    raise RuntimeError(
        f"重试循环未能执行：max_retries={max_retries} 为负值，无可用的尝试次数"
    )


def generate_stream(
    model: str,
    prompt: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
):
    """从本地 Ollama 模型流式输出响应块。

    产出每个流式块的 `response` 文本。拼接各块成最终答案由调用方负责。

    初始连接会在瞬时失败时按 host 重试最多 ``max_retries`` 次，然后尝试下一个
    宿主条目（语义同 :func:`generate`）。流中途的失败不会重连（生成器直接抛错）。

    Raises
    ------
    RuntimeError
        若 API 不可达或模型不可用。
    """
    _validate_format(format)
    _warn_if_no_headroom("generate_stream")
    options = _with_default_num_ctx(options)
    hosts = _resolve_hosts(url)
    resp: Optional[Any] = None
    host_url = hosts[0]
    for offset in range(len(hosts)):
        idx = (_host_cursor + offset) % len(hosts)
        host_url = hosts[idx]
        req = _build_stream_request(host_url, model, prompt, keep_alive, options, think, format)
        try:
            resp = _open_stream_with_retries(
                req, timeout, max_retries, retry_backoff
            )
        except Exception as e:
            if _should_retry(e) and offset < len(hosts) - 1:
                _logger.warning(
                    "Ollama host %s unreachable (%s); trying next host (%d/%d)",
                    host_url, e, offset + 1, len(hosts) - 1,
                )
                continue
            raise _connection_error(e, host_url, max_retries) from e
        _advance_host_cursor(idx, len(hosts))
        break
    if resp is None:  # 防御式；循环要么设置 resp 要么抛错
        raise RuntimeError(f"Could not reach any Ollama host in {hosts}")
    try:
        with resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if "error" in chunk:
                    raise RuntimeError(f"Ollama API error: {chunk['error']}")
                yield str(chunk.get("response", ""))
                if chunk.get("done", False):
                    break
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama API at {host_url}. Is the server running? ({e})"
        ) from e


def _build_stream_request(
    url: str,
    model: str,
    prompt: str,
    keep_alive: str,
    options: Optional[Dict[str, Any]],
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> urllib.request.Request:
    """为单个 host 构建流式 generate 请求。"""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
    }
    if options is not None:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    if format is not None:
        payload["format"] = format
    data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _open_stream_with_retries(
    req: urllib.request.Request,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> Any:
    """打开一个流式响应，对瞬时连接失败进行重试。"""
    for attempt in range(max_retries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            if attempt < max_retries and _should_retry(e):
                delay = retry_backoff * (2 ** attempt)
                _logger.warning(
                    "Ollama stream connect failed (%s); retrying in %.1fs (attempt %d/%d)",
                    e, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            raise

    # **循环能走到这里只有一种情况**：``max_retries < 0`` ⇒ ``range(0)`` 为空
    # ⇒ 一次都没尝试。此前会**隐式返回 None**，调用方把它当响应对象用，
    # 在后面某处抛 AttributeError——**报错点离根因很远**（实测确认）。
    # 显式抛错，让根因在**这里**暴露（``max_retries`` 的调用方一眼能看出传错了）。
    #
    # 用 ``InputValidationError``（语义类型）而不是裸 ``ValueError``：项目的
    # 纪律是"核心模块不抛裸 ValueError"（见 `tests/test_error_taxonomy.py`），
    # 因为下游要按**类别**处置（输入错 vs 环境错 vs 数值极限），裸类型丢掉了这层
    # 信息。它本身是 ``ValueError`` 的子类，故既有 ``except ValueError`` 仍生效。
    raise InputValidationError(
        f"max_retries 不能为负（实际 {max_retries}）："
        "range(max_retries + 1) 为空会导致一次都不尝试。",
        field="max_retries", value=max_retries,
    )


def _ollama_api_request(
    model: str,
    prompt: str,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    keep_alive: str = "5m",
    options: Optional[Dict[str, Any]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    think: Optional[bool] = None,
    format: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """向 Ollama API 发送一个 generate 请求并返回 JSON 响应。

    瞬时失败（URLError / 超时 / 连接重置 / broken pipe）会以指数退避重试最多
    ``max_retries`` 次；重试耗尽时重新抛出最后一个异常。
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
    }
    if options is not None:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    if format is not None:
        payload["format"] = format
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < max_retries and _should_retry(e):
                delay = retry_backoff * (2 ** attempt)
                _logger.warning(
                    "Ollama request failed (%s); retrying in %.1fs (attempt %d/%d)",
                    e, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            raise
    # 同 ``_generate_with_retries``：显式兜住 ``max_retries < 0`` 这条
    # “循环一次都不跑”的路径，不让函数静默返回 None。
    raise RuntimeError(
        f"重试循环未能执行：max_retries={max_retries} 为负值，无可用的尝试次数"
    )


def is_server_ready(url: str = DEFAULT_OLLAMA_URL) -> bool:
    """若任一已配置的 Ollama host 有响应，则返回 True。"""
    for host in _resolve_hosts(url):
        try:
            req = urllib.request.Request(
                host.replace("/api/generate", "/api/tags"), method="GET"
            )
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT):
                return True
        except Exception as exc:
            _logger.warning("Ollama server not ready at %s: %s", host, exc)
    return False
