#!/usr/bin/env python
r""".kpt 格式解释器 —— 查看、导出、复核证明轨迹文件。

**为什么需要它**：``.kpt`` 是自定格式，没有现成工具能读。这个 CLI 是格式的
"解释器"：把文件翻译成人能读的形式（``show``）、给机器消费的形式（``json``），
以及复核结论（``verify``）。三个速度档位对应三种需求：

    kpt head    run.kpt          # 快速打开：只读头部，长 trace 也是微秒级
    kpt steps   run.kpt -n 5     # 流式导入：逐步读，不载入整个文件
    kpt show    run.kpt          # 完整渲染成可读文本
    kpt verify  run.kpt          # 独立复核（重算体积下界 + 验哈希链）
    kpt json    run.kpt          # 结构化输出（管道给别的工具）

两种跑法（同一个实现）：

    .venv/bin/python scripts/orchestration/kpt.py head run.kpt      # 仓库入口（薄壳）
    .venv/bin/python -m kakeya_ai.kpt_cli head run.kpt  # 模块入口
"""

from __future__ import annotations

import argparse
import json
import sys

from .trace_format import (
    iter_steps,
    peek_trace,
    read_trace,
    verify_trace,
)


def cmd_head(args: argparse.Namespace) -> int:
    """快速打开：只读头部。"""
    header = peek_trace(args.path)
    if args.json:
        print(json.dumps({
            "path": header.path,
            "protocol": header.protocol,
            "created": header.created,
            "n_steps": header.n_steps,
            "n_tubes": header.n_tubes,
            "success": header.success,
            "exhausted": header.exhausted,
            "message": header.message,
            "geometry_hash": header.geometry_hash,
            "params": header.params,
            "final": header.final,
            "header_hash": header.header_hash,
        }, ensure_ascii=False, indent=2))
    else:
        print(header.summary())
    return 0


def cmd_steps(args: argparse.Namespace) -> int:
    """流式导入：逐行读数据体，只渲染被请求的那几步。"""
    count = 0
    for i, step in enumerate(iter_steps(args.path, limit=args.limit,
                                        offset=args.offset)):
        count += 1
        idx = step.get("index", i + args.offset)
        action = step.get("action", "?")
        if args.json:
            print(json.dumps(step, ensure_ascii=False))
            continue
        after = step.get("after_human", {})
        notes = step.get("notes", [])
        vr = after.get("volume_ratio")
        vr_s = f"{vr:.6g}" if isinstance(vr, (int, float)) else "—"
        print(f"[{idx:>3}] {action:<26} vol_ratio={vr_s}")
        for n in notes:
            print(f"      · {n}")
    if not args.json:
        print(f"\n共读出 {count} 步"
              + (f"（offset={args.offset}）" if args.offset else ""))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """完整渲染：头部 + 几何概要 + 每步。"""
    tf = read_trace(args.path)
    print(peek_trace(args.path).summary())
    print()
    print(f"几何：{len(tf.tubes)} 根管")
    if tf.tubes:
        t0 = tf.tubes[0]
        print(f"  首管：center={t0.center.tolist()} dir={t0.direction.v.tolist()} "
              f"δ={t0.delta} len={t0.length}")
    if tf.shading is not None:
        print(f"  染色：{len(tf.shading.densities)} 条，"
              f"λ_min={float(tf.shading.densities.min()):.6g}")
    print()
    print("步骤：")
    for step in tf.steps:
        idx = step.get("index", "?")
        action = step.get("action", "?")
        after = step.get("after_human", {})
        vr = after.get("volume_ratio")
        vr_s = f"{vr:.6g}" if isinstance(vr, (int, float)) else "—"
        print(f"  [{idx:>3}] {action:<26} vol_ratio={vr_s}  "
              f"hash={str(step.get('hash', ''))[:12]}…")
        for n in step.get("notes", []):
            print(f"        · {n}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """独立复核。退出码 0 = 通过，1 = 未通过（便于脚本用）。"""
    report = verify_trace(args.path, recompute=not args.no_recompute)
    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "n_steps": report.n_steps,
            "file_hash_ok": report.file_hash_ok,
            "geometry_hash_ok": report.geometry_hash_ok,
            "decimal_hex_ok": report.decimal_hex_ok,
            "chain_ok": report.chain_ok,
            "notes": report.notes,
            "verdicts": [
                {"index": v.index, "action": v.action, "status": v.status,
                 "detail": v.detail}
                for v in report.verdicts
            ],
        }, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
    return 0 if report.ok else 1


def cmd_json(args: argparse.Namespace) -> int:
    """结构化输出整个文件（供其他工具消费）。"""
    tf = read_trace(args.path)
    payload = {
        "header": tf.header,
        "geometry": {
            "n_tubes": len(tf.tubes),
            "hash": tf.geometry_hash,
        },
        "steps": tf.steps,
        "footer": tf.footer,
    }
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=".kpt 证明轨迹格式解释器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_head = sub.add_parser("head", help="快速打开：只读头部元数据")
    p_head.add_argument("path")
    p_head.add_argument("--json", action="store_true", help="输出 JSON")
    p_head.set_defaults(func=cmd_head)

    p_steps = sub.add_parser("steps", help="流式导入：逐步读出")
    p_steps.add_argument("path")
    p_steps.add_argument("-n", "--limit", type=int, default=None,
                         help="最多读多少步")
    p_steps.add_argument("--offset", type=int, default=0, help="跳过前 N 步")
    p_steps.add_argument("--json", action="store_true", help="每步一行 JSON")
    p_steps.set_defaults(func=cmd_steps)

    p_show = sub.add_parser("show", help="完整渲染成可读文本")
    p_show.add_argument("path")
    p_show.set_defaults(func=cmd_show)

    p_verify = sub.add_parser("verify", help="独立复核（哈希链 + 重算）")
    p_verify.add_argument("path")
    p_verify.add_argument("--no-recompute", action="store_true",
                          help="只验哈希链，不重算体积下界")
    p_verify.add_argument("--json", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_json = sub.add_parser("json", help="结构化输出整个文件")
    p_json.add_argument("path")
    p_json.add_argument("--pretty", action="store_true", help="缩进输出")
    p_json.set_defaults(func=cmd_json)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError:
        print(f"文件不存在：{args.path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"格式错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
