"""Curses-based TUI for live QM training metrics.

This TUI follows the JSONL logs produced by `src/04_train.py` for the
QM transfer-learning run (configured in `configs/train_qm.yaml`).

Typical usage:

    # Terminal 1: run training
    bash train_qm.sh

    # Terminal 2: launch live viewer
    python scripts/qm_train_tui.py
"""
from __future__ import annotations

import argparse
import curses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricsBuffers:
    steps: List[int]
    values: List[float]
    last_record: Optional[Dict]


def _read_new_jsonl(path: Path, offset: int) -> Tuple[List[Dict], int]:
    """Read any new JSONL records from `path` starting at `offset`."""
    if not path.exists():
        return [], offset

    try:
        size = path.stat().st_size
    except OSError:
        return [], offset

    # File was truncated or replaced
    if size < offset:
        offset = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            lines = handle.readlines()
            offset = handle.tell()
    except OSError:
        return [], offset

    records: List[Dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip partial or malformed lines (e.g., mid-write)
            continue
    return records, offset


def _update_buffers(
    buffers: MetricsBuffers,
    records: List[Dict],
    metric_key: str,
    max_points: int,
) -> None:
    for rec in records:
        if metric_key not in rec:
            continue
        value = rec[metric_key]
        step = rec.get("step")
        if step is None:
            continue
        try:
            step_int = int(step)
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value_float):
            continue
        buffers.steps.append(step_int)
        buffers.values.append(value_float)
        buffers.last_record = rec

    # Keep the full history so the plot always reflects all
    # available data, scaled to the current window. The
    # max_points argument is retained for compatibility but
    # no longer used to enforce a sliding window.


def _draw_centered(stdscr: "curses._CursesWindow", row: int, text: str) -> None:
    height, width = stdscr.getmaxyx()
    if row < 0 or row >= height:
        return
    x = max((width - len(text)) // 2, 0)
    try:
        stdscr.addstr(row, x, text[: max(width - x, 0)])
    except curses.error:
        pass


def _draw_plot(
    stdscr: "curses._CursesWindow",
    y: int,
    x: int,
    height: int,
    width: int,
    train_buf: MetricsBuffers,
    val_buf: Optional[MetricsBuffers],
    metric_key: str,
) -> None:
    if height <= 2 or width <= 10:
        return

    steps = train_buf.steps
    values = train_buf.values
    if not steps:
        _draw_centered(stdscr, y + height // 2, "Waiting for training metrics...")
        return

    all_values = list(values)
    if val_buf is not None and val_buf.values:
        all_values.extend(val_buf.values)

    v_min = min(all_values)
    v_max = max(all_values)
    if not math.isfinite(v_min) or not math.isfinite(v_max):
        return
    if v_max == v_min:
        delta = max(abs(v_max), 1.0) * 0.1
        v_min -= delta
        v_max += delta

    value_range = v_max - v_min

    step_min = min(steps)
    step_max = max(steps)
    if step_max == step_min:
        step_max = step_min + 1
    step_range = float(step_max - step_min)

    x_margin = 10
    plot_width = max(width - x_margin - 1, 1)
    plot_height = height - 2

    # Axis labels
    top_label = f"{metric_key} max {v_max:.4g}"
    bottom_label = f"{metric_key} min {v_min:.4g} | steps {step_min}-{step_max}"
    try:
        stdscr.addstr(y, x, top_label[: width])
        stdscr.addstr(y + height - 1, x, bottom_label[: width])
    except curses.error:
        pass

    def step_to_col(step: int) -> int:
        rel = (step - step_min) / step_range
        rel = min(max(rel, 0.0), 1.0)
        return x + x_margin + int(rel * (plot_width - 1))

    def value_to_row(val: float) -> int:
        rel = (val - v_min) / value_range
        rel = min(max(rel, 0.0), 1.0)
        return y + 1 + (plot_height - 1 - int(rel * (plot_height - 1)))

    # Draw training curve
    for step, val in zip(steps, values):
        col = step_to_col(step)
        row = value_to_row(val)
        if row < y + 1 or row >= y + 1 + plot_height:
            continue
        if col < x + x_margin or col >= x + x_margin + plot_width:
            continue
        try:
            stdscr.addch(row, col, "*")
        except curses.error:
            continue

    # Draw validation markers (if available)
    if val_buf is not None and val_buf.steps and val_buf.values:
        for step, val in zip(val_buf.steps, val_buf.values):
            col = step_to_col(step)
            row = value_to_row(val)
            if row < y + 1 or row >= y + 1 + plot_height:
                continue
            if col < x + x_margin or col >= x + x_margin + plot_width:
                continue
            try:
                stdscr.addch(row, col, "+")
            except curses.error:
                continue


def _format_record_summary(prefix: str, rec: Optional[Dict], metric_key: str) -> str:
    if rec is None:
        return f"{prefix}: (no data)"
    step = rec.get("step", "?")
    epoch = rec.get("epoch", "?")
    metric_val = rec.get(metric_key)
    try:
        metric_str = f"{float(metric_val):.6g}" if metric_val is not None else "n/a"
    except (TypeError, ValueError):
        metric_str = "n/a"
    return f"{prefix}: step={step} epoch={epoch} {metric_key}={metric_str}"


def run_tui(stdscr: "curses._CursesWindow", args: argparse.Namespace) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)

    train_path = args.log_dir / "train_metrics.jsonl"
    val_path = args.log_dir / "val_metrics.jsonl"

    train_buf = MetricsBuffers(steps=[], values=[], last_record=None)
    val_buf = MetricsBuffers(steps=[], values=[], last_record=None)

    train_offset = 0
    val_offset = 0

    metric_key = args.metric
    refresh_interval = max(args.refresh, 0.1)
    max_points = max(args.max_points, 10)

    last_update = 0.0

    while True:
        now = time.time()
        if now - last_update >= refresh_interval:
            last_update = now

            train_records, train_offset = _read_new_jsonl(train_path, train_offset)
            val_records, val_offset = _read_new_jsonl(val_path, val_offset)

            _update_buffers(train_buf, train_records, metric_key, max_points)
            _update_buffers(val_buf, val_records, f"val_{metric_key}", max_points)

        stdscr.erase()
        height, width = stdscr.getmaxyx()

        plot_height = max(height - 6, 5)
        plot_width = width

        _draw_plot(
            stdscr,
            y=0,
            x=0,
            height=plot_height,
            width=plot_width,
            train_buf=train_buf,
            val_buf=val_buf,
            metric_key=metric_key,
        )

        info_row = plot_height + 1
        help_row = height - 2

        train_summary = _format_record_summary("Train", train_buf.last_record, metric_key)
        val_summary = _format_record_summary("Val", val_buf.last_record, f"val_{metric_key}")
        status = f"Log dir: {str(args.log_dir)}"

        try:
            stdscr.addstr(info_row, 0, train_summary[: width])
            stdscr.addstr(info_row + 1, 0, val_summary[: width])
            stdscr.addstr(info_row + 2, 0, status[: width])
            stdscr.addstr(help_row, 0, "Controls: q to quit")
        except curses.error:
            pass

        stdscr.refresh()

        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1
        if ch in (ord("q"), ord("Q")):
            break

        time.sleep(0.05)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live TUI for QM training metrics (loss vs step).")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/logs_qm"),
        help="Directory containing train_metrics.jsonl and val_metrics.jsonl.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        help="Training metric key to plot (e.g., loss, loss_pos, loss_vel).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="Refresh interval in seconds for reading logs.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Maximum number of points to keep in memory for plotting.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    curses.wrapper(run_tui, args)


if __name__ == "__main__":
    main()
