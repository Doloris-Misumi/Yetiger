"""Audit exported YesTiger prediction JSON files.

The script is intentionally lightweight: it reads the prediction exports from
``test_bar.py`` and produces a Markdown report with segment tables, boundary
misses/extra boundaries, and dominant label confusions.

Usage:
  python audit_predictions.py \
    --pred-dir predictions_bar_coarse_paper_test \
    --metrics test_results_bar_coarse_paper_test.json \
    --songs brushupbrassup itsuaietara athiscode futarikoto \
    --out prediction_audit_paper_test.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Segment = Dict[str, object]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _duration(segment: Segment) -> float:
    return float(segment["end"]) - float(segment["start"])


def _overlap(a: Segment, b: Segment) -> float:
    return max(
        0.0,
        min(float(a["end"]), float(b["end"]))
        - max(float(a["start"]), float(b["start"])),
    )


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _segment_table(segments: Sequence[Segment]) -> List[str]:
    lines = [
        "| # | start | end | dur | label |",
        "|---:|---:|---:|---:|---|",
    ]
    for index, seg in enumerate(segments, start=1):
        lines.append(
            "| "
            f"{index} | {_fmt_seconds(float(seg['start']))} "
            f"| {_fmt_seconds(float(seg['end']))} "
            f"| {_fmt_seconds(_duration(seg))} "
            f"| {seg['label']} |"
        )
    return lines


def _boundaries(segments: Sequence[Segment]) -> List[Tuple[float, str, str]]:
    result = []
    for prev_seg, next_seg in zip(segments, segments[1:]):
        result.append((
            float(prev_seg["end"]),
            str(prev_seg["label"]),
            str(next_seg["label"]),
        ))
    return result


def _boundary_match_report(
    gt_segments: Sequence[Segment],
    pred_segments: Sequence[Segment],
    tolerance_s: float,
) -> Tuple[List[str], List[str]]:
    gt_boundaries = _boundaries(gt_segments)
    pred_boundaries = _boundaries(pred_segments)
    used_pred = set()
    missed = []
    for gt_index, (gt_time, gt_left, gt_right) in enumerate(gt_boundaries):
        best_index = None
        best_delta = None
        for pred_index, (pred_time, _, _) in enumerate(pred_boundaries):
            if pred_index in used_pred:
                continue
            delta = abs(pred_time - gt_time)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = pred_index
        if best_index is not None and best_delta is not None and best_delta <= tolerance_s:
            used_pred.add(best_index)
            continue
        nearest = _nearest_boundary(gt_time, pred_boundaries)
        missed.append(_boundary_line(gt_time, gt_left, gt_right, nearest))

    extra = []
    for pred_index, (pred_time, pred_left, pred_right) in enumerate(pred_boundaries):
        if pred_index in used_pred:
            continue
        nearest = _nearest_boundary(pred_time, gt_boundaries)
        extra.append(_boundary_line(pred_time, pred_left, pred_right, nearest))
    return missed, extra


def _nearest_boundary(
    target_time: float,
    boundaries: Sequence[Tuple[float, str, str]],
) -> Optional[Tuple[float, str, str, float]]:
    if not boundaries:
        return None
    time, left, right = min(
        boundaries,
        key=lambda item: abs(item[0] - target_time),
    )
    return time, left, right, time - target_time


def _boundary_line(
    time: float,
    left: str,
    right: str,
    nearest: Optional[Tuple[float, str, str, float]],
) -> str:
    if nearest is None:
        return f"- {_fmt_seconds(time)} `{left} -> {right}`; no opposite boundary"
    near_time, near_left, near_right, signed_delta = nearest
    return (
        f"- {_fmt_seconds(time)} `{left} -> {right}`; nearest "
        f"{_fmt_seconds(near_time)} `{near_left} -> {near_right}` "
        f"({signed_delta:+.2f}s)"
    )


def _label_confusions(
    gt_segments: Sequence[Segment],
    pred_segments: Sequence[Segment],
) -> List[Tuple[str, str, float]]:
    overlaps: Dict[Tuple[str, str], float] = defaultdict(float)
    for gt in gt_segments:
        for pred in pred_segments:
            seconds = _overlap(gt, pred)
            if seconds <= 0:
                continue
            gt_label = str(gt["label"])
            pred_label = str(pred["label"])
            if gt_label != pred_label:
                overlaps[(gt_label, pred_label)] += seconds
    return sorted(
        ((gt, pred, seconds) for (gt, pred), seconds in overlaps.items()),
        key=lambda item: item[2],
        reverse=True,
    )


def _problem_gt_segments(
    gt_segments: Sequence[Segment],
    pred_segments: Sequence[Segment],
    min_mismatch_ratio: float = 0.25,
) -> List[str]:
    lines = []
    for index, gt in enumerate(gt_segments, start=1):
        gt_label = str(gt["label"])
        dur = max(_duration(gt), 1e-8)
        by_label: Dict[str, float] = defaultdict(float)
        for pred in pred_segments:
            seconds = _overlap(gt, pred)
            if seconds > 0:
                by_label[str(pred["label"])] += seconds
        correct = by_label.get(gt_label, 0.0)
        mismatch = max(0.0, dur - correct)
        if mismatch / dur < min_mismatch_ratio:
            continue
        top_pred, top_seconds = max(
            by_label.items(),
            key=lambda item: item[1],
            default=("none", 0.0),
        )
        lines.append(
            "- GT "
            f"#{index} `{gt_label}` "
            f"{_fmt_seconds(float(gt['start']))}-{_fmt_seconds(float(gt['end']))} "
            f"({ _fmt_seconds(dur) }): "
            f"correct={correct / dur:.0%}, mostly_pred=`{top_pred}` "
            f"({top_seconds / dur:.0%})"
        )
    return lines


def _coverage_warnings(
    gt_segments: Sequence[Segment],
    pred_segments: Sequence[Segment],
    tolerance_s: float = 0.5,
) -> List[str]:
    if not gt_segments or not pred_segments:
        return ["- missing GT or predicted segments"]
    gt_start = float(gt_segments[0]["start"])
    gt_end = float(gt_segments[-1]["end"])
    pred_start = float(pred_segments[0]["start"])
    pred_end = float(pred_segments[-1]["end"])
    warnings = []
    if pred_start - gt_start > tolerance_s:
        warnings.append(
            f"- prediction starts {_fmt_seconds(pred_start - gt_start)} "
            "after GT start"
        )
    if gt_start - pred_start > tolerance_s:
        warnings.append(
            f"- prediction starts {_fmt_seconds(gt_start - pred_start)} "
            "before GT start"
        )
    if gt_end - pred_end > tolerance_s:
        warnings.append(
            f"- prediction ends {_fmt_seconds(gt_end - pred_end)} "
            "before GT end; tail GT cannot be matched by predicted segments"
        )
    if pred_end - gt_end > tolerance_s:
        warnings.append(
            f"- prediction ends {_fmt_seconds(pred_end - gt_end)} "
            "after GT end"
        )
    return warnings if warnings else ["- none"]


def _metrics_by_song(metrics_path: Optional[Path]) -> Dict[str, Dict]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    payload = _read_json(metrics_path)
    return {
        item.get("test_song", ""): item
        for item in payload.get("per_song", [])
        if item.get("test_song")
    }


def _metric(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def _song_sort_key(song_id: str, metrics: Dict[str, Dict]) -> Tuple[float, str]:
    return (metrics.get(song_id, {}).get("macro_seg_f1", 999.0), song_id)


def _prediction_files(pred_dir: Path, songs: Sequence[str]) -> List[Path]:
    if songs:
        return [pred_dir / f"{song}.prediction.json" for song in songs]
    return sorted(pred_dir.glob("*.prediction.json"))


def build_report(
    pred_dir: Path,
    metrics_path: Optional[Path],
    songs: Sequence[str],
    boundary_tolerance_s: float,
) -> str:
    metrics = _metrics_by_song(metrics_path)
    files = _prediction_files(pred_dir, songs)
    missing = [path for path in files if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prediction files: {missing_text}")

    payloads = [_read_json(path) for path in files]
    payloads.sort(
        key=lambda payload: _song_sort_key(str(payload["song_id"]), metrics)
    )

    lines = [
        "# Prediction Audit",
        "",
        f"- Prediction directory: `{pred_dir}`",
        f"- Metrics file: `{metrics_path}`",
        f"- Boundary tolerance: `{boundary_tolerance_s:.1f}s`",
        "",
        "## Overview",
        "",
        "| song | acc | bar_f1 | seg_f1 | b@0.5 | b@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for payload in payloads:
        song_id = str(payload["song_id"])
        song_metrics = metrics.get(song_id, {})
        lines.append(
            f"| {song_id} "
            f"| {_metric(song_metrics.get('accuracy'))} "
            f"| {_metric(song_metrics.get('macro_f1'))} "
            f"| {_metric(song_metrics.get('macro_seg_f1'))} "
            f"| {_metric(song_metrics.get('boundary_f1_0_5s'))} "
            f"| {_metric(song_metrics.get('boundary_f1_3s'))} |"
        )

    for payload in payloads:
        song_id = str(payload["song_id"])
        gt_segments = payload["ground_truth"]
        pred_segments = payload["predicted"]
        song_metrics = metrics.get(song_id, {})
        missed, extra = _boundary_match_report(
            gt_segments,
            pred_segments,
            tolerance_s=boundary_tolerance_s,
        )
        confusions = _label_confusions(gt_segments, pred_segments)
        problems = _problem_gt_segments(gt_segments, pred_segments)
        coverage_warnings = _coverage_warnings(gt_segments, pred_segments)

        lines.extend([
            "",
            f"## {song_id}",
            "",
            (
                f"- Metrics: acc={_metric(song_metrics.get('accuracy'))}, "
                f"bar_f1={_metric(song_metrics.get('macro_f1'))}, "
                f"seg_f1={_metric(song_metrics.get('macro_seg_f1'))}, "
                f"b@0.5={_metric(song_metrics.get('boundary_f1_0_5s'))}, "
                f"b@3={_metric(song_metrics.get('boundary_f1_3s'))}"
            ),
            f"- GT segments: {len(gt_segments)}; predicted segments: {len(pred_segments)}",
            "",
            "### Timeline coverage",
            "",
        ])
        lines.extend(coverage_warnings)
        lines.extend([
            "",
            "### GT segments",
            "",
            *_segment_table(gt_segments),
            "",
            "### Predicted segments",
            "",
            *_segment_table(pred_segments),
            "",
            f"### Missed GT boundaries @ {boundary_tolerance_s:.1f}s",
            "",
        ])
        lines.extend(missed if missed else ["- none"])
        lines.extend([
            "",
            f"### Extra predicted boundaries @ {boundary_tolerance_s:.1f}s",
            "",
        ])
        lines.extend(extra if extra else ["- none"])
        lines.extend([
            "",
            "### Main label confusions by overlap",
            "",
        ])
        if confusions:
            for gt_label, pred_label, seconds in confusions[:10]:
                lines.append(
                    f"- `{gt_label}` predicted as `{pred_label}`: "
                    f"{_fmt_seconds(seconds)}"
                )
        else:
            lines.append("- none")
        lines.extend([
            "",
            "### GT segments with substantial mismatch",
            "",
        ])
        lines.extend(problems if problems else ["- none"])

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exported YesTiger prediction JSON files."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("predictions_bar_coarse_paper_test"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("test_results_bar_coarse_paper_test.json"),
    )
    parser.add_argument("--songs", nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("prediction_audit.md"))
    parser.add_argument("--boundary-tolerance", type=float, default=3.0)
    args = parser.parse_args()

    report = build_report(
        pred_dir=args.pred_dir,
        metrics_path=args.metrics,
        songs=args.songs,
        boundary_tolerance_s=args.boundary_tolerance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Audit saved to {args.out}")


if __name__ == "__main__":
    main()
