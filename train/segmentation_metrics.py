"""Research-oriented metrics for music structure segmentation."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

Segment = Tuple[float, float, str]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _internal_boundaries(segments: Sequence[Segment]) -> List[float]:
    """Return sorted boundaries excluding track start and end markers."""
    if not segments:
        return []
    points = sorted({float(start) for start, _, _ in segments}
                    | {float(end) for _, end, _ in segments})
    return points[1:-1]


def boundary_prf(
    reference: Sequence[float],
    estimated: Sequence[float],
    window: float,
) -> Tuple[float, float, float]:
    """One-to-one boundary matching equivalent to MIR hit-rate semantics."""
    ref = sorted(float(x) for x in reference)
    est = sorted(float(x) for x in estimated)
    i = j = matches = 0
    while i < len(ref) and j < len(est):
        if abs(ref[i] - est[j]) <= window:
            matches += 1
            i += 1
            j += 1
        elif est[j] < ref[i] - window:
            j += 1
        else:
            i += 1

    precision = matches / len(est) if est else 0.0
    recall = matches / len(ref) if ref else 0.0
    return precision, recall, _f1(precision, recall)


def _iou(a: Segment, b: Segment) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0.0


def _one_to_one_segment_matches(
    reference: Sequence[Segment],
    estimated: Sequence[Segment],
    label: str,
    iou_threshold: float,
) -> int:
    ref_ids = [i for i, seg in enumerate(reference) if seg[2] == label]
    est_ids = [i for i, seg in enumerate(estimated) if seg[2] == label]
    if not ref_ids or not est_ids:
        return 0
    adjacency = np.asarray([
        [
            _iou(reference[ri], estimated[ei]) >= iou_threshold
            for ei in est_ids
        ]
        for ri in ref_ids
    ], dtype=np.int8)
    matching = maximum_bipartite_matching(
        csr_matrix(adjacency), perm_type="column"
    )
    return int(np.sum(matching >= 0))


def segment_level_metrics(
    pred_segments: Sequence[Segment],
    gt_segments: Sequence[Segment],
    labels: Iterable[str],
    segment_iou_threshold: float = 0.5,
) -> Dict:
    """Compute MIR-style boundary scores and semantic segment F1.

    Boundary scores exclude the trivial track start/end markers and use
    one-to-one matching. Semantic segments are matched one-to-one when their
    labels agree and IoU is at least ``segment_iou_threshold``.
    """
    label_list = list(labels)
    if not gt_segments or not pred_segments:
        zeros = {label: 0.0 for label in label_list}
        return {
            "boundary_precision_0_5s": 0.0,
            "boundary_recall_0_5s": 0.0,
            "boundary_f1": 0.0,
            "boundary_f1_exact": 0.0,
            "boundary_f1_0_5s": 0.0,
            "boundary_f1_3s": 0.0,
            "segment_iou": 0.0,
            "macro_seg_f1": 0.0,
            "per_class_seg_f1": zeros,
        }

    gt_bounds = _internal_boundaries(gt_segments)
    pred_bounds = _internal_boundaries(pred_segments)
    _, _, exact_f1 = boundary_prf(gt_bounds, pred_bounds, window=1e-6)
    precision_05, recall_05, f1_05 = boundary_prf(
        gt_bounds, pred_bounds, window=0.5
    )
    _, _, f1_3 = boundary_prf(gt_bounds, pred_bounds, window=3.0)

    best_ious = [
        max((_iou(pred, gt) for gt in gt_segments), default=0.0)
        for pred in pred_segments
    ]

    per_class_f1 = {}
    for label in label_list:
        n_ref = sum(seg[2] == label for seg in gt_segments)
        n_est = sum(seg[2] == label for seg in pred_segments)
        tp = _one_to_one_segment_matches(
            gt_segments, pred_segments, label, segment_iou_threshold
        )
        precision = tp / n_est if n_est else 0.0
        recall = tp / n_ref if n_ref else 0.0
        per_class_f1[label] = round(_f1(precision, recall), 4)

    return {
        "boundary_precision_0_5s": round(precision_05, 4),
        "boundary_recall_0_5s": round(recall_05, 4),
        "boundary_f1": round(f1_05, 4),
        "boundary_f1_exact": round(exact_f1, 4),
        "boundary_f1_0_5s": round(f1_05, 4),
        "boundary_f1_3s": round(f1_3, 4),
        "segment_iou": round(float(np.mean(best_ious)), 4),
        "macro_seg_f1": round(float(np.mean(list(per_class_f1.values()))), 4),
        "per_class_seg_f1": per_class_f1,
    }
