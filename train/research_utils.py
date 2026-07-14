"""Shared utilities for reproducible YesTiger training experiments."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .label_schema import (
        ANNOTATION_ONLY_LABELS,
        COARSE_LABELS,
        FINE_LABELS,
        FINE_TO_COARSE,
    )
    from .postprocess import PostprocessConfig, postprocess_coarse_logits
except ImportError:
    from label_schema import (
        ANNOTATION_ONLY_LABELS,
        COARSE_LABELS,
        FINE_LABELS,
        FINE_TO_COARSE,
    )
    from postprocess import PostprocessConfig, postprocess_coarse_logits

# Backwards-compatible name used by the training scripts.
MODEL_LABELS = FINE_LABELS

FINE_LABEL2ID = {label: index for index, label in enumerate(FINE_LABELS)}
COARSE_LABEL2ID = {label: index for index, label in enumerate(COARSE_LABELS)}
FINE_ID_TO_COARSE_ID = tuple(
    COARSE_LABEL2ID[FINE_TO_COARSE[label]] for label in FINE_LABELS
)


TRAIN_IDS = [
    "ameagarikimito", "angles", "animarupartykaisai-chuu", "bravejewel",
    "breakthrough", "chiisanakoinouta", "choirschoir", "chouchoumusubi",
    "circiling", "crucifix_x", "daten", "doki_doki_scary", "dokidokidate",
    "dongaragasshan",
    "drive_your_heart", "exist", "firebird", "girlscode", "gunjou_biyori",
    "hachibouseidansu", "hachigatsunoif", "happyhappyparty",
    "georgettemegeorgetteyou", "hashirihajimetabakarinokimini", "hekitenbansou",
    "hellorhell", "hellowink", "heroic_advent", "hidamariroodonaito",
    "hoshinoyakusoku", "introduction",
    "jibun_restart", "jumpin", "kaijuu_no_hanauta", "kao", "karenorakugaki",
    "kimigahajimaru", "kirayumesingsgirl", "kizunamusic", "komyuchakkafire",
    "light_delight", "live_beyond", "marking", "marunouchisadistic",
    "masuerade_rhapsody_reuest", "mayoiuta", "moonlight_walk", "more_jump_more",
    "moudoku_ga_osou",
    "mugenmyworld", "nakanainakanai", "nesshokustarmine", "nijuu_no_niji",
    "ourai", "pilgrim", "poppindream", "repaint", "requiem_for_fate",
    "ringing_bloom", "riot", "ryuusencontorasuto", "seishuntobecontinued",
    "senkaiurundasora", "shakunetsubonfire", "shin_ai", "shiori",
    "shiruetto_dansu", "songiam", "starbeat", "stepbystep", "sunlitmusical",
    "sweetsban", "talktomytone", "tanebi", "tarinai", "teardrops", "tomorrowsdoor",
    "trash_life", "tuning", "unstoppable", "violetline", "vipmonster",
    "what’s_the_popipa", "xiuwaxiuwa", "yesbang_dream",
]

VAL_IDS = [
    "blackbirthday", "dokimekiexperience", "godknows", "home_street",
    "killkiss", "kiseki", "r", "shinjinruiwakasosekainoyumeomiruka",
    "starttruedreams", "yakusoku",
]

DEV_TEST_IDS = [
    "dododo", "hitoshizuku", "ishizue_no_hanakanmuri", "louder", "sophie",
]

# Backwards-compatible name used by existing scripts. Treat it as dev test:
# useful for historical comparisons, not as the final blind paper test.
TEST_IDS = DEV_TEST_IDS

PAPER_TEST_IDS = [
    "athiscode", "brushupbrassup", "divinespell", "dokidokisingout",
    "futarikoto", "itsuaietara", "kokokarakokokara", "lemonsour",
    "shunkansummerday", "soundscape",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def annotation_end_time(
    segments: List[Tuple[float, float, str]],
) -> float:
    """Return the end time of the last annotated music segment."""
    return max((float(end) for _, end, _ in segments), default=0.0)


def extend_tail_downbeats(
    downbeats: List[float],
    target_end_s: float,
    lookback_bars: int = 8,
    tolerance_s: float = 0.5,
    max_new_downbeats: int = 64,
    min_bar_s: float = 0.5,
    max_bar_s: float = 8.0,
) -> Tuple[List[float], Dict[str, float]]:
    """Extend the final downbeat grid to cover a known tail end time.

    Some songs fade out after drums disappear, so beat/downbeat trackers may
    stop early. For bar-level segmentation, that means the tail simply has no
    bars and cannot be labeled. This function extrapolates only the tail using
    the median of recent bar durations and appends the exact target end as the
    final boundary.
    """
    clean = [float(t) for t in downbeats if t is not None]
    target_end = float(target_end_s)
    if len(clean) < 2:
        return clean, {
            "enabled": True,
            "added_downbeats": 0,
            "original_end": clean[-1] if clean else 0.0,
            "target_end": target_end,
            "bar_duration": 0.0,
        }

    original_end = clean[-1]
    intervals = [
        b - a
        for a, b in zip(clean, clean[1:])
        if min_bar_s <= b - a <= max_bar_s
    ]
    recent = intervals[-max(1, lookback_bars):]
    if not recent:
        return clean, {
            "enabled": True,
            "added_downbeats": 0,
            "original_end": original_end,
            "target_end": target_end,
            "bar_duration": 0.0,
        }

    bar_duration = float(np.median(recent))
    if target_end <= original_end + max(0.0, tolerance_s):
        return clean, {
            "enabled": True,
            "added_downbeats": 0,
            "original_end": original_end,
            "target_end": target_end,
            "bar_duration": bar_duration,
        }

    extended = list(clean)
    added = 0
    while (
        extended[-1] + bar_duration < target_end
        and added < max_new_downbeats
    ):
        extended.append(round(extended[-1] + bar_duration, 6))
        added += 1

    if extended[-1] < target_end and added < max_new_downbeats:
        extended.append(round(target_end, 6))
        added += 1

    return extended, {
        "enabled": True,
        "added_downbeats": added,
        "original_end": original_end,
        "target_end": target_end,
        "bar_duration": bar_duration,
    }


def chunk_starts(length: int, chunk_size: int, stride: int) -> List[int]:
    """Return starts that cover a sequence completely, including its tail."""
    if chunk_size <= 0 or stride <= 0:
        raise ValueError("chunk_size and stride must be positive")
    if length <= 0:
        return [0]
    if length <= chunk_size:
        return [0]
    starts = list(range(0, length - chunk_size + 1, stride))
    tail_start = length - chunk_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Dict,
) -> None:
    """Save weights with experiment metadata."""
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    map_location: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load both new metadata checkpoints and legacy raw state dictionaries."""
    payload = torch.load(path, weights_only=True, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("config", {})
    return payload, {}


def fine_labels_to_coarse(labels: torch.Tensor) -> torch.Tensor:
    """Map fine label IDs to coarse family IDs while preserving ignore labels."""
    coarse = torch.full_like(labels, -1)
    valid = labels >= 0
    if valid.any():
        mapping = torch.tensor(
            FINE_ID_TO_COARSE_ID,
            dtype=torch.long,
            device=labels.device,
        )
        coarse[valid] = mapping[labels[valid]]
    return coarse


def coarse_label_for(fine_label: str) -> str:
    """Return the parent family for one trainable fine label."""
    return FINE_TO_COARSE[fine_label]


def segments_to_coarse(
    segments: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Map fine segments to coarse families and merge adjacent equal families."""
    coarse_segments: List[Tuple[float, float, str]] = []
    for start, end, fine_label in segments:
        coarse_label = FINE_TO_COARSE.get(fine_label)
        if coarse_label is None:
            continue
        if (
            coarse_segments
            and coarse_segments[-1][2] == coarse_label
            and abs(coarse_segments[-1][1] - start) <= 0.05
        ):
            previous = coarse_segments[-1]
            coarse_segments[-1] = (previous[0], end, coarse_label)
        else:
            coarse_segments.append((start, end, coarse_label))
    return coarse_segments


def _hierarchical_logits(output) -> Dict[str, torch.Tensor]:
    """Normalize model output to fine/coarse logits."""
    if isinstance(output, torch.Tensor):
        return {"fine": output}
    if isinstance(output, dict):
        fine = output.get("fine") if "fine" in output else output.get("fine_logits")
        coarse = (
            output.get("coarse")
            if "coarse" in output
            else output.get("coarse_logits")
        )
        if fine is None:
            raise ValueError("Hierarchical model output is missing fine logits")
        result = {"fine": fine}
        if coarse is not None:
            result["coarse"] = coarse
        return result
    if isinstance(output, (tuple, list)) and output:
        result = {"fine": output[0]}
        if len(output) > 1:
            result["coarse"] = output[1]
        return result
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def _coarse_logits(output) -> torch.Tensor:
    """Extract coarse-family logits from a model output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        coarse = (
            output.get("coarse")
            if "coarse" in output
            else output.get("coarse_logits")
        )
        if coarse is None:
            raise ValueError("Model output is missing coarse logits")
        return coarse
    if isinstance(output, (tuple, list)) and output:
        return output[-1]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


@torch.no_grad()
def predict_coarse_logits(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    chunk_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> torch.Tensor:
    """Predict averaged coarse-family logits for one complete song."""
    model.eval()
    length = len(features)
    if length == 0:
        return torch.zeros(0, len(COARSE_LABELS))

    if chunk_size is None or length <= chunk_size:
        logits = _coarse_logits(
            model(features.unsqueeze(0).to(device))
        ).squeeze(0).cpu()
        return logits

    stride = stride or chunk_size
    starts = chunk_starts(length, chunk_size, stride)
    logits_sum = None
    counts = torch.zeros(length, 1)
    for start in starts:
        end = min(start + chunk_size, length)
        logits = _coarse_logits(
            model(features[start:end].unsqueeze(0).to(device))
        ).squeeze(0).cpu()
        if logits_sum is None:
            logits_sum = torch.zeros(length, logits.shape[-1])
        logits_sum[start:end] += logits
        counts[start:end] += 1
    return logits_sum / counts.clamp_min(1)


@torch.no_grad()
def predict_coarse_sequence(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    chunk_size: Optional[int] = None,
    stride: Optional[int] = None,
    postprocess_config: Optional[PostprocessConfig] = None,
) -> torch.Tensor:
    """Predict coarse structure families for one complete song."""
    logits = predict_coarse_logits(
        model,
        features,
        device,
        chunk_size=chunk_size,
        stride=stride,
    )
    return postprocess_coarse_logits(
        logits,
        COARSE_LABELS,
        config=postprocess_config,
    )


@torch.no_grad()
def predict_hierarchical_sequence(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    chunk_size: Optional[int] = None,
    stride: Optional[int] = None,
    coarse_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Predict fine labels and their coarse families for one complete song.

    Fine logits are combined with the probability assigned to their parent
    family. The exported coarse prediction is derived from the final fine
    prediction, which guarantees a valid hierarchy. ``coarse_direct`` keeps the
    independent coarse-head prediction for diagnostics.
    """
    model.eval()
    length = len(features)
    if length == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return {
            "fine": empty,
            "coarse": empty,
            "coarse_direct": empty,
        }

    if chunk_size is None or length <= chunk_size:
        output = _hierarchical_logits(
            model(features.unsqueeze(0).to(device))
        )
        logits = {key: value.squeeze(0).cpu() for key, value in output.items()}
    else:
        stride = stride or chunk_size
        starts = chunk_starts(length, chunk_size, stride)
        logits_sum: Dict[str, torch.Tensor] = {}
        counts = torch.zeros(length, 1)
        for start in starts:
            end = min(start + chunk_size, length)
            output = _hierarchical_logits(
                model(features[start:end].unsqueeze(0).to(device))
            )
            for key, value in output.items():
                chunk_logits = value.squeeze(0).cpu()
                if key not in logits_sum:
                    logits_sum[key] = torch.zeros(length, chunk_logits.shape[-1])
                logits_sum[key][start:end] += chunk_logits
            counts[start:end] += 1
        logits = {
            key: value / counts.clamp_min(1)
            for key, value in logits_sum.items()
        }

    fine_logits = logits["fine"]
    if "coarse" in logits:
        coarse_logits = logits["coarse"]
        parent_ids = torch.tensor(FINE_ID_TO_COARSE_ID, dtype=torch.long)
        parent_log_probs = F.log_softmax(coarse_logits, dim=-1)[:, parent_ids]
        fine_scores = F.log_softmax(fine_logits, dim=-1)
        fine_predictions = (
            fine_scores + coarse_weight * parent_log_probs
        ).argmax(dim=-1)
        coarse_direct = coarse_logits.argmax(dim=-1)
    else:
        fine_predictions = fine_logits.argmax(dim=-1)
        coarse_direct = fine_labels_to_coarse(fine_predictions)

    coarse_predictions = fine_labels_to_coarse(fine_predictions)
    return {
        "fine": fine_predictions,
        "coarse": coarse_predictions,
        "coarse_direct": coarse_direct,
    }


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    chunk_size: Optional[int] = None,
    stride: Optional[int] = None,
    level: str = "fine",
    coarse_weight: float = 0.5,
) -> torch.Tensor:
    """Predict one hierarchy level without dropping frames/bars.

    Overlapping chunk logits are averaged. Keeping songs separate prevents
    recurrent state from crossing song boundaries.
    """
    predictions = predict_hierarchical_sequence(
        model,
        features,
        device,
        chunk_size=chunk_size,
        stride=stride,
        coarse_weight=coarse_weight,
    )
    if level not in predictions:
        raise ValueError(
            f"Unknown prediction level {level!r}; choose from {sorted(predictions)}"
        )
    return predictions[level]
