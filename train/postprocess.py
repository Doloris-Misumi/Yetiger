"""Post-processing for coarse bar-level structure predictions.

The model predicts one label per bar independently-ish. This module adds a
lightweight musical prior at inference time:

1. Smooth neighbouring bar probabilities.
2. Decode with a transition grammar, so label changes are not too jittery.
3. Merge very short interior segments into a more plausible neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_MIN_BARS = {
    "bridge": 2,
    "chorus": 3,
    "instrumental": 2,
    "intro": 2,
    "outro": 2,
    "pre_chorus": 2,
    "verse": 3,
}


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration for inference-time structure post-processing."""

    mode: str = "none"
    smoothing_window: int = 3
    transition_penalty: float = 0.12
    use_transition_grammar: bool = False
    use_min_duration: bool = False
    min_bars: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MIN_BARS)
    )

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def as_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "smoothing_window": self.smoothing_window,
            "transition_penalty": self.transition_penalty,
            "use_transition_grammar": self.use_transition_grammar,
            "use_min_duration": self.use_min_duration,
            "min_bars": dict(self.min_bars),
        }

    def summary(self) -> str:
        if not self.enabled:
            return "none"
        steps = []
        if self.smoothing_window > 1:
            steps.append(f"smoothing(window={self.smoothing_window})")
        if self.use_transition_grammar:
            steps.append(f"grammar(penalty={self.transition_penalty})")
        if self.use_min_duration:
            min_desc = ", ".join(
                f"{label}={bars}" for label, bars in sorted(self.min_bars.items())
            )
            steps.append(f"min-duration({min_desc})")
        return " + ".join(steps) if steps else self.mode


def parse_min_bars(
    spec: Optional[str],
    base: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Parse ``label=bars,label=bars`` overrides for min-duration merging."""
    min_bars = dict(DEFAULT_MIN_BARS if base is None else base)
    if not spec:
        return min_bars
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "min bars must look like label=bars,label=bars; "
                f"got {item!r}"
            )
        label, raw_value = [part.strip() for part in item.split("=", 1)]
        if label not in min_bars:
            raise ValueError(
                f"unknown min-duration label {label!r}; "
                f"known labels: {', '.join(sorted(min_bars))}"
            )
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"min-duration value for {label!r} must be an integer"
            ) from exc
        if value < 1:
            raise ValueError(
                f"min-duration value for {label!r} must be >= 1"
            )
        min_bars[label] = value
    return min_bars


def make_postprocess_config(
    mode: str = "none",
    smoothing_window: int = 3,
    transition_penalty: float = 0.12,
    min_bars_spec: Optional[str] = None,
    min_bars: Optional[Dict[str, int]] = None,
) -> PostprocessConfig:
    """Build a config from CLI-friendly arguments."""
    if mode not in {"none", "smooth", "merge", "full"}:
        raise ValueError(
            "postprocess mode must be one of: none, smooth, merge, full"
        )
    if smoothing_window < 1:
        raise ValueError("postprocess smoothing window must be >= 1")
    if transition_penalty < 0:
        raise ValueError("postprocess transition penalty must be >= 0")

    min_bars = parse_min_bars(min_bars_spec, base=min_bars)
    if mode == "none":
        return PostprocessConfig(
            mode="none",
            smoothing_window=1,
            transition_penalty=0.0,
            use_transition_grammar=False,
            use_min_duration=False,
            min_bars=min_bars,
        )
    if mode == "smooth":
        return PostprocessConfig(
            mode="smooth",
            smoothing_window=smoothing_window,
            transition_penalty=0.0,
            use_transition_grammar=False,
            use_min_duration=False,
            min_bars=min_bars,
        )
    if mode == "merge":
        return PostprocessConfig(
            mode="merge",
            smoothing_window=1,
            transition_penalty=0.0,
            use_transition_grammar=False,
            use_min_duration=True,
            min_bars=min_bars,
        )
    return PostprocessConfig(
        mode="full",
        smoothing_window=smoothing_window,
        transition_penalty=transition_penalty,
        use_transition_grammar=True,
        use_min_duration=True,
        min_bars=min_bars,
    )


def config_from_mapping(raw: Optional[Dict]) -> PostprocessConfig:
    """Load a config from checkpoint metadata."""
    if not raw:
        return make_postprocess_config("none")
    return PostprocessConfig(
        mode=raw.get("mode", "none"),
        smoothing_window=int(raw.get("smoothing_window", 3)),
        transition_penalty=float(raw.get("transition_penalty", 0.12)),
        use_transition_grammar=bool(raw.get("use_transition_grammar", False)),
        use_min_duration=bool(raw.get("use_min_duration", False)),
        min_bars={
            **DEFAULT_MIN_BARS,
            **{key: int(value) for key, value in raw.get("min_bars", {}).items()},
        },
    )


def postprocess_coarse_logits(
    logits: torch.Tensor,
    labels: Sequence[str],
    config: Optional[PostprocessConfig] = None,
) -> torch.Tensor:
    """Turn coarse logits into a cleaned bar-label sequence."""
    if logits.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if config is None or not config.enabled:
        return logits.argmax(dim=-1).cpu()

    labels = list(labels)
    log_probs = _smoothed_log_probs(logits.cpu(), config.smoothing_window)
    if config.use_transition_grammar:
        decoded = _viterbi_decode(log_probs, labels, config.transition_penalty)
    else:
        decoded = log_probs.argmax(dim=-1)
    if config.use_min_duration:
        decoded = _merge_short_segments(decoded, log_probs, labels, config.min_bars)
    return decoded.long().cpu()


def _odd_window(window: int) -> int:
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def _smoothed_log_probs(logits: torch.Tensor, window: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    window = _odd_window(window)
    if window > 1 and len(probs) > 1:
        pad = window // 2
        channels_first = probs.t().unsqueeze(0)
        padded = F.pad(channels_first, (pad, pad), mode="replicate")
        probs = F.avg_pool1d(
            padded,
            kernel_size=window,
            stride=1,
        ).squeeze(0).t()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probs.clamp_min(1e-8).log()


def _transition_penalties(labels: Sequence[str], base: float) -> torch.Tensor:
    n_labels = len(labels)
    penalties = torch.full((n_labels, n_labels), float(base))
    penalties.fill_diagonal_(0.0)
    index = {label: i for i, label in enumerate(labels)}

    def set_penalty(src: str, dst: str, value: float) -> None:
        if src in index and dst in index:
            penalties[index[src], index[dst]] = value

    def set_many(src: str, dsts: Iterable[str], value: float) -> None:
        for dst in dsts:
            set_penalty(src, dst, value)

    good = base * 0.25
    ok = base * 0.5
    awkward = base * 2.5
    very_awkward = base * 5.0

    set_many("intro", ["verse", "instrumental", "chorus"], good)
    set_many("verse", ["pre_chorus"], good)
    set_many("verse", ["chorus", "instrumental"], ok)
    set_many("pre_chorus", ["chorus"], good * 0.5)
    set_many("chorus", ["verse", "bridge", "instrumental", "outro"], ok)
    set_many("instrumental", ["verse", "pre_chorus", "chorus", "bridge"], ok)
    set_many("bridge", ["pre_chorus", "chorus"], good)

    for src in labels:
        if src != "intro":
            set_penalty(src, "intro", very_awkward)
        if src != "outro":
            set_penalty("outro", src, very_awkward)
    set_penalty("pre_chorus", "verse", awkward)
    set_penalty("pre_chorus", "outro", awkward)
    set_penalty("verse", "outro", awkward)
    set_penalty("intro", "outro", very_awkward)
    return penalties


def _position_penalties(
    t: int,
    total: int,
    labels: Sequence[str],
    base: float,
) -> torch.Tensor:
    penalties = torch.zeros(len(labels))
    if total <= 1 or base <= 0:
        return penalties
    frac = t / max(1, total - 1)
    index = {label: i for i, label in enumerate(labels)}

    if "intro" in index and frac > 0.25:
        penalties[index["intro"]] = base * min(2.0, 1.0 + frac)
    if "outro" in index and frac < 0.65:
        penalties[index["outro"]] = base * min(2.0, 1.0 + (0.65 - frac))
    return penalties


def _viterbi_decode(
    log_probs: torch.Tensor,
    labels: Sequence[str],
    transition_penalty: float,
) -> torch.Tensor:
    total, n_labels = log_probs.shape
    transition = _transition_penalties(labels, transition_penalty)
    dp = torch.empty(total, n_labels)
    backpointers = torch.zeros(total, n_labels, dtype=torch.long)

    dp[0] = log_probs[0] - _position_penalties(
        0, total, labels, transition_penalty
    )
    for t in range(1, total):
        scores = dp[t - 1].unsqueeze(1) - transition
        best_scores, best_prev = scores.max(dim=0)
        dp[t] = (
            best_scores
            + log_probs[t]
            - _position_penalties(t, total, labels, transition_penalty)
        )
        backpointers[t] = best_prev

    decoded = torch.empty(total, dtype=torch.long)
    decoded[-1] = int(dp[-1].argmax().item())
    for t in range(total - 1, 0, -1):
        decoded[t - 1] = backpointers[t, decoded[t]]
    return decoded


def _runs(ids: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not ids:
        return []
    runs: List[Tuple[int, int, int]] = []
    start = 0
    current = ids[0]
    for i, label_id in enumerate(ids[1:], start=1):
        if label_id != current:
            runs.append((start, i, current))
            start = i
            current = label_id
    runs.append((start, len(ids), current))
    return runs


def _merge_short_segments(
    decoded: torch.Tensor,
    log_probs: torch.Tensor,
    labels: Sequence[str],
    min_bars: Dict[str, int],
) -> torch.Tensor:
    ids = [int(item) for item in decoded.tolist()]
    if len(ids) <= 1:
        return decoded

    for _ in range(len(ids)):
        runs = _runs(ids)
        if len(runs) <= 1:
            break
        changed = False
        for run_index, (start, end, label_id) in enumerate(runs):
            label = labels[label_id]
            length = end - start
            required = int(min_bars.get(label, 1))
            if length >= required:
                continue

            # Preserve true edge cases: a one-bar pickup intro or a tiny final
            # outro may be musically intentional, and they are cheap to handle
            # downstream.
            if label == "intro" and start == 0:
                continue
            if label == "outro" and end == len(ids):
                continue

            target = _choose_merge_target(
                runs,
                run_index,
                start,
                end,
                log_probs,
            )
            for i in range(start, end):
                ids[i] = target
            changed = True
            break
        if not changed:
            break
    return torch.tensor(ids, dtype=torch.long)


def _choose_merge_target(
    runs: Sequence[Tuple[int, int, int]],
    run_index: int,
    start: int,
    end: int,
    log_probs: torch.Tensor,
) -> int:
    if run_index == 0:
        return runs[run_index + 1][2]
    if run_index == len(runs) - 1:
        return runs[run_index - 1][2]

    left_start, left_end, left_id = runs[run_index - 1]
    right_start, right_end, right_id = runs[run_index + 1]
    left_len = left_end - left_start
    right_len = right_end - right_start
    left_score = float(log_probs[start:end, left_id].mean().item())
    right_score = float(log_probs[start:end, right_id].mean().item())
    # A small context bonus avoids making a long stable section flicker.
    left_score += 0.01 * left_len
    right_score += 0.01 * right_len
    return left_id if left_score >= right_score else right_id
