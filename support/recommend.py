"""Rule-based support-action recommender for YesTiger structure predictions.

Input:
  - prediction JSON exported by ``train/test_bar.py``
  - knowledge/call_mix_library.json

Output:
  - JSON recommendations with section windows, action ids, confidence, risk,
    and reasons.

This is deliberately conservative and explainable. It is the first product
layer after music-structure segmentation, not a learned preference model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BAR_SECONDS = 2.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_SEMANTICS_PATH = PROJECT_ROOT / "knowledge" / "action_semantics.json"


SECTION_CONTEXTS = {
    "intro": ["intro", "long_intro", "high_energy_break"],
    "verse": ["verse", "mid_energy_vocal", "quiet_listening_section"],
    "pre_chorus": ["pre_chorus", "pre_chorus_build", "chorus_entry"],
    "chorus": [
        "chorus",
        "repeated_chorus",
        "final_chorus",
        "rock_chorus",
        "anthem_hook",
    ],
    "post_chorus": [
        "post_chorus",
        "post_chorus_interlude",
        "high_energy_break",
        "chorus_entry",
    ],
    "instrumental": [
        "instrumental_break",
        "instrumental_riff",
        "extended_instrumental",
        "high_energy_break",
        "post_chorus_interlude",
    ],
    "instrumental_break": [
        "instrumental_break",
        "instrumental_riff",
        "extended_instrumental",
        "high_energy_break",
        "post_chorus_interlude",
    ],
    "solo": [
        "solo",
        "instrumental_break",
        "instrumental_riff",
        "high_energy_break",
    ],
    "bridge": ["bridge", "pre_final_chorus_gap", "high_energy_break"],
    "outro": ["outro", "high_energy_outro", "post_chorus_interlude"],
}


CATEGORY_PREFERENCES = {
    "intro": {
        "mix": 0.22,
        "underground_gei": 0.04,
        "rhythmcall": 0.02,
        "keepspace": -0.05,
    },
    "verse": {
        "keepspace": 0.25,
        "rhythmcall": 0.05,
        "mix": -0.35,
        "underground_gei": -0.55,
    },
    "pre_chorus": {
        "mix": 0.24,
        "rhythmcall": 0.12,
        "keepspace": -0.18,
        "underground_gei": -0.65,
    },
    "chorus": {
        "underground_gei": 0.46,
        "rhythmcall": -0.20,
        "mix": -0.25,
        "keepspace": -0.80,
    },
    "post_chorus": {
        "mix": 0.18,
        "underground_gei": 0.18,
        "rhythmcall": 0.10,
        "keepspace": -0.18,
    },
    "instrumental": {
        "mix": 0.25,
        "underground_gei": 0.16,
        "rhythmcall": 0.08,
        "keepspace": -0.15,
    },
    "instrumental_break": {
        "mix": 0.25,
        "underground_gei": 0.16,
        "rhythmcall": 0.08,
        "keepspace": -0.15,
    },
    "solo": {
        "mix": 0.26,
        "rhythmcall": 0.05,
        "keepspace": -0.18,
        "underground_gei": -0.40,
    },
    "bridge": {
        "mix": 0.10,
        "underground_gei": 0.08,
        "rhythmcall": 0.04,
        "keepspace": 0.02,
    },
    "outro": {
        "rhythmcall": 0.12,
        "mix": 0.05,
        "underground_gei": 0.02,
        "keepspace": 0.08,
    },
}


RISK_PENALTY = {
    "low": 0.0,
    "medium": 0.08,
    "high": 0.16,
}


PRECHORUS_MIX_EXCEPTIONS = {"bismarck_mix_first_half", "tsunagaridai_mix"}
FORCE_CHORUS_UNDERGROUND_GEI = True
KEEP_SPACE_FALLBACK_ONLY = True
GEI_CHAIN_LABELS = {"chorus", "post_chorus", "instrumental", "instrumental_break"}
GEI_CHAIN_CONFIDENCE = 0.56
GEI_CHAIN_CONTINUATION_BONUS = 0.06
GEI_CHAIN_NEIGHBOR_INTERRUPTION_PENALTY = 0.08
GEI_CHAIN_BRIDGE_BONUS = 0.12
GEI_CHAIN_INTERRUPTION_PENALTY = 0.16
GEI_PLAN_SLOT_BARS = 8.0
GEI_PLAN_MIN_BARS = 3.5
GEI_PLAN_FULL_CHAIN_THRESHOLD_BARS = 14.0
GEI_PLAN_LENGTH_TOLERANCE_BARS = 0.35
TIMELINE_OVERLAP_EPSILON = 0.05
VERSE_FILL_STEP_BARS = 4.0
VERSE_FILL_TARGET_BARS = 2.0
PRECHORUS_ENTRY_TARGET_BARS = 1.0
VERSE_FILL_ACTION_ORDER = [
    "hai_hai",
    "oi_oi",
    "ppph",
    "clap",
]
PRECHORUS_OPENING_ACTION_ORDER = [
    "ppph",
    "hai_hai",
    "oi_oi",
    "fufu_call",
]
PRECHORUS_MID_ACTION_ORDER = [
    "jp_mix_seigyaku_dt",
    "ainu_second_half_mix",
    "bismarck_mix_first_half",
    "tsunagaridai_mix",
    "fufu_call",
]
PRECHORUS_ENTRY_ACTION_ORDER = [
    "ietora",
    "haiseno_activation",
    "bismarck_mix_first_half",
    "fufu_call",
]
SEMANTIC_CONTEXT_PENALTY = 0.18
SEMANTIC_DISCOURAGED_STRUCTURE_PENALTY = 0.08


@dataclass
class Section:
    index: int
    start: float
    end: float
    label: str
    estimated_bars: float
    bar_seconds: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_action_semantics(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None or not path.exists():
        return {}
    payload = _read_json(path)
    actions = payload.get("actions", {})
    return actions if isinstance(actions, dict) else {}


def _flatten_strings(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        strings = []
        for key, item in value.items():
            strings.append(str(key))
            strings.extend(_flatten_strings(item))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for item in value:
            strings.extend(_flatten_strings(item))
        return strings
    return [str(value)]


def _song_context_from_prediction(prediction: Dict) -> Dict:
    context: Dict = {}
    song = prediction.get("song")
    if isinstance(song, dict):
        context.update(song)
    for key in (
        "song_id",
        "title",
        "artist",
        "franchise",
        "characters",
        "song_tags",
        "tags",
        "live_policy",
        "venue_policy",
        "audience_experience",
        "song_tradition",
    ):
        if key in prediction and key not in context:
            context[key] = prediction[key]
    return context


def _context_values_for_requirement(song_context: Dict, requirement_key: str) -> List[str]:
    fields_by_kind = {
        "franchise": ["franchise", "franchises", "project"],
        "artist": ["artist", "artists", "unit", "group"],
        "character": ["character", "characters", "focused_character", "member", "members"],
        "song_tag": ["song_tags", "tags", "title", "song_id"],
        "metadata": list(song_context.keys()),
        "live_policy": ["live_policy", "policy_tags", "venue_policy"],
        "venue_policy": ["venue_policy", "policy_tags", "live_policy"],
        "audience_experience": ["audience_experience", "audience_tags", "audience"],
        "song_tradition": ["song_tradition", "song_traditions", "known_calls", "tags", "song_tags"],
    }
    kind = requirement_key.removesuffix("_any")
    fields = fields_by_kind.get(kind, [kind])
    values = []
    for field in fields:
        if field in song_context:
            values.extend(_flatten_strings(song_context.get(field)))
    return [item.lower() for item in values if str(item).strip()]


def _matches_context_requirement(song_context: Dict, requires_context: Dict) -> bool:
    if not requires_context:
        return True
    for key, required_values in requires_context.items():
        haystack = "\n".join(_context_values_for_requirement(song_context, key))
        if not haystack:
            continue
        for value in _flatten_strings(required_values):
            needle = str(value).strip().lower()
            if needle and needle in haystack:
                return True
    return False


def _semantic_context_missing(semantics: Optional[Dict], song_context: Dict) -> bool:
    if not semantics:
        return False
    requires_context = semantics.get("requires_context") or {}
    return bool(requires_context) and not _matches_context_requirement(
        song_context,
        requires_context,
    )


def _passes_semantic_filters(
    action: Dict,
    semantics: Optional[Dict],
    section: Section,
    song_context: Dict,
) -> bool:
    if not semantics:
        return True
    if section.label in set(semantics.get("forbidden_structures", [])):
        return False
    allowed = semantics.get("allowed_structures") or []
    if allowed and section.label not in set(allowed):
        return False
    if _semantic_context_missing(semantics, song_context):
        return semantics.get("if_context_missing") != "forbid"
    return True


def _semantic_score_adjustment(
    semantics: Optional[Dict],
    section: Section,
    song_context: Dict,
) -> Tuple[float, List[str], List[str]]:
    if not semantics:
        return 0.0, [], []
    delta = 0.0
    reasons = []
    warnings = []
    if section.label in set(semantics.get("discouraged_structures", [])):
        delta -= SEMANTIC_DISCOURAGED_STRUCTURE_PENALTY
        reasons.append("semantic=discouraged_structure")
    if (
        _semantic_context_missing(semantics, song_context)
        and semantics.get("if_context_missing") == "penalize"
    ):
        delta -= SEMANTIC_CONTEXT_PENALTY
        reasons.append("semantic=context_missing_penalty")
        warnings.append("semantic context missing; action kept with penalty")
    return delta, reasons, warnings


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _duration(segment: Dict) -> float:
    return float(segment["end"]) - float(segment["start"])


def _segment_end(prediction: Dict) -> float:
    segments = prediction.get("predicted", [])
    return max((float(seg["end"]) for seg in segments), default=0.0)


def _extend_downbeats_to_end(
    downbeats: List[float],
    target_end: float,
    lookback: int = 8,
) -> List[float]:
    clean = [float(t) for t in downbeats if t is not None]
    if len(clean) < 2:
        return clean
    intervals = [
        b - a
        for a, b in zip(clean, clean[1:])
        if 0.5 <= b - a <= 8.0
    ]
    if not intervals:
        return clean
    step = sorted(intervals[-max(1, lookback):])[len(intervals[-max(1, lookback):]) // 2]
    extended = list(clean)
    while extended[-1] + step < target_end:
        extended.append(round(extended[-1] + step, 6))
    if extended[-1] < target_end:
        extended.append(round(target_end, 6))
    return extended


def _bar_count_from_downbeats(
    start: float,
    end: float,
    downbeats: Sequence[float],
) -> Optional[int]:
    if len(downbeats) < 2:
        return None
    count = 0
    for t0, t1 in zip(downbeats, downbeats[1:]):
        overlap = max(0.0, min(end, t1) - max(start, t0))
        if overlap >= 0.5 * max(1e-8, t1 - t0):
            count += 1
    return count or None


def _infer_sections(
    prediction: Dict,
    struct_path: Optional[Path],
    bar_seconds: Optional[float],
) -> Tuple[List[Section], Dict]:
    segments = prediction.get("predicted", [])
    meta = {
        "bar_estimation": "fixed_bar_seconds",
        "bar_seconds": float(bar_seconds or DEFAULT_BAR_SECONDS),
    }

    downbeats = None
    if struct_path is not None and struct_path.exists():
        struct = _read_json(struct_path)
        downbeats = _extend_downbeats_to_end(
            struct.get("downbeats", []),
            _segment_end(prediction),
        )
        meta = {
            "bar_estimation": "struct_downbeats",
            "struct": str(struct_path),
            "downbeat_count": len(downbeats),
        }

    sections: List[Section] = []
    for index, seg in enumerate(segments, start=1):
        start = float(seg["start"])
        end = float(seg["end"])
        dur = max(0.0, end - start)
        if downbeats is not None:
            bar_count = _bar_count_from_downbeats(start, end, downbeats)
            if bar_count:
                local_bar_seconds = dur / max(bar_count, 1)
                estimated_bars = float(bar_count)
            else:
                local_bar_seconds = float(bar_seconds or DEFAULT_BAR_SECONDS)
                estimated_bars = dur / max(local_bar_seconds, 1e-8)
        else:
            local_bar_seconds = float(bar_seconds or DEFAULT_BAR_SECONDS)
            estimated_bars = dur / max(local_bar_seconds, 1e-8)
        sections.append(Section(
            index=index,
            start=start,
            end=end,
            label=str(seg["label"]),
            estimated_bars=estimated_bars,
            bar_seconds=local_bar_seconds,
        ))
    return sections, meta


def _action_allowed_bars(action: Dict) -> List[float]:
    requires = action.get("requires", {})
    allowed = requires.get("allowed_bars") or []
    return [float(item) for item in allowed]


def _preferred_bars(action: Dict) -> float:
    duration = action.get("duration", {})
    requires = action.get("requires", {})
    if "preferred_bars" in duration:
        return float(duration["preferred_bars"])
    allowed = _action_allowed_bars(action)
    if allowed:
        return float(allowed[0])
    return float(requires.get("min_bars", 1.0))


def _fit_bars(action: Dict, section_bars: float) -> Optional[float]:
    requires = action.get("requires", {})
    min_bars = float(requires.get("min_bars", 0.0))
    if section_bars + 1e-6 < min_bars:
        return None

    allowed = sorted(_action_allowed_bars(action))
    if allowed:
        fit = [bars for bars in allowed if bars <= section_bars + 1e-6]
        if not fit:
            return None
        preferred = _preferred_bars(action)
        return min(fit, key=lambda bars: abs(bars - preferred))

    preferred = _preferred_bars(action)
    return min(preferred, section_bars)


def _context_score(action: Dict, contexts: Sequence[str]) -> Tuple[float, List[str]]:
    action_contexts = set(action.get("best_context", []))
    matched = sorted(action_contexts.intersection(contexts))
    if matched:
        return 0.22 + min(0.08, 0.02 * len(matched)), matched
    return -0.08, []


def _is_context_candidate(
    action: Dict,
    section: Section,
    contexts: Sequence[str],
    enable_underground_gei: bool,
) -> bool:
    category = action.get("category", "")
    if category == "underground_gei" and not enable_underground_gei:
        return False
    if FORCE_CHORUS_UNDERGROUND_GEI and section.label == "chorus":
        return category == "underground_gei"
    if section.label == "pre_chorus" and category == "underground_gei":
        return False
    if section.label == "solo" and category == "underground_gei":
        return False
    if category == "keepspace":
        return section.label in {"verse", "bridge", "outro"}
    if section.label == "pre_chorus" and action.get("id") in PRECHORUS_MIX_EXCEPTIONS:
        return True
    action_contexts = set(action.get("best_context", []))
    if action_contexts.intersection(contexts):
        return True
    # Product decision: chorus may use underground-gei as an optional high-risk
    # action even when the action's context list is incomplete.
    if section.label == "chorus" and category == "underground_gei":
        return True
    return False


def _window_for_action(section: Section, fit_bars: float) -> Tuple[float, float]:
    end = min(section.end, section.start + fit_bars * section.bar_seconds)
    if end <= section.start:
        end = section.end
    return round(section.start, 2), round(end, 2)


def _coverage_bars_for_action(action: Dict, section: Section, fit_bars: float) -> float:
    category = action.get("category", "")
    if FORCE_CHORUS_UNDERGROUND_GEI and section.label == "chorus" and category == "underground_gei":
        return section.estimated_bars
    if category == "keepspace":
        return section.estimated_bars
    return fit_bars


def _recommend_for_section(
    section: Section,
    actions: Sequence[Dict],
    action_semantics: Dict[str, Dict],
    song_context: Dict,
    max_per_section: int,
    enable_underground_gei: bool,
    min_confidence: float,
) -> List[Dict]:
    contexts = SECTION_CONTEXTS.get(section.label, [section.label])
    candidates = []
    category_prefs = CATEGORY_PREFERENCES.get(section.label, {})

    for action in actions:
        category = action.get("category", "")
        semantics = action_semantics.get(str(action.get("id", "")))
        if not _passes_semantic_filters(action, semantics, section, song_context):
            continue
        if not _is_context_candidate(action, section, contexts, enable_underground_gei):
            continue
        fit_bars = _fit_bars(action, section.estimated_bars)
        if fit_bars is None:
            continue

        context_bonus, matched_contexts = _context_score(action, contexts)
        semantic_delta, semantic_reasons, semantic_warnings = _semantic_score_adjustment(
            semantics,
            section,
            song_context,
        )
        category_bonus = category_prefs.get(category, -0.05)
        risk = str(action.get("risk", "medium"))
        risk_penalty = RISK_PENALTY.get(risk, 0.08)
        duration_fit = _clamp(fit_bars / max(_preferred_bars(action), 1e-8), 0, 1)
        intensity = float(action.get("intensity", 0.5))
        score = (
            0.48
            + context_bonus
            + category_bonus
            + 0.10 * duration_fit
            + 0.03 * min(intensity, 1.0)
            + semantic_delta
            - risk_penalty
        )
        if section.label == "chorus" and category == "underground_gei":
            score += 0.06
        if section.label == "pre_chorus" and category == "mix":
            score += 0.08
        if section.label == "solo" and category == "mix":
            score += 0.08
        if section.label == "verse" and category in {"mix", "underground_gei"}:
            score -= 0.20

        coverage_bars = _coverage_bars_for_action(action, section, fit_bars)
        start, end = _window_for_action(section, coverage_bars)
        warnings = []
        if risk == "high":
            warnings.append("high-risk action; venue/fandom policy should override")
        if category == "underground_gei" and section.label == "chorus":
            warnings.append("chorus is forced to underground-gei for dense callbook mode")
        if section.label == "solo" and category == "mix":
            warnings.append("solo MIX assumes clear bar lines and stable beat")
        warnings.extend(semantic_warnings)
        if section.estimated_bars < _preferred_bars(action):
            warnings.append("section shorter than preferred action length")

        reason_bits = [
            f"section={section.label}",
            f"estimated_bars={section.estimated_bars:.1f}",
        ]
        if matched_contexts:
            reason_bits.append("matched_context=" + ",".join(matched_contexts))
        if (
            section.label == "pre_chorus"
            and action.get("id") in PRECHORUS_MIX_EXCEPTIONS
        ):
            reason_bits.append("pre_chorus_mix_exception=allowed")
        if coverage_bars > fit_bars + 1e-6:
            reason_bits.append("coverage=full_section")
        reason_bits.extend(semantic_reasons)
        reason_bits.append(f"category={category}")

        confidence = round(_clamp(score), 3)
        if confidence < min_confidence:
            continue

        candidates.append({
            "section_index": section.index,
            "section_label": section.label,
            "section_start": round(section.start, 2),
            "section_end": round(section.end, 2),
            "section_estimated_bars": round(section.estimated_bars, 2),
            "start": start,
            "end": end,
            "fit_bars": round(coverage_bars, 2),
            "action_fit_bars": round(fit_bars, 2),
            "action_id": action.get("id"),
            "action_name": action.get("display_name", action.get("id")),
            "category": category,
            "risk": risk,
            "confidence": confidence,
            "reason": "; ".join(reason_bits),
            "warnings": warnings,
        })

    if KEEP_SPACE_FALLBACK_ONLY:
        active_candidates = [
            item for item in candidates if item.get("category") != "keepspace"
        ]
        if active_candidates:
            candidates = active_candidates

    candidates.sort(
        key=lambda item: (
            item["confidence"],
            -RISK_PENALTY.get(item["risk"], 0.08),
        ),
        reverse=True,
    )
    return _diverse_top(candidates, max_per_section)


def _gei_action_already_used(item: Dict, used_gei_action_ids: Optional[set]) -> bool:
    return (
        used_gei_action_ids is not None
        and item.get("category") == "underground_gei"
        and item.get("action_id") in used_gei_action_ids
    )


def _mark_gei_action_used(item: Dict, used_gei_action_ids: Optional[set]) -> None:
    if used_gei_action_ids is not None and item.get("category") == "underground_gei":
        used_gei_action_ids.add(item.get("action_id"))


def _diverse_top(
    candidates: Sequence[Dict],
    max_items: int,
    used_gei_action_ids: Optional[set] = None,
) -> List[Dict]:
    selected = []
    used_categories = set()
    for item in candidates:
        if len(selected) >= max_items:
            break
        if _gei_action_already_used(item, used_gei_action_ids):
            continue
        if item["category"] in used_categories:
            continue
        selected.append(item)
        used_categories.add(item["category"])
        _mark_gei_action_used(item, used_gei_action_ids)
    for item in candidates:
        if len(selected) >= max_items:
            break
        if _gei_action_already_used(item, used_gei_action_ids):
            continue
        if item not in selected:
            selected.append(item)
            _mark_gei_action_used(item, used_gei_action_ids)
    return selected


def _append_reason(item: Dict, bit: str) -> None:
    reason = item.get("reason", "")
    parts = [part for part in reason.split("; ") if part]
    if bit not in parts:
        parts.append(bit)
    item["reason"] = "; ".join(parts)


def _append_warning(item: Dict, warning: str) -> None:
    warnings = item.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)


def _adjust_confidence(item: Dict, delta: float) -> None:
    item["confidence"] = round(_clamp(float(item["confidence"]) + delta), 3)


def _has_strong_gei_candidate(items: Sequence[Dict]) -> bool:
    return any(
        item.get("category") == "underground_gei"
        and float(item.get("confidence", 0.0)) >= GEI_CHAIN_CONFIDENCE
        for item in items
    )


def _apply_gei_chain_coherence(
    recommendations: Sequence[Dict],
    sections: Sequence[Section],
    max_per_section: int,
    min_confidence: float,
    enforce_gei_uniqueness: bool = True,
) -> List[Dict]:
    by_section: Dict[int, List[Dict]] = {section.index: [] for section in sections}
    for item in recommendations:
        by_section.setdefault(int(item["section_index"]), []).append(dict(item))

    strong_gei = {
        section.index: (
            section.label in GEI_CHAIN_LABELS
            and _has_strong_gei_candidate(by_section.get(section.index, []))
        )
        for section in sections
    }

    for pos, section in enumerate(sections):
        if section.label not in GEI_CHAIN_LABELS:
            continue
        items = by_section.get(section.index, [])
        if not items:
            continue

        prev_has_gei = pos > 0 and strong_gei.get(sections[pos - 1].index, False)
        next_has_gei = (
            pos + 1 < len(sections)
            and strong_gei.get(sections[pos + 1].index, False)
        )
        if not (prev_has_gei or next_has_gei):
            continue

        has_gei_here = any(item.get("category") == "underground_gei" for item in items)
        for item in items:
            category = item.get("category")
            if category == "underground_gei":
                _adjust_confidence(item, GEI_CHAIN_CONTINUATION_BONUS)
                _append_reason(item, "gei_chain=continues_neighbor_gei")
                continue

            if has_gei_here:
                _adjust_confidence(item, -GEI_CHAIN_NEIGHBOR_INTERRUPTION_PENALTY)
                _append_reason(item, "gei_chain=competes_with_neighbor_gei")

            if prev_has_gei and next_has_gei and has_gei_here:
                _adjust_confidence(item, -GEI_CHAIN_INTERRUPTION_PENALTY)
                _append_reason(item, "gei_chain=interrupts_between_gei")
                _append_warning(
                    item,
                    "interrupts underground-gei chain between neighboring sections",
                )

        if prev_has_gei and next_has_gei and has_gei_here:
            for item in items:
                if item.get("category") == "underground_gei":
                    _adjust_confidence(item, GEI_CHAIN_BRIDGE_BONUS)
                    _append_reason(item, "gei_chain=bridges_neighbor_gei")

    reranked: List[Dict] = []
    used_gei_action_ids: Optional[set] = set() if enforce_gei_uniqueness else None
    for section in sections:
        items = [
            item
            for item in by_section.get(section.index, [])
            if float(item.get("confidence", 0.0)) >= min_confidence
        ]
        items.sort(
            key=lambda item: (
                item["confidence"],
                -RISK_PENALTY.get(item["risk"], 0.08),
            ),
            reverse=True,
        )
        reranked.extend(_diverse_top(
            items,
            max_per_section,
            used_gei_action_ids=used_gei_action_ids,
        ))
    return reranked


def _group_by_section(items: Sequence[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for item in items:
        grouped.setdefault(int(item.get("section_index") or 0), []).append(dict(item))
    for candidates in grouped.values():
        candidates.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                -RISK_PENALTY.get(str(item.get("risk", "medium")), 0.08),
                float(item.get("action_fit_bars") or item.get("fit_bars") or 0.0),
            ),
            reverse=True,
        )
    return grouped


def _is_underground_gei(item: Dict) -> bool:
    return item.get("category") == "underground_gei"


def _item_action_bars(item: Dict) -> float:
    return float(item.get("action_fit_bars") or item.get("fit_bars") or 0.0)


def _intervals_overlap(
    start: float,
    end: float,
    intervals: Sequence[Tuple[float, float]],
) -> bool:
    return any(
        start < existing_end - TIMELINE_OVERLAP_EPSILON
        and end > existing_start + TIMELINE_OVERLAP_EPSILON
        for existing_start, existing_end in intervals
    )


def _sections_overlapped_by_window(
    sections: Sequence[Section],
    start: float,
    end: float,
) -> List[Section]:
    return [
        section
        for section in sections
        if start < section.end - TIMELINE_OVERLAP_EPSILON
        and end > section.start + TIMELINE_OVERLAP_EPSILON
    ]


def _section_at_time(sections: Sequence[Section], time_s: float) -> Section:
    for section in sections:
        if section.start - TIMELINE_OVERLAP_EPSILON <= time_s < section.end - TIMELINE_OVERLAP_EPSILON:
            return section
    return sections[-1]


def _gei_chains(sections: Sequence[Section]) -> List[List[Section]]:
    chains: List[List[Section]] = []
    current: List[Section] = []
    for section in sections:
        if section.label in GEI_CHAIN_LABELS:
            current.append(section)
            continue
        if current:
            chains.append(current)
            current = []
    if current:
        chains.append(current)
    return chains


def _chain_bar_seconds(chain: Sequence[Section]) -> float:
    total_bars = sum(max(0.0, section.estimated_bars) for section in chain)
    duration = max(0.0, chain[-1].end - chain[0].start)
    if total_bars <= 1e-6 or duration <= 1e-6:
        return DEFAULT_BAR_SECONDS
    return duration / total_bars


def _chain_gei_candidates(
    by_section: Dict[int, List[Dict]],
    chain: Sequence[Section],
) -> List[Dict]:
    candidates: List[Dict] = []
    seen: set = set()
    for section in chain:
        for item in by_section.get(section.index, []):
            if not _is_underground_gei(item):
                continue
            key = (
                item.get("action_id"),
                int(item.get("section_index") or 0),
                _item_action_bars(item),
            )
            if key in seen:
                continue
            candidates.append(dict(item))
            seen.add(key)
    return candidates


def _has_candidate_near_bars(
    candidates: Sequence[Dict],
    bars: float,
) -> bool:
    return any(
        abs(_item_action_bars(item) - bars) <= GEI_PLAN_LENGTH_TOLERANCE_BARS
        for item in candidates
    )


def _gei_slot_targets(total_bars: float, candidates: Sequence[Dict]) -> List[float]:
    if total_bars < GEI_PLAN_MIN_BARS:
        return []
    if total_bars < GEI_PLAN_FULL_CHAIN_THRESHOLD_BARS:
        if total_bars >= 10.0 and _has_candidate_near_bars(candidates, 12.0):
            return [min(12.0, total_bars)]
        if total_bars >= 6.0:
            return [min(GEI_PLAN_SLOT_BARS, total_bars)]
        return [total_bars]

    slot_count = int(total_bars // GEI_PLAN_SLOT_BARS)
    targets = [GEI_PLAN_SLOT_BARS] * max(1, slot_count)
    remainder = total_bars - GEI_PLAN_SLOT_BARS * slot_count
    if remainder >= GEI_PLAN_MIN_BARS:
        targets.append(remainder)
    return targets


def _select_gei_candidate_for_slot(
    candidates: Sequence[Dict],
    target_bars: float,
    remaining_bars: float,
    used_gei_action_ids: set,
) -> Optional[Dict]:
    usable = []
    for item in candidates:
        if item.get("action_id") in used_gei_action_ids:
            continue
        action_bars = _item_action_bars(item)
        if action_bars < GEI_PLAN_MIN_BARS:
            continue
        if action_bars > remaining_bars + GEI_PLAN_LENGTH_TOLERANCE_BARS:
            continue
        usable.append(item)
    if not usable:
        return None

    usable.sort(
        key=lambda item: (
            -abs(_item_action_bars(item) - target_bars),
            float(item.get("confidence", 0.0)),
            -RISK_PENALTY.get(str(item.get("risk", "medium")), 0.08),
            _item_action_bars(item),
        ),
        reverse=True,
    )
    return dict(usable[0])


def _planned_gei_item(
    candidate: Dict,
    chain: Sequence[Section],
    slot_start: float,
    slot_end: float,
    slot_bars: float,
    slot_index: int,
    slot_count: int,
) -> Dict:
    item = dict(candidate)
    active_section = _section_at_time(chain, slot_start)
    covered_sections = _sections_overlapped_by_window(chain, slot_start, slot_end)
    item.update({
        "section_index": active_section.index,
        "section_label": active_section.label,
        "section_start": round(active_section.start, 2),
        "section_end": round(active_section.end, 2),
        "section_estimated_bars": round(active_section.estimated_bars, 2),
        "start": round(slot_start, 2),
        "end": round(slot_end, 2),
        "fit_bars": round(slot_bars, 2),
        "arrangement_role": "primary",
        "arrangement_mode": "non_overlapping_gei_chain",
        "arrangement_slot": f"{slot_index + 1}/{slot_count}",
        "planned_section_indices": [section.index for section in covered_sections],
        "planned_section_labels": [section.label for section in covered_sections],
    })
    _append_reason(item, "arrangement=non_overlapping_gei_chain")
    _append_reason(item, f"arrangement_slot={slot_index + 1}/{slot_count}")
    if len(covered_sections) > 1:
        _append_reason(item, "arrangement=borrows_adjacent_section")
        _append_warning(
            item,
            "planned action crosses adjacent structure boundary to complete a bar phrase",
        )
    return item


def _plan_gei_chain(
    chain: Sequence[Section],
    candidates: Sequence[Dict],
    used_gei_action_ids: set,
) -> List[Dict]:
    total_bars = sum(max(0.0, section.estimated_bars) for section in chain)
    if not candidates:
        return []
    if not any(section.label == "chorus" for section in chain) and not _has_strong_gei_candidate(candidates):
        return []

    targets = _gei_slot_targets(total_bars, candidates)
    if not targets:
        return []

    bar_seconds = _chain_bar_seconds(chain)
    chain_start = chain[0].start
    cursor_bars = 0.0
    planned: List[Dict] = []

    for slot_index, target_bars in enumerate(targets):
        remaining_bars = total_bars - cursor_bars
        if remaining_bars < GEI_PLAN_MIN_BARS:
            break
        target_bars = min(target_bars, remaining_bars)
        candidate = _select_gei_candidate_for_slot(
            candidates,
            target_bars,
            remaining_bars,
            used_gei_action_ids,
        )
        if candidate is None:
            break

        action_bars = _item_action_bars(candidate)
        slot_bars = min(action_bars, remaining_bars)
        slot_start = chain_start + cursor_bars * bar_seconds
        slot_end = min(chain[-1].end, slot_start + slot_bars * bar_seconds)
        if slot_end <= slot_start:
            break

        planned_item = _planned_gei_item(
            candidate,
            chain,
            slot_start,
            slot_end,
            slot_bars,
            slot_index,
            len(targets),
        )
        planned.append(planned_item)
        used_gei_action_ids.add(candidate.get("action_id"))
        cursor_bars += slot_bars

    return planned


def _best_single_candidate(
    candidates: Sequence[Dict],
    used_gei_action_ids: set,
    occupied: Sequence[Tuple[float, float]],
) -> Optional[Dict]:
    for item in candidates:
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        if end <= start:
            continue
        if _intervals_overlap(start, end, occupied):
            continue
        if _is_underground_gei(item) and item.get("action_id") in used_gei_action_ids:
            continue
        selected = dict(item)
        selected["arrangement_role"] = "primary"
        selected["arrangement_mode"] = "single_section_primary"
        selected["planned_section_indices"] = [int(selected.get("section_index") or 0)]
        selected["planned_section_labels"] = [str(selected.get("section_label") or "unknown")]
        _append_reason(selected, "arrangement=single_section_primary")
        if _is_underground_gei(selected):
            used_gei_action_ids.add(selected.get("action_id"))
        return selected
    return None


def _select_slot_candidate(
    candidates: Sequence[Dict],
    action_order: Sequence[str],
    target_bars: float,
    available_bars: float,
    excluded_action_ids: Optional[set] = None,
) -> Optional[Dict]:
    order_rank = {action_id: index for index, action_id in enumerate(action_order)}
    excluded_action_ids = excluded_action_ids or set()
    usable = []
    for item in candidates:
        action_id = str(item.get("action_id") or "")
        if action_id not in order_rank:
            continue
        if action_id in excluded_action_ids:
            continue
        action_bars = _item_action_bars(item)
        if action_bars <= 0:
            continue
        if action_bars > available_bars + GEI_PLAN_LENGTH_TOLERANCE_BARS:
            continue
        usable.append(item)
    if not usable:
        return None
    usable.sort(
        key=lambda item: (
            abs(_item_action_bars(item) - target_bars),
            order_rank[str(item.get("action_id") or "")],
            -float(item.get("confidence", 0.0)),
            RISK_PENALTY.get(str(item.get("risk", "medium")), 0.08),
        )
    )
    return dict(usable[0])


def _planned_section_fill_item(
    candidate: Dict,
    section: Section,
    start_bar: float,
    slot_bars: float,
    arrangement_mode: str,
    slot_name: str,
) -> Dict:
    item = dict(candidate)
    start = section.start + start_bar * section.bar_seconds
    end = min(section.end, start + slot_bars * section.bar_seconds)
    item.update({
        "section_index": section.index,
        "section_label": section.label,
        "section_start": round(section.start, 2),
        "section_end": round(section.end, 2),
        "section_estimated_bars": round(section.estimated_bars, 2),
        "start": round(start, 2),
        "end": round(end, 2),
        "fit_bars": round(max(0.0, (end - start) / max(section.bar_seconds, 1e-8)), 2),
        "arrangement_role": "primary",
        "arrangement_mode": arrangement_mode,
        "arrangement_slot": slot_name,
        "planned_section_indices": [section.index],
        "planned_section_labels": [section.label],
    })
    _append_reason(item, f"arrangement={arrangement_mode}")
    _append_reason(item, f"arrangement_slot={slot_name}")
    return item


def _add_section_fill_slot(
    planned: List[Dict],
    section: Section,
    candidates: Sequence[Dict],
    action_order: Sequence[str],
    start_bar: float,
    target_bars: float,
    available_bars: float,
    occupied: Sequence[Tuple[float, float]],
    used_action_ids: set,
    arrangement_mode: str,
    slot_name: str,
) -> Optional[Dict]:
    if available_bars < 0.75:
        return None
    candidate = _select_slot_candidate(
        candidates,
        action_order,
        target_bars=target_bars,
        available_bars=available_bars,
        excluded_action_ids=used_action_ids,
    )
    if candidate is None:
        candidate = _select_slot_candidate(
            candidates,
            action_order,
            target_bars=target_bars,
            available_bars=available_bars,
        )
    if candidate is None:
        return None
    slot_bars = min(_item_action_bars(candidate), available_bars)
    item = _planned_section_fill_item(
        candidate,
        section,
        start_bar=start_bar,
        slot_bars=slot_bars,
        arrangement_mode=arrangement_mode,
        slot_name=slot_name,
    )
    start = float(item.get("start") or 0.0)
    end = float(item.get("end") or start)
    if end <= start or _intervals_overlap(start, end, occupied):
        return None
    planned.append(item)
    used_action_ids.add(str(item.get("action_id") or ""))
    return item


def _plan_verse_fill(
    section: Section,
    candidates: Sequence[Dict],
    occupied: Sequence[Tuple[float, float]],
) -> List[Dict]:
    planned: List[Dict] = []
    used_action_ids: set = set()
    start_bar = 0.0
    slot_index = 1
    while start_bar + 0.75 <= section.estimated_bars:
        available = min(VERSE_FILL_TARGET_BARS, section.estimated_bars - start_bar)
        if available < 1.0:
            break
        item = _add_section_fill_slot(
            planned,
            section,
            candidates,
            VERSE_FILL_ACTION_ORDER,
            start_bar=start_bar,
            target_bars=VERSE_FILL_TARGET_BARS,
            available_bars=available,
            occupied=[*occupied, *[
                (float(item.get("start") or 0.0), float(item.get("end") or 0.0))
                for item in planned
            ]],
            used_action_ids=used_action_ids,
            arrangement_mode="verse_sparse_call_fill",
            slot_name=f"verse_fill_{slot_index}",
        )
        if item is not None:
            slot_index += 1
        start_bar += VERSE_FILL_STEP_BARS
    return planned


def _plan_prechorus_fill(
    section: Section,
    candidates: Sequence[Dict],
    occupied: Sequence[Tuple[float, float]],
) -> List[Dict]:
    planned: List[Dict] = []
    used_action_ids: set = set()
    local_occupied = lambda: [
        (float(item.get("start") or 0.0), float(item.get("end") or 0.0))
        for item in planned
    ]
    total_bars = section.estimated_bars

    if total_bars >= 2.0:
        _add_section_fill_slot(
            planned,
            section,
            candidates,
            PRECHORUS_OPENING_ACTION_ORDER,
            start_bar=0.0,
            target_bars=2.0,
            available_bars=min(2.0, total_bars),
            occupied=[*occupied, *local_occupied()],
            used_action_ids=used_action_ids,
            arrangement_mode="prechorus_build_fill",
            slot_name="opening_call",
        )

    entry_bars = PRECHORUS_ENTRY_TARGET_BARS
    entry_start = max(0.0, total_bars - entry_bars)
    if total_bars >= 4.0:
        _add_section_fill_slot(
            planned,
            section,
            candidates,
            PRECHORUS_ENTRY_ACTION_ORDER,
            start_bar=entry_start,
            target_bars=entry_bars,
            available_bars=min(2.0, total_bars - entry_start),
            occupied=[*occupied, *local_occupied()],
            used_action_ids=used_action_ids,
            arrangement_mode="prechorus_chorus_entry_fill",
            slot_name="chorus_entry",
        )

    mid_start = 2.0
    mid_end = entry_start if total_bars >= 4.0 else total_bars
    mid_available = max(0.0, mid_end - mid_start)
    if mid_available >= 1.5:
        target = 4.0 if mid_available >= 3.5 else 2.0
        _add_section_fill_slot(
            planned,
            section,
            candidates,
            PRECHORUS_MID_ACTION_ORDER,
            start_bar=mid_start,
            target_bars=target,
            available_bars=mid_available,
            occupied=[*occupied, *local_occupied()],
            used_action_ids=used_action_ids,
            arrangement_mode="prechorus_build_mix_fill",
            slot_name="middle_build",
        )

    planned.sort(key=lambda item: float(item.get("start") or 0.0))
    return planned


def _plan_section_fills(
    section: Section,
    candidates: Sequence[Dict],
    occupied: Sequence[Tuple[float, float]],
) -> List[Dict]:
    if section.label == "verse":
        return _plan_verse_fill(section, candidates, occupied)
    if section.label == "pre_chorus":
        return _plan_prechorus_fill(section, candidates, occupied)
    return []


def _build_primary_action_plan(
    candidate_pool: Sequence[Dict],
    sections: Sequence[Section],
) -> List[Dict]:
    by_section = _group_by_section(candidate_pool)
    plan: List[Dict] = []
    occupied: List[Tuple[float, float]] = []
    used_gei_action_ids: set = set()

    for chain in _gei_chains(sections):
        candidates = _chain_gei_candidates(by_section, chain)
        planned_chain = _plan_gei_chain(chain, candidates, used_gei_action_ids)
        for item in planned_chain:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if _intervals_overlap(start, end, occupied):
                continue
            plan.append(item)
            occupied.append((start, end))

    for section in sections:
        if section.label not in {"verse", "pre_chorus"}:
            continue
        candidates = by_section.get(section.index, [])
        for item in _plan_section_fills(section, candidates, occupied):
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if _intervals_overlap(start, end, occupied):
                continue
            plan.append(item)
            occupied.append((start, end))

    for section in sections:
        candidates = by_section.get(section.index, [])
        selected = _best_single_candidate(candidates, used_gei_action_ids, occupied)
        if selected is None:
            continue
        start = float(selected.get("start") or 0.0)
        end = float(selected.get("end") or start)
        plan.append(selected)
        occupied.append((start, end))

    plan.sort(
        key=lambda item: (
            float(item.get("start") or 0.0),
            float(item.get("end") or 0.0),
            int(item.get("section_index") or 0),
        )
    )
    return plan


def recommend(
    prediction_path: Path,
    library_path: Path,
    struct_path: Optional[Path] = None,
    action_semantics_path: Optional[Path] = DEFAULT_ACTION_SEMANTICS_PATH,
    song_context: Optional[Dict] = None,
    bar_seconds: Optional[float] = None,
    max_per_section: int = 3,
    enable_underground_gei: bool = True,
    min_confidence: float = 0.5,
) -> Dict:
    prediction = _read_json(prediction_path)
    library = _read_json(library_path)
    actions = library.get("actions", [])
    action_semantics = _load_action_semantics(action_semantics_path)
    resolved_song_context = _song_context_from_prediction(prediction)
    if song_context:
        resolved_song_context.update(song_context)
    sections, bar_meta = _infer_sections(prediction, struct_path, bar_seconds)

    candidate_pool = []
    candidate_pool_size = max(max_per_section * 4, max_per_section + 8, 16)
    for section in sections:
        candidate_pool.extend(_recommend_for_section(
            section,
            actions,
            action_semantics=action_semantics,
            song_context=resolved_song_context,
            max_per_section=candidate_pool_size,
            enable_underground_gei=enable_underground_gei,
            min_confidence=min_confidence,
        ))
    candidate_pool = _apply_gei_chain_coherence(
        candidate_pool,
        sections,
        max_per_section=candidate_pool_size,
        min_confidence=min_confidence,
        enforce_gei_uniqueness=False,
    )
    recommendations = _build_primary_action_plan(candidate_pool, sections)

    return {
        "song_id": prediction.get("song_id", prediction_path.stem),
        "source_prediction": str(prediction_path),
        "library": str(library_path),
        "bar_estimation": bar_meta,
        "policy": {
            "max_per_section": max_per_section,
            "min_confidence": min_confidence,
            "action_semantics": str(action_semantics_path) if action_semantics_path else None,
            "semantic_actions_loaded": len(action_semantics),
            "enable_underground_gei": enable_underground_gei,
            "chorus_underground_gei": "forced_full_section_dense_callbook",
            "keepspace_policy": "fallback_only_when_no_active_candidate_survives",
            "pre_chorus_mix": "preferred_tension_ramp",
            "solo_mix": "preferred_clear_barline_instrumental",
            "underground_gei_chain_coherence": "rerank_to_avoid_gei_callmix_gei_interruptions",
            "arrangement_policy": "non_overlapping_primary_timeline",
            "gei_arrangement": "plan adjacent chorus/post_chorus/instrumental chains as bar-length slots, preferring 8-bar underground-gei phrases and borrowing adjacent sections when needed",
            "verse_prechorus_fill": "sparse non-overlapping call/mix slots: verse prefers generic military-call style rhythmcalls; pre_chorus uses opening calls, build MIX, and optional chorus-entry Ietora/Haiseno-style accents",
            "underground_gei_action_uniqueness": "same underground-gei action_id appears at most once per song in the primary plan",
            "candidate_pool_size_per_section": candidate_pool_size,
        },
        "recommendations": recommendations,
    }


def _iter_prediction_paths(path: Optional[Path], pred_dir: Optional[Path]) -> List[Path]:
    if path is not None:
        return [path]
    if pred_dir is None:
        raise ValueError("Either --prediction or --pred-dir is required")
    return sorted(pred_dir.glob("*.prediction.json"))


def _default_struct_path(prediction_path: Path, struct_dir: Optional[Path]) -> Optional[Path]:
    if struct_dir is None:
        return None
    song_id = prediction_path.name.replace(".prediction.json", "")
    candidate = struct_dir / f"{song_id}.json"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend support actions from YesTiger structure predictions."
    )
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("knowledge/call_mix_library.json"),
    )
    parser.add_argument(
        "--action-semantics",
        type=Path,
        default=DEFAULT_ACTION_SEMANTICS_PATH,
        help="Structured action semantics used for hard filters and context penalties.",
    )
    parser.add_argument("--struct", type=Path, default=None)
    parser.add_argument("--struct-dir", type=Path, default=None)
    parser.add_argument("--bar-seconds", type=float, default=None)
    parser.add_argument("--max-per-section", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("support/recommendations"))
    parser.add_argument(
        "--disable-underground-gei",
        action="store_true",
        help="Disable high-risk underground-gei candidates.",
    )
    args = parser.parse_args()

    paths = _iter_prediction_paths(args.prediction, args.pred_dir)
    if args.max_per_section < 1:
        parser.error("--max-per-section must be >= 1")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")

    for prediction_path in paths:
        struct_path = args.struct or _default_struct_path(
            prediction_path,
            args.struct_dir,
        )
        payload = recommend(
            prediction_path=prediction_path,
            library_path=args.library,
            struct_path=struct_path,
            action_semantics_path=args.action_semantics,
            bar_seconds=args.bar_seconds,
            max_per_section=args.max_per_section,
            enable_underground_gei=not args.disable_underground_gei,
            min_confidence=args.min_confidence,
        )
        if args.out is not None and len(paths) == 1:
            out_path = args.out
        else:
            song_id = payload["song_id"]
            out_path = args.out_dir / f"{song_id}.support.json"
        _write_json(out_path, payload)
        print(f"Recommendations saved to {out_path}")


if __name__ == "__main__":
    main()
