"""Current YesTiger web analysis pipeline.

The old LOSO/tiny-pipeline web analyzer is archived under ``webapp/_archive``.
This module keeps the same server-facing API while routing uploads through the
current frozen bar-level coarse structure model and the support recommendation
engine.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
SUPPORT_DIR = ROOT / "support"
STATIC_EXAMPLES_DIR = THIS_DIR / "static" / "examples"
RUN_DIR = Path(os.environ.get("YESTIGER_RUN_DIR") or (ROOT / "webapp_runs")).resolve()
UPLOAD_DIR = RUN_DIR / "uploads"
JOB_DIR = RUN_DIR / "jobs"
WEB_CACHE_DIR = RUN_DIR / "feature_cache"
LIBRARY_PATH = ROOT / "knowledge" / "call_mix_library.json"
FROZEN_CANDIDATE_PATH = TRAIN_DIR / "frozen_candidate.json"
CUSTOM_ACTION_DIR = RUN_DIR / "custom_actions"

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

for path in (TRAIN_DIR, SUPPORT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from postprocess import config_from_mapping  # noqa: E402
from recommend import recommend as recommend_support  # noqa: E402
from research_utils import extend_tail_downbeats, load_checkpoint  # noqa: E402
from train_bar import (  # noqa: E402
    COARSE_LABELS,
    SAMPLE_RATE,
    StructureBiLSTM,
    append_bar_context_features,
    bar_context_feature_dim,
    bar_pooling,
    coarse_bars_to_segments,
    create_feature_extractor,
    load_audio,
)
from research_utils import predict_coarse_sequence  # noqa: E402


ROLE_COLORS = {
    "keepspace": "keepspace",
    "rhythmcall": "rhythmcall",
    "mix": "mix",
    "underground_gei": "underground_gei",
}


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(value).strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "uploaded_song"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fmt_time(seconds: float) -> str:
    safe = max(0.0, float(seconds or 0.0))
    minutes = int(safe // 60)
    secs = safe - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


_action_library_cache: Dict[str, Any] = {}
_action_library_mtime: float = 0.0


def action_library_by_id() -> Dict[str, Dict[str, Any]]:
    global _action_library_cache, _action_library_mtime
    mtime = LIBRARY_PATH.stat().st_mtime if LIBRARY_PATH.exists() else 0.0
    if _action_library_cache and abs(mtime - _action_library_mtime) < 0.1:
        return _action_library_cache
    data = read_json(LIBRARY_PATH)
    _action_library_cache = {
        str(action["id"]): action
        for action in data.get("actions", [])
        if isinstance(action, dict) and action.get("id")
    }
    _action_library_mtime = mtime
    return _action_library_cache


def custom_action_library_by_id() -> Dict[str, Dict[str, Any]]:
    actions: Dict[str, Dict[str, Any]] = {}
    if not CUSTOM_ACTION_DIR.exists():
        return actions
    for path in sorted(CUSTOM_ACTION_DIR.glob("*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        action_id = str(payload.get("id") or "").strip()
        if not action_id:
            continue
        item = dict(payload)
        item.setdefault("source", "user_custom")
        actions[action_id] = item
    return actions


def merged_action_library_by_id() -> Dict[str, Dict[str, Any]]:
    merged = dict(action_library_by_id())
    merged.update(custom_action_library_by_id())
    return merged


def normalize_custom_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("custom action payload must be an object")
    action_id = slugify(payload.get("id") or payload.get("display_name") or "custom_mix")
    display_name = str(payload.get("display_name") or action_id).strip()
    category = str(payload.get("category") or "mix").strip()
    if category not in {"mix", "rhythmcall", "underground_gei", "keepspace"}:
        raise ValueError(f"unknown action category: {category}")
    risk = str(payload.get("risk") or "medium").strip()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    typical_text = str(payload.get("typical_text") or "").strip()
    tutorial_text = payload.get("tutorial_text")
    if not typical_text and isinstance(tutorial_text, dict):
        bars = tutorial_text.get("bars")
        if isinstance(bars, list):
            typical_text = " / ".join(str(item).strip() for item in bars if str(item).strip())
    if not typical_text:
        raise ValueError("custom action needs typical_text or tutorial_text.bars")

    action = dict(payload)
    action.update({
        "id": action_id,
        "display_name": display_name,
        "category": category,
        "risk": risk,
        "typical_text": typical_text,
        "source": "user_custom",
    })
    if tutorial_text is not None:
        action["tutorial_text"] = tutorial_text
    return action


def save_custom_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = normalize_custom_action(payload)
    CUSTOM_ACTION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CUSTOM_ACTION_DIR / f"{action['id']}.json", action)
    return action


def list_custom_actions() -> List[Dict[str, Any]]:
    actions = custom_action_library_by_id()
    return [
        actions[action_id]
        for action_id in sorted(actions)
    ]


def _candidate_config() -> Dict[str, Any]:
    if FROZEN_CANDIDATE_PATH.exists():
        return read_json(FROZEN_CANDIDATE_PATH)
    return {
        "run_id": "best_model_bar_coarse",
        "checkpoint": "best_model_bar_coarse.pt",
        "model": {
            "backbone": "mert",
            "target_level": "coarse",
            "pool_mode": "meanmaxstd",
            "feature_layers": [4, 8, 12],
            "mert_layers": [4, 8, 12],
        },
        "inference": {
            "postprocess": {"mode": "merge"},
            "tail_downbeat_extension": {
                "enabled": True,
                "lookback_bars": 8,
                "tolerance_s": 0.5,
                "max_new_downbeats": 64,
            },
        },
    }


def _checkpoint_path(config: Dict[str, Any]) -> Path:
    raw = str(config.get("checkpoint") or "best_model_bar_coarse.pt")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (TRAIN_DIR / path).resolve()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _feature_layers_from_config(
    checkpoint_config: Dict[str, Any],
    candidate_model: Dict[str, Any],
    backbone: str,
) -> Optional[List[int]]:
    layers = (
        checkpoint_config.get("feature_layers")
        or checkpoint_config.get(f"{backbone}_layers")
        or checkpoint_config.get("mert_layers")
        or checkpoint_config.get("muq_layers")
        or candidate_model.get("feature_layers")
        or candidate_model.get(f"{backbone}_layers")
    )
    if layers is None:
        return None
    return [int(item) for item in layers]


def _backbone_label(backbone: str) -> str:
    labels = {
        "mert": "MERT",
        "muq": "MuQ",
    }
    return labels.get(str(backbone).lower(), str(backbone).upper())


@lru_cache(maxsize=1)
def _model_bundle() -> Dict[str, Any]:
    candidate = _candidate_config()
    checkpoint = _checkpoint_path(candidate)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Frozen model checkpoint not found: {checkpoint}"
        )

    device = _device()
    state_dict, checkpoint_config = load_checkpoint(
        checkpoint,
        map_location=device,
    )
    target_level = checkpoint_config.get("target_level", "coarse")
    if target_level != "coarse":
        raise ValueError(
            f"Web app expects a coarse checkpoint, got {target_level!r}"
        )
    input_dim = int(state_dict["lstm.weight_ih_l0"].shape[1])
    model = StructureBiLSTM(
        input_dim=input_dim,
        hidden_dim=int(checkpoint_config.get("hidden_dim", 256)),
        num_layers=int(checkpoint_config.get("num_layers", 2)),
        dropout=float(checkpoint_config.get("dropout", 0.5)),
        target_level="coarse",
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    candidate_model = candidate.get("model") or {}
    backbone = str(
        checkpoint_config.get("backbone")
        or candidate_model.get("backbone")
        or "mert"
    ).lower()
    pool_mode = checkpoint_config.get(
        "pool_mode",
        candidate_model.get("pool_mode", "meanmaxstd"),
    )
    feature_layers = _feature_layers_from_config(
        checkpoint_config,
        candidate_model,
        backbone,
    )
    muq_chunk_seconds = float(
        checkpoint_config.get(
            "muq_chunk_seconds",
            candidate_model.get("muq_chunk_seconds", 30.0),
        )
    )
    muq_overlap_seconds = float(
        checkpoint_config.get(
            "muq_overlap_seconds",
            candidate_model.get("muq_overlap_seconds", 1.0),
        )
    )
    postprocess_config = config_from_mapping(
        checkpoint_config.get(
            "postprocess",
            (candidate.get("inference") or {}).get("postprocess"),
        )
    )

    return {
        "candidate": candidate,
        "checkpoint": checkpoint,
        "checkpoint_config": checkpoint_config,
        "device": device,
        "model": model,
        "backbone": backbone,
        "backbone_model": checkpoint_config.get("backbone_model")
        or candidate_model.get("backbone_model"),
        "feature_layers": feature_layers,
        "mert_layers": feature_layers if backbone == "mert" else None,
        "muq_layers": feature_layers if backbone == "muq" else None,
        "muq_chunk_seconds": muq_chunk_seconds,
        "muq_overlap_seconds": muq_overlap_seconds,
        "pool_mode": pool_mode,
        "postprocess_config": postprocess_config,
        "bar_context_features": checkpoint_config.get(
            "bar_context_features",
            candidate_model.get("bar_context_features", "none"),
        ),
    }


@lru_cache(maxsize=4)
def _feature_extractor(
    backbone: str,
    layers_key: Tuple[int, ...],
    muq_chunk_seconds: float,
    muq_overlap_seconds: float,
):
    layers = list(layers_key) if layers_key else None
    return create_feature_extractor(
        backbone=backbone,
        device=_device(),
        feature_layers=layers,
        model_cache_dir=ROOT / ".hf",
        muq_chunk_seconds=muq_chunk_seconds,
        muq_overlap_seconds=muq_overlap_seconds,
    )


def _resolve_allin1_executable() -> str:
    candidates = [
        ROOT / ".venv" / "Scripts" / "allin1.exe",
        ROOT / ".venv" / "bin" / "allin1",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    for command in ("allin1", "allin1.exe"):
        resolved = shutil.which(command)
        if resolved:
            return resolved
    raise FileNotFoundError("allin1 executable not found in .venv or PATH")


def run_allin1(audio_path: Path, output_dir: Path, timeout: Optional[int] = None) -> Dict[str, Any]:
    if timeout is None:
        timeout = int(os.environ.get("YESTIGER_ALLIN1_TIMEOUT", "600"))
    output_dir.mkdir(parents=True, exist_ok=True)
    allin1_exe = _resolve_allin1_executable()
    for cache_dir in (
        ROOT / ".cache" / "matplotlib",
        ROOT / ".cache" / "huggingface",
        ROOT / ".cache" / "torch",
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)
    demix_dir = ROOT / "demix" / "htdemucs"
    demix_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib"),
        "HF_HOME": str(ROOT / ".cache" / "huggingface"),
        "TORCH_HOME": str(ROOT / ".cache" / "torch"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    device = "cpu"
    result = subprocess.run(
        [
            allin1_exe,
            str(audio_path),
            "-o",
            str(output_dir),
            "--demix-dir",
            str(demix_dir),
            "-k",
            "--no-multiprocess",
            "-d",
            device,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "allin1 failed: " + (result.stderr or result.stdout)[-1200:]
        )
    struct_files = sorted(output_dir.glob("*.json"))
    if not struct_files:
        raise FileNotFoundError(f"allin1 produced no JSON in {output_dir}")
    return read_json(struct_files[0])


def _regular_downbeats(duration_s: float, bar_seconds: float = 2.4) -> List[float]:
    count = max(2, int(duration_s / max(0.5, bar_seconds)) + 1)
    downbeats = [round(index * bar_seconds, 6) for index in range(count)]
    if not downbeats or downbeats[0] != 0.0:
        downbeats.insert(0, 0.0)
    if downbeats[-1] < duration_s:
        downbeats.append(round(duration_s, 6))
    else:
        downbeats[-1] = round(duration_s, 6)
    return downbeats


def _stored_struct_for_song(song_id: str) -> Optional[Dict[str, Any]]:
    candidate = TRAIN_DIR / "struct" / f"{song_id}.json"
    if candidate.exists():
        return read_json(candidate)
    return None


def _struct_for_audio(
    audio_path: Path,
    job_id: str,
    song_id: str,
    duration_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stored = _stored_struct_for_song(song_id)
    if stored is not None:
        return stored, {
            "status": "stored_train_struct",
            "detail": f"train/struct/{song_id}.json",
        }
    struct_dir = JOB_DIR / job_id / "struct"
    try:
        struct = run_allin1(audio_path, struct_dir)
        return struct, {
            "status": "allin1",
            "detail": "allin1 downbeat grid",
        }
    except Exception as exc:
        downbeats = _regular_downbeats(duration_s)
        return {
            "bpm": 100.0,
            "beats": [],
            "downbeats": downbeats,
        }, {
            "status": "regular_fallback",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _tail_extension_options() -> Dict[str, Any]:
    candidate = _candidate_config()
    raw = (candidate.get("inference") or {}).get("tail_downbeat_extension") or {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "lookback_bars": int(raw.get("lookback_bars", 8)),
        "tolerance_s": float(raw.get("tolerance_s", 0.5)),
        "max_new_downbeats": int(raw.get("max_new_downbeats", 64)),
    }


def _predict_structure(
    audio_path: Path,
    job_id: str,
    song_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    bundle = _model_bundle()
    waveform = load_audio(audio_path)
    duration_s = float(len(waveform) / SAMPLE_RATE)
    struct, struct_status = _struct_for_audio(audio_path, job_id, song_id, duration_s)

    raw_downbeats = [float(item) for item in struct.get("downbeats", [])]
    tail_options = _tail_extension_options()
    if tail_options["enabled"]:
        downbeats, tail_stats = extend_tail_downbeats(
            raw_downbeats,
            target_end_s=duration_s,
            lookback_bars=tail_options["lookback_bars"],
            tolerance_s=tail_options["tolerance_s"],
            max_new_downbeats=tail_options["max_new_downbeats"],
        )
    else:
        downbeats = raw_downbeats
        tail_stats = {
            "enabled": False,
            "added_downbeats": 0,
            "original_end": raw_downbeats[-1] if raw_downbeats else 0.0,
            "target_end": duration_s,
            "bar_duration": 0.0,
        }
    if len(downbeats) < 2:
        downbeats = _regular_downbeats(duration_s)
    struct = {**struct, "downbeats": downbeats}

    layers = bundle["feature_layers"] or []
    extractor = _feature_extractor(
        bundle["backbone"],
        tuple(int(item) for item in layers),
        float(bundle["muq_chunk_seconds"]),
        float(bundle["muq_overlap_seconds"]),
    )
    frame_features = extractor.extract_all(audio_path, WEB_CACHE_DIR / job_id)
    bar_features = bar_pooling(
        frame_features,
        struct.get("beats", []),
        downbeats,
        duration_s,
        pool_mode=bundle["pool_mode"],
    )
    bar_features = append_bar_context_features(
        bar_features,
        struct.get("beats", []),
        downbeats,
        duration_s,
        mode=bundle["bar_context_features"],
    )
    if len(bar_features) == 0:
        raise ValueError("No bar features could be built for the uploaded audio")

    predictions = predict_coarse_sequence(
        bundle["model"],
        bar_features,
        bundle["device"],
        postprocess_config=bundle["postprocess_config"],
    )
    pred_segments = coarse_bars_to_segments(predictions, downbeats)
    prediction_payload = {
        "song_id": job_id,
        "target_level": "coarse",
        "source": f"{bundle['backbone']}_bar_coarse",
        "predicted": [
            {
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "label": label,
            }
            for start, end, label in pred_segments
            if end > start
        ],
    }
    meta = {
        "duration_s": duration_s,
        "bar_count": max(0, len(downbeats) - 1),
        "tempo": float(struct.get("bpm") or 0.0),
        "struct_status": struct_status,
        "tail_downbeat_extension": tail_stats,
        "downbeats": [round(float(d), 2) for d in downbeats],
        "model": {
            "backbone": bundle["backbone"],
            "backbone_model": bundle["backbone_model"],
            "run_id": bundle["candidate"].get("run_id"),
            "checkpoint": str(bundle["checkpoint"]),
            "pool_mode": bundle["pool_mode"],
            "bar_context_features": bundle["bar_context_features"],
            "bar_context_feature_dim": bar_context_feature_dim(
                bundle["bar_context_features"]
            ),
            "feature_layers": bundle["feature_layers"],
            "mert_layers": bundle["mert_layers"],
            "muq_layers": bundle["muq_layers"],
            "muq_chunk_seconds": bundle["muq_chunk_seconds"],
            "muq_overlap_seconds": bundle["muq_overlap_seconds"],
            "postprocess": bundle["postprocess_config"].as_dict(),
            "labels": COARSE_LABELS,
        },
    }
    return prediction_payload, struct, meta


def _prediction_audio_path(song_id: str) -> Optional[Path]:
    candidate = TRAIN_DIR / "songs" / f"{song_id}.mp3"
    return candidate if candidate.exists() else None


def _annotation_title(song_id: str) -> str:
    path = TRAIN_DIR / "annotations" / song_id / f"{song_id}.annotation.json"
    if not path.exists():
        return song_id
    try:
        data = read_json(path)
    except Exception:
        return song_id
    return str((data.get("song") or {}).get("title") or song_id)


def _prediction_path_for_support(support_payload: Dict[str, Any], song_id: str) -> Path:
    candidate_config = _candidate_config()
    current_backbone = str(
        ((candidate_config.get("model") or {}).get("backbone") or "mert")
    ).lower()
    raw = support_payload.get("source_prediction")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = ROOT / path
        path_text = str(path).replace("\\", "/").lower()
        if path.exists() and (
            current_backbone == "mert" or f"_{current_backbone}" in path_text
        ):
            return path
    for base in (
        TRAIN_DIR / "predictions_bar_coarse_muq_paper_test",
        TRAIN_DIR / "predictions_bar_coarse_muq",
        TRAIN_DIR / "predictions_bar_coarse_paper_test",
        TRAIN_DIR / "predictions_bar_coarse",
    ):
        candidate = base / f"{song_id}.prediction.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction JSON found for {song_id}")


def _support_path_for_song(song_id: str) -> Path:
    candidates = [
        SUPPORT_DIR / "recommendations" / "paper_test" / f"{song_id}.support.json",
        SUPPORT_DIR / "recommendations" / f"{song_id}.support.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No support recommendation found for {song_id}")


def list_example_songs() -> List[Dict[str, str]]:
    examples = []
    paths = sorted((SUPPORT_DIR / "recommendations" / "paper_test").glob("*.support.json"))
    if not paths:
        paths = sorted((SUPPORT_DIR / "recommendations").glob("*.support.json"))
    for path in paths:
        song_id = path.name.replace(".support.json", "")
        examples.append({"song_id": song_id, "title": _annotation_title(song_id)})
    if not examples:
        for path in sorted(STATIC_EXAMPLES_DIR.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            song = payload.get("song") or {}
            song_id = str(song.get("song_id") or path.stem)
            title = str(song.get("title") or song_id)
            examples.append({"song_id": song_id, "title": title})
    return examples


def _music_segments_from_prediction(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = []
    for segment in prediction.get("predicted", []):
        label = str(segment.get("label") or "unknown")
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if end <= start:
            continue
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "music_label": label,
            "struct_label": label,
            "source": str(prediction.get("source") or "bar_coarse"),
        })
    return segments


def _recommendations_by_section(recommendations: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for recommendation in recommendations:
        planned_indices = recommendation.get("planned_section_indices")
        if isinstance(planned_indices, list) and planned_indices:
            indices = [int(index) for index in planned_indices if int(index or 0) > 0]
        else:
            indices = [int(recommendation.get("section_index") or 0)]
        for index in indices:
            if index <= 0:
                continue
            grouped.setdefault(index, []).append(recommendation)
    for items in grouped.values():
        items.sort(
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                -float(item.get("fit_bars") or 0.0),
            ),
            reverse=True,
        )
    return grouped


def _timeline_from_support(recommendations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    library = action_library_by_id()
    timeline = []
    for recommendation in recommendations:
        action_id = str(recommendation.get("action_id") or "")
        action = library.get(action_id, {})
        start = float(recommendation.get("start") or 0.0)
        end = float(recommendation.get("end") or start)
        if end <= start:
            continue
        category = str(recommendation.get("category") or action.get("category") or "keepspace")
        role = ROLE_COLORS.get(category, category)
        planned_labels = recommendation.get("planned_section_labels")
        if isinstance(planned_labels, list) and planned_labels:
            seen_labels = []
            for label in planned_labels:
                label = str(label)
                if label not in seen_labels:
                    seen_labels.append(label)
            section_label = "+".join(seen_labels)
        else:
            section_label = str(recommendation.get("section_label") or "unknown")
        warnings = recommendation.get("warnings") or []
        notes = []
        if recommendation.get("reason"):
            notes.append(str(recommendation["reason"]))
        notes.extend(str(item) for item in warnings)
        timeline.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "time": f"{fmt_time(start)}-{fmt_time(end)}",
            "action_id": action_id,
            "display_name": str(
                recommendation.get("action_name")
                or action.get("display_name")
                or action_id
                or "Keep Space"
            ),
            "role": role,
            "music_label": section_label,
            "struct_label": section_label,
            "risk": str(recommendation.get("risk") or action.get("risk") or "medium"),
            "bar_count": recommendation.get("fit_bars"),
            "typical_text": str(action.get("typical_text") or ""),
            "tutorial_text": action.get("tutorial_text"),
            "confidence": recommendation.get("confidence"),
            "mode": "support_recommendation",
            "section_index": recommendation.get("section_index"),
            "notes": "; ".join(notes),
        })
    timeline.sort(
        key=lambda item: (
            float(item["start"]),
            -float(item.get("confidence") or 0.0),
            float(item["end"]),
        )
    )
    return timeline


def _call_spans_from_support(
    music_segments: Sequence[Dict[str, Any]],
    support_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grouped = _recommendations_by_section(support_payload.get("recommendations", []))
    library = action_library_by_id()
    spans = []
    for index, segment in enumerate(music_segments, start=1):
        recs = grouped.get(index, [])
        primary = recs[0] if recs else {}
        role = str(primary.get("category") or "keepspace")
        action_plan = []
        for rec in recs:
            action_id = str(rec.get("action_id") or "")
            action = library.get(action_id, {})
            start = float(rec.get("start") or segment["start"])
            end = float(rec.get("end") or segment["end"])
            action_plan.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "time": f"{fmt_time(start)}-{fmt_time(end)}",
                "action_id": action_id,
                "display_name": str(
                    rec.get("action_name")
                    or action.get("display_name")
                    or action_id
                ),
                "typical_text": str(action.get("typical_text") or ""),
                "risk": str(rec.get("risk") or action.get("risk") or "medium"),
                "bar_count": rec.get("fit_bars"),
                "tutorial_text": action.get("tutorial_text"),
                "confidence": rec.get("confidence"),
                "mode": "support_recommendation",
            })
        spans.append({
            "start": segment["start"],
            "end": segment["end"],
            "call_role": ROLE_COLORS.get(role, role),
            "music_label_context": segment["music_label"],
            "allin1_struct_context": segment["struct_label"],
            "bar_start": None,
            "bar_end": None,
            "bars": round(float(primary.get("section_estimated_bars") or 0.0), 2),
            "method": "bar_coarse_support",
            "recommended_actions": [
                str(item.get("action_id") or "")
                for item in recs
                if item.get("action_id")
            ],
            "action_plan": action_plan,
            "action_candidates": recs,
            "action_selection": {
                "mode": "support_rule_recommender",
                "min_confidence": (support_payload.get("policy") or {}).get("min_confidence"),
                "max_per_section": (support_payload.get("policy") or {}).get("max_per_section"),
            },
        })
    return spans


def callbook_to_markdown(song_id: str, timeline: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"# YesTiger Tutorial Plan: {song_id}",
        "",
        "| Time | Section | Role | Action | Bars | Risk | Confidence | Text |",
        "|---:|---|---|---|---:|---|---:|---|",
    ]
    for item in timeline:
        confidence = item.get("confidence")
        confidence_text = f"{float(confidence):.3f}" if confidence is not None else "-"
        lines.append(
            f"| {item.get('time')} | {item.get('music_label')} | "
            f"{item.get('role')} | {item.get('display_name')} | "
            f"{item.get('bar_count') if item.get('bar_count') is not None else '-'} | "
            f"{item.get('risk')} | {confidence_text} | "
            f"{item.get('typical_text') or '-'} |"
        )
    return "\n".join(lines) + "\n"


def signal_process_summary(
    rows_count: int,
    music_segment_count: int,
    call_span_count: int,
    timeline_count: int,
    struct_status: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_meta = model_meta or {}
    backbone = str(model_meta.get("backbone") or "mert").lower()
    backbone_label = _backbone_label(backbone)
    layers = model_meta.get("feature_layers") or model_meta.get(f"{backbone}_layers")
    context_mode = str(model_meta.get("bar_context_features") or "none")
    if layers:
        layer_text = "/".join(str(item) for item in layers)
    else:
        layer_text = "default"
    if backbone == "muq":
        feature_detail = (
            f"Extract frozen {backbone_label} layers {layer_text} with chunked "
            "audio inference and pool them to bars."
        )
    else:
        feature_detail = (
            f"Extract frozen {backbone_label} layers {layer_text} and pool them to bars."
        )
    if struct_status and struct_status.get("status") == "allin1":
        downbeat_detail = "allin1 beat/downbeat timing; extend the tail grid when needed."
    elif struct_status and struct_status.get("status") == "stored_train_struct":
        downbeat_detail = f"Pre-computed train struct ({struct_status.get('detail', '')})."
    else:
        downbeat_detail = (
            "allin1 timed out or failed — using uniform 2.4 s/bar fallback grid. "
            "Model still runs but results are degraded."
        )
    steps = [
        {
            "name": "Audio decode",
            "detail": "Load uploaded audio and preserve it for synchronized playback.",
        },
        {
            "name": "Downbeat grid",
            "detail": downbeat_detail,
        },
        {
            "name": f"{backbone_label} embeddings",
            "detail": feature_detail,
        },
        {
            "name": "Structure model",
            "detail": "Apply the frozen 7-class bar-level coarse checkpoint with merge post-processing.",
        },
        {
            "name": "Support planning",
            "detail": "Use the rule-based support recommender and action library to fill each section.",
        },
    ]
    if context_mode != "none":
        steps.insert(3, {
            "name": "Bar context",
            "detail": (
                f"Append {context_mode} timing features "
                f"(dim={bar_context_feature_dim(context_mode)})."
            ),
        })
    summary = {
        "status": "current_model",
        "structure": f"{backbone}_bar_coarse_support",
        "rows": rows_count,
        "music_segments": music_segment_count,
        "call_spans": call_span_count,
        "actions": timeline_count,
        "steps": steps,
    }
    if struct_status and struct_status.get("status") != "allin1":
        summary["status"] = "degraded_grid"
        summary["fallback_reason"] = (
            f"allin1 downbeat detection failed ({struct_status.get('detail', '')}); "
            "MuQ model and support recommender still ran on a uniform 2.4 s/bar grid. "
            "Results will be less accurate — increase YESTIGER_ALLIN1_TIMEOUT if this persists."
        )
    return summary


def build_webapp_result(
    *,
    song_id: str,
    title: str,
    audio_filename: Optional[str],
    prediction: Dict[str, Any],
    support_payload: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    audio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    meta = meta or {}
    model_info = meta.get("model") or {}
    backbone = str(model_info.get("backbone") or "mert").lower()
    backbone_label = _backbone_label(backbone)
    music_segments = _music_segments_from_prediction(prediction)
    call_spans = _call_spans_from_support(music_segments, support_payload)
    timeline = _timeline_from_support(support_payload.get("recommendations", []))
    duration = float(
        meta.get("duration_s")
        or max((float(item["end"]) for item in music_segments), default=0.0)
    )
    result = {
        "job_id": song_id,
        "song": {
            "song_id": song_id,
            "title": title or song_id,
            "audio_filename": audio_filename,
            "duration": round(duration, 3),
            "tempo": round(float(meta.get("tempo") or 0.0), 2) or None,
            "bar_count": int(meta.get("bar_count") or 0),
        },
        "method": {
            "structure": f"{backbone}_bar_coarse_frozen_candidate",
            "actions": "support_rule_recommender",
            "run_id": model_info.get("run_id"),
            "notes": [
                f"Uses the current frozen bar-level {backbone_label} coarse checkpoint.",
                "Support actions are selected by the explainable rule-based recommender.",
            ],
        },
        "pipeline_status": "current_model",
        "signal_process": signal_process_summary(
            int(meta.get("bar_count") or 0),
            len(music_segments),
            len(call_spans),
            len(timeline),
            struct_status=meta.get("struct_status"),
            model_meta=model_info,
        ),
        "music_segments": music_segments,
        "call_spans": call_spans,
        "timeline": timeline,
        "prediction": prediction,
        "support": support_payload,
        "model_meta": _json_ready(meta),
        "markdown": callbook_to_markdown(song_id, timeline),
    }
    if audio_path and audio_path.exists():
        result["audio_path"] = str(audio_path)
    return result


def analyze_audio(
    audio_path: Path,
    title: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = job_id or uuid.uuid4().hex[:12]
    song_id = slugify(title or audio_path.stem)
    work_dir = JOB_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    prediction, struct, meta = _predict_structure(audio_path, job_id, song_id)
    prediction_path = work_dir / "prediction.json"
    struct_path = work_dir / "struct.json"
    write_json(prediction_path, prediction)
    write_json(struct_path, struct)

    support_payload = recommend_support(
        prediction_path=prediction_path,
        library_path=LIBRARY_PATH,
        struct_path=struct_path,
        max_per_section=3,
        enable_underground_gei=True,
        min_confidence=0.5,
    )
    write_json(work_dir / "support.json", support_payload)

    return build_webapp_result(
        song_id=song_id,
        title=title or audio_path.stem,
        audio_filename=audio_path.name,
        prediction=prediction,
        support_payload=support_payload,
        meta=meta,
        audio_path=audio_path,
    )


def save_analysis_result(result: Dict[str, Any], job_dir: Path) -> Tuple[Path, Path]:
    job_dir.mkdir(parents=True, exist_ok=True)
    json_path = job_dir / "result.json"
    md_path = job_dir / "callbook.md"
    write_json(json_path, result)
    md_path.write_text(result.get("markdown", ""), encoding="utf-8")
    if "prediction" in result:
        write_json(job_dir / "prediction.json", result["prediction"])
    if "support" in result:
        write_json(job_dir / "support.json", result["support"])
    return json_path, md_path


def save_curated_result(
    job_id: str,
    music_segments: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    notes: Any = None,
) -> Dict[str, Any]:
    """Save human-curated music segments, timeline, and notes for a job."""
    job_dir = JOB_DIR / slugify(job_id)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    original = read_json(job_dir / "result.json")
    curated = dict(original)
    # notes: array of {start, end, text} or legacy string
    if notes is not None:
        curated["notes"] = notes
    elif "notes" not in curated:
        curated["notes"] = []

    curated["music_segments"] = [
        {
            "start": round(float(seg.get("start", 0)), 2),
            "end": round(float(seg.get("end", 0)), 2),
            "music_label": str(seg.get("music_label") or "unknown"),
            "struct_label": str(seg.get("struct_label") or seg.get("music_label") or "unknown"),
            "source": "human_curated",
            "original_label": seg.get("original_label") or seg.get("music_label"),
        }
        for seg in music_segments
        if float(seg.get("end", 0)) > float(seg.get("start", 0))
    ]

    curated["timeline"] = [
        {
            "start": round(float(item.get("start", 0)), 2),
            "end": round(float(item.get("end", 0)), 2),
            "time": f"{fmt_time(float(item.get('start', 0)))}-{fmt_time(float(item.get('end', 0)))}",
            "action_id": str(item.get("action_id") or ""),
            "display_name": str(item.get("display_name") or "Keep Space"),
            "role": str(item.get("role") or "keepspace"),
            "music_label": str(item.get("music_label") or "unknown"),
            "struct_label": str(item.get("struct_label") or item.get("music_label") or "unknown"),
            "risk": str(item.get("risk") or "medium"),
            "bar_count": item.get("bar_count"),
            "typical_text": str(item.get("typical_text") or ""),
            "tutorial_text": item.get("tutorial_text"),
            "confidence": item.get("confidence"),
            "mode": "human_curated",
            "notes": str(item.get("notes") or ""),
        }
        for item in timeline
        if float(item.get("end", 0)) > float(item.get("start", 0))
    ]

    curated["signal_process"]["status"] = "human_curated"
    curated["pipeline_status"] = "human_curated"
    curated["method"]["actions"] = "human_curated"
    curated["markdown"] = callbook_to_markdown(
        curated.get("song", {}).get("song_id", job_id),
        curated["timeline"],
    )

    write_json(job_dir / "curated_result.json", curated)
    write_json(job_dir / "curated_callbook.md", curated["markdown"])

    return curated


def list_action_library() -> List[Dict[str, Any]]:
    """Return a lightweight action list for the frontend search dropdown."""
    library = merged_action_library_by_id()
    return [
        {
            "id": action_id,
            "display_name": str(action.get("display_name") or action_id),
            "category": str(action.get("category") or "keepspace"),
            "risk": str(action.get("risk") or "medium"),
            "typical_text": str(action.get("typical_text") or ""),
            "tutorial_text": action.get("tutorial_text"),
            "source": str(action.get("source") or "builtin"),
        }
        for action_id, action in sorted(library.items())
    ]


def load_example_result(song_id: str) -> Dict[str, Any]:
    support_path = _support_path_for_song(song_id)
    support_payload = read_json(support_path)
    try:
        prediction_path = _prediction_path_for_support(support_payload, song_id)
        prediction = read_json(prediction_path)
    except FileNotFoundError:
        static_path = STATIC_EXAMPLES_DIR / f"{song_id}.json"
        if static_path.exists():
            result = read_json(static_path)
            result["job_id"] = result.get("job_id") or f"example_{song_id}"
            result.setdefault("downloads", {})
            return result
        raise
    audio_path = _prediction_audio_path(song_id)
    duration = max(
        (float(item.get("end") or 0.0) for item in prediction.get("predicted", [])),
        default=0.0,
    )
    struct_path = TRAIN_DIR / "struct" / f"{song_id}.json"
    struct = read_json(struct_path) if struct_path.exists() else {}
    candidate = _candidate_config()
    meta = {
        "duration_s": duration,
        "bar_count": max(0, len(struct.get("downbeats", [])) - 1),
        "tempo": float(struct.get("bpm") or 0.0),
        "struct_status": {"status": "stored_prediction", "detail": "paper-test prediction"},
        "downbeats": [round(float(d), 2) for d in struct.get("downbeats", [])],
        "model": {
            **(candidate.get("model") or {}),
            "run_id": candidate.get("run_id"),
        },
    }
    result = build_webapp_result(
        song_id=song_id,
        title=_annotation_title(song_id),
        audio_filename=audio_path.name if audio_path else None,
        prediction=prediction,
        support_payload=support_payload,
        meta=meta,
        audio_path=audio_path,
    )
    result["job_id"] = f"example_{song_id}"
    return result
