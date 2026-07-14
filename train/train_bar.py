"""
YesTiger MERT Bar-level Structure Segmentation — Training Script
=================================================================
Pipeline:
  MP3 → MERT-v1-95M (frozen) → frame embeddings (cache)
  struct.json → beats/downbeats → bar pooling
  Annotations → bar-level label mapping
  BiLSTM head → Train/Val/Test → metrics

Usage:
  python train_bar.py --data-dir . --epochs 25 --batch-size 16
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from research_utils import (
    ANNOTATION_ONLY_LABELS,
    COARSE_LABELS,
    MODEL_LABELS,
    TEST_IDS,
    TRAIN_IDS,
    VAL_IDS,
    annotation_end_time,
    chunk_starts,
    coarse_label_for,
    extend_tail_downbeats,
    fine_labels_to_coarse,
    load_checkpoint,
    predict_coarse_sequence,
    predict_hierarchical_sequence,
    save_checkpoint,
    segments_to_coarse,
    set_seed,
)
from postprocess import make_postprocess_config
from segmentation_metrics import segment_level_metrics as compute_segment_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
MUQ_MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
SAMPLE_RATE = 24000
LABELS = MODEL_LABELS
LABEL2ID = {lb: i for i, lb in enumerate(LABELS)}
ID2LABEL = {i: lb for i, lb in enumerate(LABELS)}
COARSE_LABEL2ID = {lb: i for i, lb in enumerate(COARSE_LABELS)}
COARSE_ID2LABEL = {i: lb for i, lb in enumerate(COARSE_LABELS)}
NUM_CLASSES = len(LABELS)
NUM_COARSE_CLASSES = len(COARSE_LABELS)


def _json_ready(value):
    """Convert argparse/config values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_run_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-_.") or "run"


def _tail_extension_config(args) -> Dict:
    return {
        "enabled": bool(args.extend_tail_downbeats),
        "target": "annotation_end",
        "lookback_bars": args.tail_extension_lookback,
        "tolerance_s": args.tail_extension_tolerance,
        "max_new_downbeats": args.tail_extension_max_bars,
    }


def _tail_extension_summary(stats_by_song: Dict[str, Dict]) -> Dict:
    extended = {
        song_id: stats
        for song_id, stats in stats_by_song.items()
        if int(stats.get("added_downbeats", 0)) > 0
    }
    return {
        "songs_extended": len(extended),
        "total_added_downbeats": int(sum(
            int(stats.get("added_downbeats", 0))
            for stats in extended.values()
        )),
        "by_song": extended,
    }


def create_run_directory(
    runs_dir: Path,
    backbone: str,
    target_level: str,
    pool_mode: str,
    feature_layers: Optional[List[int]],
    seed: int,
    run_name: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Create a unique directory for one training run."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        base_name = _safe_run_component(run_name)
    else:
        layer_name = (
            "L" + "-".join(str(layer) for layer in feature_layers)
            if feature_layers
            else "Llast"
        )
        time_name = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = _safe_run_component(
            f"{time_name}_{backbone}_{target_level}_{pool_mode}_{layer_name}_seed{seed}"
        )
    candidate = runs_dir / base_name
    suffix = 2
    while candidate.exists():
        candidate = runs_dir / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def checkpoint_selection_score(
    validation_metrics: Dict,
    target_level: str,
    bar_weight: float = 0.3,
    segment_weight: float = 0.5,
    boundary_weight: float = 0.2,
) -> float:
    """Score a checkpoint using validation-only structural metrics."""
    total_weight = bar_weight + segment_weight + boundary_weight
    if total_weight <= 0:
        raise ValueError("Checkpoint selection weights must sum to a positive value")
    if target_level == "coarse":
        bar_f1 = validation_metrics["macro_f1"]
        segment_f1 = validation_metrics["macro_seg_f1_mean"]
        boundary_f1 = validation_metrics["boundary_f1_3s_mean"]
    else:
        bar_f1 = validation_metrics["coarse_macro_f1"]
        segment_f1 = validation_metrics["coarse_macro_seg_f1_mean"]
        boundary_f1 = validation_metrics["coarse_boundary_f1_3s_mean"]
    return (
        bar_weight * bar_f1
        + segment_weight * segment_f1
        + boundary_weight * boundary_f1
    ) / total_weight


def _add_song_metric_means(result: Dict, per_song: List[Dict]) -> Dict:
    """Attach per-song means used for validation and checkpoint selection."""
    result["per_song"] = per_song
    metric_keys = [
        "macro_seg_f1",
        "boundary_f1_0_5s",
        "boundary_f1_3s",
        "coarse_macro_seg_f1",
        "coarse_boundary_f1_0_5s",
        "coarse_boundary_f1_3s",
    ]
    for key in metric_keys:
        values = [metrics[key] for metrics in per_song if key in metrics]
        if values:
            result[f"{key}_mean"] = round(float(np.mean(values)), 4)
    return result


# ---------------------------------------------------------------------------
# Data loading (shared with frame-level)
# ---------------------------------------------------------------------------

def load_annotation(
    ann_path: Path,
    include_annotation_markers: bool = False,
) -> List[Tuple[float, float, str]]:
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    segs = ann.get("segments", [])
    return [
        (s["start"], s["end"], s["music_label"])
        for s in segs
        if include_annotation_markers
        or s["music_label"] not in ANNOTATION_ONLY_LABELS
    ]


def load_struct(struct_path: Path) -> Dict:
    """Load beat/downbeat info from struct JSON."""
    return json.loads(struct_path.read_text(encoding="utf-8"))


def _load_with_ffmpeg(audio_path: Path, target_sr: int) -> torch.Tensor:
    import subprocess, tempfile, os
    ffmpeg = None
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.venv', 'Scripts', 'ffmpeg.exe'),
        'ffmpeg', 'ffmpeg.exe',
    ]:
        try:
            subprocess.run([candidate, '-version'], capture_output=True, timeout=5)
            ffmpeg = candidate; break
        except Exception:
            continue
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([ffmpeg, '-y', '-i', str(audio_path), '-ar', str(target_sr),
                        '-ac', '1', '-f', 'wav', tmp_path],
                       capture_output=True, timeout=60, check=True)
        import torchaudio
        waveform, sr = torchaudio.load(tmp_path)
        return waveform.squeeze(0)
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass


def load_audio(audio_path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    import torchaudio
    import torchaudio.functional as AF
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        waveform, sr = _load_with_ffmpeg(audio_path, target_sr).unsqueeze(0), target_sr
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)


# ---------------------------------------------------------------------------
# MERT feature extraction (shared)
# ---------------------------------------------------------------------------

class MERTFeatureExtractor:
    def __init__(self, device: str = "cuda", mert_layers: Optional[List[int]] = None):
        """
        mert_layers: which transformer layers to use, e.g. [4, 8, 12].
                     If None, uses last hidden state only (layer 12, 768d).
                     If list, concats specified layers → len(layers)*768 dims.
        """
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        self.device = device
        self.mert_layers = mert_layers
        self.model = None

    def _ensure_model(self):
        """Load MERT only when a requested embedding is not cached."""
        if self.model is not None:
            return
        from transformers import Wav2Vec2Model
        try:
            self.model = Wav2Vec2Model.from_pretrained(
                MERT_MODEL_NAME, use_safetensors=True
            ).to(self.device)
        except Exception:
            import torch as _torch
            _torch_orig = _torch.load
            _torch.load = lambda *a, **kw: _torch_orig(*a, **{**kw, 'weights_only': False})
            self.model = Wav2Vec2Model.from_pretrained(MERT_MODEL_NAME).to(self.device)
            _torch.load = _torch_orig
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        self._ensure_model()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        output = self.model(waveform, output_hidden_states=True)
        if self.mert_layers:
            # Concat specified layers: (T, num_layers*768)
            hs = [output.hidden_states[l].squeeze(0).cpu() for l in self.mert_layers]
            return torch.cat(hs, dim=-1)
        return output.last_hidden_state.squeeze(0).cpu()

    def extract_all(self, audio_path: Path, cache_dir: Path) -> torch.Tensor:
        suffix = f".mert{'_L' + ''.join(str(l) for l in self.mert_layers) if self.mert_layers else ''}.pt"
        cache_path = cache_dir / f"{audio_path.stem}{suffix}"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        waveform = load_audio(audio_path)
        emb = self.extract(waveform)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(emb, cache_path)
        return emb


class MuQFeatureExtractor:
    def __init__(
        self,
        device: str = "cuda",
        muq_layers: Optional[List[int]] = None,
        model_name: str = MUQ_MODEL_NAME,
        model_cache_dir: Optional[Path] = None,
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 1.0,
    ):
        """
        muq_layers: conformer hidden states to concat, e.g. [4, 8, 12].
                    If None, uses last hidden state only.
        """
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        self.device = device
        self.muq_layers = muq_layers
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = max(0.0, float(overlap_seconds))
        self.model = None

    def _ensure_model(self):
        """Load MuQ only when a requested embedding is not cached."""
        if self.model is not None:
            return
        from muq import MuQ
        kwargs = {}
        if self.model_cache_dir is not None:
            kwargs["cache_dir"] = str(self.model_cache_dir)
        self.model = MuQ.from_pretrained(self.model_name, **kwargs).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        self._ensure_model()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.float().to(self.device)
        output = self.model(waveform, output_hidden_states=True)
        if self.muq_layers:
            hidden_states = output.hidden_states
            max_layer = max(self.muq_layers)
            if max_layer >= len(hidden_states):
                raise ValueError(
                    f"MuQ returned {len(hidden_states)} hidden states; "
                    f"cannot select layer {max_layer}."
                )
            hs = [hidden_states[l].squeeze(0).float().cpu() for l in self.muq_layers]
            return torch.cat(hs, dim=-1)
        return output.last_hidden_state.squeeze(0).float().cpu()

    def extract_chunked(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.chunk_seconds <= 0:
            return self.extract(waveform)
        self._ensure_model()
        if waveform.dim() != 1:
            waveform = waveform.squeeze(0)
        total_samples = waveform.numel()
        chunk_samples = max(1, int(round(self.chunk_seconds * SAMPLE_RATE)))
        overlap_samples = max(0, int(round(self.overlap_seconds * SAMPLE_RATE)))
        label_rate = int(getattr(getattr(self.model, "config", None), "label_rate", 25) or 25)
        pieces = []
        start = 0
        while start < total_samples:
            end = min(total_samples, start + chunk_samples)
            chunk_start = max(0, start - overlap_samples)
            chunk_end = min(total_samples, end + overlap_samples)
            chunk = waveform[chunk_start:chunk_end]
            emb = self.extract(chunk)
            trim_left = int(round((start - chunk_start) / SAMPLE_RATE * label_rate))
            trim_right = int(round((chunk_end - end) / SAMPLE_RATE * label_rate))
            if trim_right > 0:
                emb = emb[trim_left:-trim_right]
            else:
                emb = emb[trim_left:]
            if emb.numel() > 0:
                pieces.append(emb)
            start = end
        if not pieces:
            return self.extract(waveform)
        return torch.cat(pieces, dim=0)

    def extract_all(self, audio_path: Path, cache_dir: Path) -> torch.Tensor:
        chunk_tag = (
            f"_c{str(self.chunk_seconds).replace('.', 'p')}"
            f"o{str(self.overlap_seconds).replace('.', 'p')}"
            if self.chunk_seconds > 0
            else "_full"
        )
        suffix = f".muq{'_L' + ''.join(str(l) for l in self.muq_layers) if self.muq_layers else ''}{chunk_tag}.pt"
        cache_path = cache_dir / f"{audio_path.stem}{suffix}"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        waveform = load_audio(audio_path)
        emb = self.extract_chunked(waveform)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(emb, cache_path)
        return emb


def create_feature_extractor(
    backbone: str,
    device: str,
    feature_layers: Optional[List[int]],
    model_cache_dir: Optional[Path] = None,
    muq_chunk_seconds: float = 30.0,
    muq_overlap_seconds: float = 1.0,
):
    if backbone == "mert":
        return MERTFeatureExtractor(device=device, mert_layers=feature_layers)
    if backbone == "muq":
        return MuQFeatureExtractor(
            device=device,
            muq_layers=feature_layers,
            model_cache_dir=model_cache_dir,
            chunk_seconds=muq_chunk_seconds,
            overlap_seconds=muq_overlap_seconds,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


def feature_layers_for_args(args) -> Optional[List[int]]:
    if args.backbone == "muq":
        return args.muq_layers
    return args.mert_layers


def backbone_artifact_suffix(backbone: str) -> str:
    return "" if backbone == "mert" else f"_{backbone}"


# ---------------------------------------------------------------------------
# Bar-level preprocessing
# ---------------------------------------------------------------------------

def bar_pooling(
    frame_embeddings: torch.Tensor,
    beats: List[float],
    downbeats: List[float],
    audio_duration_s: float,
    pool_mode: str = "mean",
) -> torch.Tensor:
    """
    Pool frame embeddings (N, D) into bar embeddings (M, D).
    Each bar spans from downbeats[i] to downbeats[i+1].
    """
    frame_hop = audio_duration_s / max(frame_embeddings.shape[0], 1)
    frame_times = torch.arange(frame_embeddings.shape[0]) * frame_hop

    bar_embs = []
    num_bars = len(downbeats) - 1
    for i in range(num_bars):
        t0, t1 = downbeats[i], downbeats[i + 1]
        mask = (frame_times >= t0) & (frame_times < t1)
        if mask.sum() == 0:
            midpoint = (t0 + t1) / 2
            nearest = int(torch.argmin(torch.abs(frame_times - midpoint)).item())
            frames = frame_embeddings[nearest:nearest + 1]
        else:
            frames = frame_embeddings[mask]
        if pool_mode == "mean":
            bar_embs.append(frames.mean(dim=0))
        elif pool_mode == "meanmax":
            bar_embs.append(torch.cat([frames.mean(dim=0), frames.max(dim=0).values]))
        elif pool_mode == "meanmaxstd":
            bar_embs.append(torch.cat([
                frames.mean(dim=0),
                frames.max(dim=0).values,
                frames.std(dim=0, unbiased=False),
            ]))
    if bar_embs:
        return torch.stack(bar_embs)
    pool_factor = {"mean": 1, "meanmax": 2, "meanmaxstd": 3}[pool_mode]
    return torch.zeros(0, frame_embeddings.shape[1] * pool_factor)


BAR_CONTEXT_FEATURE_CHOICES = ("none", "rhythm", "rhythm_pos")


def _zscore_feature(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() == 0:
        return values
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    mean = values.mean()
    return ((values - mean) / std).clamp(-3.0, 3.0) / 3.0


def build_bar_context_features(
    beats: List[float],
    downbeats: List[float],
    audio_duration_s: float,
    mode: str = "none",
) -> torch.Tensor:
    """Build low-dimensional bar timing features available at inference time."""
    mode = str(mode or "none")
    if mode not in BAR_CONTEXT_FEATURE_CHOICES:
        raise ValueError(
            "bar context features must be one of: "
            + ", ".join(BAR_CONTEXT_FEATURE_CHOICES)
        )

    num_bars = max(0, len(downbeats) - 1)
    if mode == "none":
        return torch.zeros(num_bars, 0, dtype=torch.float32)
    if num_bars <= 0:
        base_dim = 6 if mode == "rhythm" else 13
        return torch.zeros(0, base_dim, dtype=torch.float32)

    starts = torch.tensor(downbeats[:-1], dtype=torch.float32)
    ends = torch.tensor(downbeats[1:], dtype=torch.float32)
    durations = (ends - starts).clamp_min(1e-3)
    mean_duration = durations.mean().clamp_min(1e-3)

    prev_delta = torch.zeros_like(durations)
    next_delta = torch.zeros_like(durations)
    if num_bars > 1:
        prev_delta[1:] = (durations[1:] - durations[:-1]) / mean_duration
        next_delta[:-1] = (durations[1:] - durations[:-1]) / mean_duration

    beat_values = [
        float(item)
        for item in beats
        if item is not None and np.isfinite(float(item))
    ]
    beat_counts = []
    for start, end in zip(downbeats[:-1], downbeats[1:]):
        beat_counts.append(
            sum(1 for beat in beat_values if float(start) <= beat < float(end))
        )
    beat_counts_t = torch.tensor(beat_counts, dtype=torch.float32)
    beat_density = beat_counts_t / durations

    log_duration_ratio = torch.log(durations / mean_duration)
    features = [
        _zscore_feature(durations),
        log_duration_ratio.clamp(-1.5, 1.5) / 1.5,
        prev_delta.clamp(-2.0, 2.0) / 2.0,
        next_delta.clamp(-2.0, 2.0) / 2.0,
        _zscore_feature(beat_counts_t),
        _zscore_feature(beat_density),
    ]

    if mode == "rhythm_pos":
        index = torch.arange(num_bars, dtype=torch.float32)
        denom = max(1, num_bars - 1)
        progress = index / float(denom)
        remaining = 1.0 - progress
        early_decay = torch.exp(-index / 4.0)
        late_decay = torch.exp(-(float(num_bars - 1) - index) / 4.0)
        features.extend([
            progress * 2.0 - 1.0,
            progress,
            remaining,
            early_decay,
            late_decay,
            (index < 4).float(),
            ((float(num_bars - 1) - index) < 4).float(),
        ])

    return torch.stack(features, dim=-1).float()


def append_bar_context_features(
    bar_embeddings: torch.Tensor,
    beats: List[float],
    downbeats: List[float],
    audio_duration_s: float,
    mode: str = "none",
) -> torch.Tensor:
    """Concatenate optional low-dimensional timing features to bar embeddings."""
    context = build_bar_context_features(
        beats=beats,
        downbeats=downbeats,
        audio_duration_s=audio_duration_s,
        mode=mode,
    )
    if context.shape[0] != bar_embeddings.shape[0]:
        raise ValueError(
            f"bar context rows ({context.shape[0]}) != "
            f"bar embeddings ({bar_embeddings.shape[0]})"
        )
    if context.shape[1] == 0:
        return bar_embeddings
    return torch.cat([bar_embeddings, context.to(bar_embeddings.dtype)], dim=-1)


def bar_context_feature_dim(mode: str = "none") -> int:
    if mode == "none" or not mode:
        return 0
    if mode == "rhythm":
        return 6
    if mode == "rhythm_pos":
        return 13
    raise ValueError(
        "bar context features must be one of: "
        + ", ".join(BAR_CONTEXT_FEATURE_CHOICES)
    )


def segments_to_bar_labels(
    segments: List[Tuple[float, float, str]],
    downbeats: List[float],
) -> torch.Tensor:
    """
    Map GT segments to bar-level labels.
    For each bar, find the segment with maximum overlap.
    """
    num_bars = len(downbeats) - 1
    if num_bars <= 0:
        return torch.zeros(0, dtype=torch.long)

    labels = torch.full((num_bars,), -1, dtype=torch.long)
    for i in range(num_bars):
        bar_t0, bar_t1 = downbeats[i], downbeats[i + 1]
        best_overlap = 0.0
        best_label = -1
        for seg_t0, seg_t1, seg_lb in segments:
            if seg_lb not in LABEL2ID:
                continue
            overlap = max(0.0, min(bar_t1, seg_t1) - max(bar_t0, seg_t0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = LABEL2ID[seg_lb]
        labels[i] = best_label
    return labels


def bars_to_segments(
    bar_preds: torch.Tensor,
    downbeats: List[float],
) -> List[Tuple[float, float, str]]:
    """Convert bar predictions back to (start_s, end_s, label) segments."""
    segs = []
    preds = bar_preds.tolist()
    if not preds:
        return segs

    prev = preds[0]
    start = 0
    for i, lb in enumerate(preds):
        if lb != prev:
            end = i
            if prev >= 0 and start < len(downbeats) - 1 and end <= len(downbeats) - 1:
                segs.append((downbeats[start], downbeats[end], ID2LABEL[prev]))
            start = i
            prev = lb
    if prev >= 0 and start < len(downbeats) - 1:
        segs.append((downbeats[start], downbeats[-1], ID2LABEL[prev]))
    return segs


def coarse_bars_to_segments(
    bar_preds: torch.Tensor,
    downbeats: List[float],
) -> List[Tuple[float, float, str]]:
    """Convert coarse bar predictions to merged structure-family segments."""
    segs = []
    preds = bar_preds.tolist()
    if not preds:
        return segs

    prev = preds[0]
    start = 0
    for i, label_id in enumerate(preds):
        if label_id != prev:
            if prev >= 0 and start < len(downbeats) - 1:
                segs.append((
                    downbeats[start],
                    downbeats[i],
                    COARSE_ID2LABEL[prev],
                ))
            start = i
            prev = label_id
    if prev >= 0 and start < len(downbeats) - 1:
        segs.append((downbeats[start], downbeats[-1], COARSE_ID2LABEL[prev]))
    return segs


# ---------------------------------------------------------------------------
# Hierarchical model (same architecture as frame-level)
# ---------------------------------------------------------------------------

class StructureBiLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2,
                 num_classes=NUM_CLASSES,
                 num_coarse_classes=NUM_COARSE_CLASSES, dropout=0.5,
                 target_level: str = "hierarchical"):
        super().__init__()
        if target_level not in {"coarse", "hierarchical"}:
            raise ValueError(
                "target_level must be 'coarse' or 'hierarchical'"
            )
        self.target_level = target_level
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        if target_level == "hierarchical":
            self.fine_classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.coarse_classifier = nn.Linear(hidden_dim * 2, num_coarse_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        result = {"coarse": self.coarse_classifier(out)}
        if self.target_level == "hierarchical":
            result["fine"] = self.fine_classifier(out)
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BarDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 chunk_size: int = 32, stride: int = 16):
        self.features = features
        self.labels = labels
        self.chunk_size = chunk_size
        self.stride = stride
        self.starts = chunk_starts(len(features), chunk_size, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.chunk_size
        x = self.features[start:end]
        y = self.labels[start:end]
        if x.shape[0] < self.chunk_size:
            pad = self.chunk_size - x.shape[0]
            x = F.pad(x, (0, 0, 0, pad))
            y = F.pad(y, (0, pad), value=-1)
        return x, y


# ---------------------------------------------------------------------------
# Metrics (adapted for bar-level)
# ---------------------------------------------------------------------------

def segment_level_metrics(
    pred_segs: List[Tuple[float, float, str]],
    gt_segs: List[Tuple[float, float, str]],
) -> Dict:
    return compute_segment_metrics(pred_segs, gt_segs, LABELS)


def compute_metrics(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    all_coarse_preds: Optional[torch.Tensor] = None,
    downbeats: Optional[List[float]] = None,
    gt_segments: Optional[List[Tuple[float, float, str]]] = None,
) -> Dict:
    mask = all_labels >= 0
    preds = all_preds[mask]
    labels = all_labels[mask]

    if len(labels) == 0:
        return {"accuracy": 0.0, "per_class_f1": {}}

    accuracy = (preds == labels).float().mean().item()

    per_class_f1 = {}
    for lid in range(NUM_CLASSES):
        tp = ((preds == lid) & (labels == lid)).sum().item()
        fp = ((preds == lid) & (labels != lid)).sum().item()
        fn = ((preds != lid) & (labels == lid)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        per_class_f1[ID2LABEL[lid]] = round(2 * prec * rec / (prec + rec + 1e-8), 4)

    macro_f1 = np.mean(list(per_class_f1.values()))
    result = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
    }

    coarse_labels = fine_labels_to_coarse(all_labels)
    if all_coarse_preds is None:
        all_coarse_preds = fine_labels_to_coarse(all_preds)
    coarse_mask = coarse_labels >= 0
    coarse_preds = all_coarse_preds[coarse_mask]
    valid_coarse_labels = coarse_labels[coarse_mask]
    coarse_accuracy = (
        (coarse_preds == valid_coarse_labels).float().mean().item()
        if len(valid_coarse_labels)
        else 0.0
    )
    per_class_coarse_f1 = {}
    for lid, label in enumerate(COARSE_LABELS):
        tp = ((coarse_preds == lid) & (valid_coarse_labels == lid)).sum().item()
        fp = ((coarse_preds == lid) & (valid_coarse_labels != lid)).sum().item()
        fn = ((coarse_preds != lid) & (valid_coarse_labels == lid)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        per_class_coarse_f1[label] = round(
            2 * precision * recall / (precision + recall + 1e-8), 4
        )
    result.update({
        "coarse_accuracy": round(coarse_accuracy, 4),
        "coarse_macro_f1": round(
            float(np.mean(list(per_class_coarse_f1.values()))), 4
        ),
        "per_class_coarse_f1": per_class_coarse_f1,
        "hierarchy_consistency": round(
            (
                all_coarse_preds[coarse_mask]
                == fine_labels_to_coarse(all_preds)[coarse_mask]
            ).float().mean().item()
            if coarse_mask.any()
            else 1.0,
            4,
        ),
    })

    if downbeats is not None and gt_segments is not None:
        segment_preds = all_preds.reshape(-1).clone()
        segment_preds[all_labels.reshape(-1) < 0] = -1
        pred_segs = bars_to_segments(segment_preds, downbeats)
        seg_metrics = segment_level_metrics(pred_segs, gt_segments)
        result.update(seg_metrics)
        coarse_seg_metrics = compute_segment_metrics(
            segments_to_coarse(pred_segs),
            segments_to_coarse(gt_segments),
            COARSE_LABELS,
        )
        result.update({
            f"coarse_{key}": value
            for key, value in coarse_seg_metrics.items()
        })

    return result


def compute_coarse_metrics(
    all_preds: torch.Tensor,
    fine_labels: torch.Tensor,
    downbeats: Optional[List[float]] = None,
    gt_segments: Optional[List[Tuple[float, float, str]]] = None,
) -> Dict:
    """Evaluate the seven-family structure task as the primary target."""
    all_labels = fine_labels_to_coarse(fine_labels)
    mask = all_labels >= 0
    preds = all_preds[mask]
    labels = all_labels[mask]
    if len(labels) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}}

    accuracy = (preds == labels).float().mean().item()
    per_class_f1 = {}
    for label_id, label in enumerate(COARSE_LABELS):
        tp = ((preds == label_id) & (labels == label_id)).sum().item()
        fp = ((preds == label_id) & (labels != label_id)).sum().item()
        fn = ((preds != label_id) & (labels == label_id)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        per_class_f1[label] = round(
            2 * precision * recall / (precision + recall + 1e-8), 4
        )

    result = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(list(per_class_f1.values()))), 4),
        "per_class_f1": per_class_f1,
    }
    if downbeats is not None and gt_segments is not None:
        segment_preds = all_preds.reshape(-1).clone()
        segment_preds[all_labels.reshape(-1) < 0] = -1
        pred_segments = coarse_bars_to_segments(segment_preds, downbeats)
        result.update(compute_segment_metrics(
            pred_segments,
            segments_to_coarse(gt_segments),
            COARSE_LABELS,
        ))
    return result


@torch.no_grad()
def evaluate_song_collection(
    model: nn.Module,
    features: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    song_ids: List[str],
    device: str,
    coarse_weight: float = 0.5,
    downbeats: Optional[Dict[str, List[float]]] = None,
    gt_segments: Optional[
        Dict[str, List[Tuple[float, float, str]]]
    ] = None,
) -> Dict:
    """Evaluate complete songs and aggregate every labeled bar exactly once."""
    all_preds = []
    all_coarse_preds = []
    all_labels = []
    per_song = []
    for song_id in song_ids:
        predictions = predict_hierarchical_sequence(
            model,
            features[song_id],
            device,
            coarse_weight=coarse_weight,
        )
        all_preds.append(predictions["fine"])
        all_coarse_preds.append(predictions["coarse_direct"])
        all_labels.append(labels[song_id])
        if downbeats is not None and gt_segments is not None:
            song_metrics = compute_metrics(
                predictions["fine"],
                labels[song_id],
                all_coarse_preds=predictions["coarse_direct"],
                downbeats=downbeats[song_id],
                gt_segments=gt_segments[song_id],
            )
            song_metrics["song_id"] = song_id
            per_song.append(song_metrics)
    result = compute_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        all_coarse_preds=torch.cat(all_coarse_preds),
    )
    if per_song:
        _add_song_metric_means(result, per_song)
    return result


@torch.no_grad()
def evaluate_coarse_song_collection(
    model: nn.Module,
    features: Dict[str, torch.Tensor],
    fine_labels: Dict[str, torch.Tensor],
    song_ids: List[str],
    device: str,
    downbeats: Optional[Dict[str, List[float]]] = None,
    gt_segments: Optional[
        Dict[str, List[Tuple[float, float, str]]]
    ] = None,
    postprocess_config=None,
) -> Dict:
    """Evaluate the direct coarse-family model on complete songs."""
    all_preds = []
    all_labels = []
    per_song = []
    for song_id in song_ids:
        predictions = predict_coarse_sequence(
            model,
            features[song_id],
            device,
            postprocess_config=postprocess_config,
        )
        all_preds.append(predictions)
        all_labels.append(fine_labels[song_id])
        if downbeats is not None and gt_segments is not None:
            song_metrics = compute_coarse_metrics(
                predictions,
                fine_labels[song_id],
                downbeats=downbeats[song_id],
                gt_segments=gt_segments[song_id],
            )
            song_metrics["song_id"] = song_id
            per_song.append(song_metrics)
    result = compute_coarse_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
    )
    if per_song:
        _add_song_metric_means(result, per_song)
    return result


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    coarse_loss_weight: float = 0.5,
    target_level: str = "hierarchical",
):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        coarse_logits = outputs["coarse"].reshape(-1, NUM_COARSE_CLASSES)
        y_flat = y.reshape(-1)
        mask = y_flat >= 0
        if mask.sum() == 0:
            continue
        coarse_targets = fine_labels_to_coarse(y_flat)
        coarse_loss = F.cross_entropy(
            coarse_logits[mask], coarse_targets[mask]
        )
        if target_level == "coarse":
            loss = coarse_loss
        else:
            fine_logits = outputs["fine"].reshape(-1, NUM_CLASSES)
            fine_loss = F.cross_entropy(fine_logits[mask], y_flat[mask])
            loss = fine_loss + coarse_loss_weight * coarse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YesTiger frozen-backbone bar-level segmenter")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--backbone",
        choices=["mert", "muq"],
        default="mert",
        help="Frozen audio representation backbone.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace/model cache root for downloaded backbones.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="Bar chunks (bars, not frames)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--pool-mode", type=str, default="mean",
                        choices=["mean", "meanmax", "meanmaxstd"])
    parser.add_argument(
        "--bar-context-features",
        choices=BAR_CONTEXT_FEATURE_CHOICES,
        default="none",
        help=(
            "Optional low-dimensional per-bar timing features concatenated "
            "after backbone pooling. rhythm adds duration/beat-density cues; "
            "rhythm_pos also adds coarse song-position cues."
        ),
    )
    parser.add_argument("--mert-layers", type=int, nargs="*", default=None,
                        help="MERT transformer layers to concat, e.g. --mert-layers 4 8 12 (default: last layer only)")
    parser.add_argument("--muq-layers", type=int, nargs="*", default=None,
                        help="MuQ conformer layers to concat, e.g. --muq-layers 4 8 12 (default: last layer only)")
    parser.add_argument("--muq-chunk-seconds", type=float, default=30.0,
                        help="Seconds per MuQ forward chunk; <=0 uses full-song forward.")
    parser.add_argument("--muq-overlap-seconds", type=float, default=1.0,
                        help="MuQ chunk overlap trimmed before concatenation.")
    parser.add_argument(
        "--target-level",
        choices=["coarse", "hierarchical"],
        default="coarse",
        help=(
            "coarse: train only the 7 structure families (default); "
            "hierarchical: reproduce the 7+10 dual-head experiment"
        ),
    )
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5,
                        help="Coarse auxiliary-loss weight in hierarchical mode")
    parser.add_argument("--coarse-inference-weight", type=float, default=0.5,
                        help="Parent-family score weight during fine prediction")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Experiment archive root (default: DATA_DIR/runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional stable name; a numeric suffix is added if it exists",
    )
    parser.set_defaults(write_legacy_artifacts=True)
    parser.add_argument(
        "--write-legacy-artifacts",
        dest="write_legacy_artifacts",
        action="store_true",
        help=(
            "Also write DATA_DIR best_model/results/latest/predictions "
            "artifacts for compatibility (default)."
        ),
    )
    parser.add_argument(
        "--no-write-legacy-artifacts",
        dest="write_legacy_artifacts",
        action="store_false",
        help="Keep experiment outputs inside the run directory only.",
    )
    parser.add_argument(
        "--selection-bar-weight",
        type=float,
        default=0.3,
        help="Validation Bar Macro F1 weight for checkpoint selection",
    )
    parser.add_argument(
        "--selection-segment-weight",
        type=float,
        default=0.5,
        help="Validation Segment Macro F1 weight for checkpoint selection",
    )
    parser.add_argument(
        "--selection-boundary-weight",
        type=float,
        default=0.2,
        help="Validation Boundary F1@3s weight for checkpoint selection",
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "smooth", "merge", "full"],
        default="none",
        help=(
            "Inference-time cleanup for coarse models. none preserves raw "
            "argmax; smooth averages neighbouring bar probabilities; merge "
            "only merges short interior segments; full adds transition grammar."
        ),
    )
    parser.add_argument(
        "--postprocess-smooth-window",
        type=int,
        default=3,
        help="Odd/even bar window for probability smoothing; even values are rounded up.",
    )
    parser.add_argument(
        "--postprocess-transition-penalty",
        type=float,
        default=0.12,
        help="Penalty used by transition grammar in --postprocess full.",
    )
    parser.add_argument(
        "--postprocess-min-bars",
        type=str,
        default=None,
        help=(
            "Optional min-duration overrides, e.g. "
            "verse=4,chorus=4,pre_chorus=2"
        ),
    )
    parser.set_defaults(extend_tail_downbeats=True)
    parser.add_argument(
        "--extend-tail-downbeats",
        dest="extend_tail_downbeats",
        action="store_true",
        help="Extend final downbeat grid to annotation end (default).",
    )
    parser.add_argument(
        "--no-extend-tail-downbeats",
        dest="extend_tail_downbeats",
        action="store_false",
        help="Disable tail downbeat extension for exact raw-struct reproduction.",
    )
    parser.add_argument(
        "--tail-extension-lookback",
        type=int,
        default=8,
        help="Number of recent bar intervals used to extrapolate tail downbeats.",
    )
    parser.add_argument(
        "--tail-extension-tolerance",
        type=float,
        default=0.5,
        help="Do not extend when annotation end is within this many seconds.",
    )
    parser.add_argument(
        "--tail-extension-max-bars",
        type=int,
        default=64,
        help="Safety cap for newly added tail downbeats.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (
        args.selection_bar_weight
        + args.selection_segment_weight
        + args.selection_boundary_weight
        <= 0
    ):
        parser.error("checkpoint selection weights must sum to a positive value")
    if args.tail_extension_lookback < 1:
        parser.error("--tail-extension-lookback must be >= 1")
    if args.tail_extension_tolerance < 0:
        parser.error("--tail-extension-tolerance must be >= 0")
    if args.tail_extension_max_bars < 1:
        parser.error("--tail-extension-max-bars must be >= 1")
    if args.muq_chunk_seconds < 0:
        parser.error("--muq-chunk-seconds must be >= 0")
    if args.muq_overlap_seconds < 0:
        parser.error("--muq-overlap-seconds must be >= 0")

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    try:
        postprocess_config = make_postprocess_config(
            mode=args.postprocess,
            smoothing_window=args.postprocess_smooth_window,
            transition_penalty=args.postprocess_transition_penalty,
            min_bars_spec=args.postprocess_min_bars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Device: {device}")
    print(f"Target level: {args.target_level}")
    print(f"Postprocess: {postprocess_config.summary()}")
    print(
        "Tail downbeat extension: "
        f"{'on' if args.extend_tail_downbeats else 'off'} "
        f"(target=annotation_end)"
    )

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / "cache").resolve()
    model_cache_dir = (args.model_cache_dir or data_dir.parent / ".hf").resolve()
    feature_layers = feature_layers_for_args(args)
    artifact_suffix = backbone_artifact_suffix(args.backbone)
    songs_dir = data_dir / "songs"
    ann_dir = data_dir / "annotations"
    struct_dir = data_dir / "struct"
    runs_dir = args.runs_dir or data_dir / "runs"
    if not runs_dir.is_absolute():
        runs_dir = data_dir / runs_dir
    run_dir = create_run_directory(
        runs_dir.resolve(),
        backbone=args.backbone,
        target_level=args.target_level,
        pool_mode=args.pool_mode,
        feature_layers=feature_layers,
        seed=args.seed,
        run_name=args.run_name,
    )
    run_id = run_dir.name
    print(f"Run directory: {run_dir}")
    _write_json(run_dir / "config.json", {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "status": "initializing",
    })

    # ---- Collect songs ----
    song_ids = []
    for d in sorted(ann_dir.iterdir()):
        if d.is_dir() and (songs_dir / f"{d.name}.mp3").exists():
            if list(d.glob("*.annotation.json")):
                song_ids.append(d.name)
    print(f"Songs: {len(song_ids)}")

    # ---- Step 1: Extract frozen-backbone embeddings ----
    print(f"\n=== Step 1: {args.backbone.upper()} embedding extraction ===")
    print(f"Backbone: {args.backbone}")
    print(f"Feature layers: {feature_layers or 'last hidden state only'}")
    if args.backbone == "muq":
        print(f"MuQ model: {MUQ_MODEL_NAME}")
        print(f"Model cache: {model_cache_dir}")
        print(f"MuQ chunking: {args.muq_chunk_seconds}s chunks, {args.muq_overlap_seconds}s overlap")
    extractor = create_feature_extractor(
        args.backbone,
        device=device,
        feature_layers=feature_layers,
        model_cache_dir=model_cache_dir,
        muq_chunk_seconds=args.muq_chunk_seconds,
        muq_overlap_seconds=args.muq_overlap_seconds,
    )
    frame_features: Dict[str, torch.Tensor] = {}

    for sid in tqdm(song_ids, desc=f"Extracting {args.backbone.upper()}", unit="song", colour="cyan"):
        audio_path = songs_dir / f"{sid}.mp3"
        emb = extractor.extract_all(audio_path, cache_dir)
        frame_features[sid] = emb

    # ---- Step 2: Bar pooling + label mapping ----
    print("\n=== Step 2: Bar pooling & label mapping ===")
    bar_features: Dict[str, torch.Tensor] = {}
    bar_labels: Dict[str, torch.Tensor] = {}
    all_downbeats: Dict[str, List[float]] = {}
    all_gt_segments: Dict[str, List[Tuple[float, float, str]]] = {}
    tail_extension_stats: Dict[str, Dict] = {}

    for sid in tqdm(song_ids, desc="Bar pooling", unit="song", colour="cyan"):
        audio_path = songs_dir / f"{sid}.mp3"
        ann_path = ann_dir / sid / f"{sid}.annotation.json"
        struct_path = struct_dir / f"{sid}.json"

        waveform = load_audio(audio_path)
        audio_dur = len(waveform) / SAMPLE_RATE
        struct = load_struct(struct_path)
        gt_segs = load_annotation(ann_path)

        downbeats = struct["downbeats"]
        if args.extend_tail_downbeats:
            downbeats, tail_stats = extend_tail_downbeats(
                downbeats,
                target_end_s=annotation_end_time(gt_segs),
                lookback_bars=args.tail_extension_lookback,
                tolerance_s=args.tail_extension_tolerance,
                max_new_downbeats=args.tail_extension_max_bars,
            )
        else:
            tail_stats = {
                "enabled": False,
                "added_downbeats": 0,
                "original_end": downbeats[-1] if downbeats else 0.0,
                "target_end": annotation_end_time(gt_segs),
                "bar_duration": 0.0,
            }
        tail_extension_stats[sid] = tail_stats
        if len(downbeats) < 2:
            continue

        # Bar pooling
        bar_emb = bar_pooling(frame_features[sid], struct["beats"], downbeats,
                              audio_dur, pool_mode=args.pool_mode)
        bar_emb = append_bar_context_features(
            bar_emb,
            struct.get("beats", []),
            downbeats,
            audio_dur,
            mode=args.bar_context_features,
        )
        bar_lab = segments_to_bar_labels(gt_segs, downbeats)
        if len(bar_emb) != len(bar_lab):
            raise ValueError(
                f"{sid}: pooled bars ({len(bar_emb)}) != labels ({len(bar_lab)})"
            )

        bar_features[sid] = bar_emb
        bar_labels[sid] = bar_lab
        all_downbeats[sid] = downbeats
        all_gt_segments[sid] = gt_segs

    print(f"Bar-level songs: {len(bar_features)}")
    tail_summary = _tail_extension_summary(tail_extension_stats)
    if args.extend_tail_downbeats:
        print(
            "Tail downbeat extension: "
            f"{tail_summary['songs_extended']} songs, "
            f"+{tail_summary['total_added_downbeats']} downbeats"
        )

    # ---- Step 3: Train/Val/Test split ----
    available = set(bar_features.keys())
    train_ids = [s for s in TRAIN_IDS if s in available]
    val_ids = [s for s in VAL_IDS if s in available]
    test_ids = [s for s in TEST_IDS if s in available]
    print(f"\n=== Step 3: Train {len(train_ids)} / Val {len(val_ids)} / Test {len(test_ids)} ===")
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "The fixed split is incomplete. Check songs/, annotations/, and struct/."
        )

    input_dim = bar_features[train_ids[0]].shape[1]
    print(
        f"Input dim: {input_dim} "
        f"(pool_mode={args.pool_mode}, "
        f"bar_context_features={args.bar_context_features}, "
        f"context_dim={bar_context_feature_dim(args.bar_context_features)})"
    )
    print(f"Chunk size: {args.chunk_size} bars")
    # Feature extraction may or may not instantiate a large backbone depending
    # on cache hits. Reset here so head initialization and dropout are not
    # affected by cache state.
    set_seed(args.seed)

    # Build one dataset per song so no recurrent chunk crosses song boundaries.
    train_ds = ConcatDataset([
        BarDataset(
            bar_features[s],
            bar_labels[s],
            chunk_size=args.chunk_size,
            stride=max(1, args.chunk_size // 2),
        )
        for s in train_ids
    ])
    generator = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    # Build model
    model = StructureBiLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers, dropout=args.dropout,
                            target_level=args.target_level).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    checkpoint_config = {
        "task": "bar",
        "run_id": run_id,
        "target_level": args.target_level,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "backbone": args.backbone,
        "backbone_model": MUQ_MODEL_NAME if args.backbone == "muq" else MERT_MODEL_NAME,
        "feature_layers": feature_layers,
        "muq_chunk_seconds": args.muq_chunk_seconds,
        "muq_overlap_seconds": args.muq_overlap_seconds,
        "pool_mode": args.pool_mode,
        "bar_context_features": args.bar_context_features,
        "bar_context_feature_dim": bar_context_feature_dim(args.bar_context_features),
        "mert_layers": args.mert_layers,
        "muq_layers": args.muq_layers,
        "chunk_size": args.chunk_size,
        "coarse_loss_weight": args.coarse_loss_weight,
        "coarse_inference_weight": args.coarse_inference_weight,
        "postprocess": postprocess_config.as_dict(),
        "tail_downbeat_extension": _tail_extension_config(args),
        "seed_reset_after_feature_extraction": True,
        "seed": args.seed,
        "fine_labels": LABELS,
        "coarse_labels": COARSE_LABELS,
        "fine_to_coarse": {
            label: coarse_label_for(label) for label in LABELS
        },
        "selection_weights": {
            "bar_macro_f1": args.selection_bar_weight,
            "segment_macro_f1": args.selection_segment_weight,
            "boundary_f1_3s": args.selection_boundary_weight,
        },
    }
    run_config = {
        **checkpoint_config,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "split": {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
        },
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "tail_downbeat_extension_summary": tail_summary,
        "status": "training",
    }
    _write_json(run_dir / "config.json", run_config)

    # Train. Checkpoint selection uses validation structure quality, not test data.
    checkpoint_name = (
        "best_model_bar_coarse.pt"
        if args.target_level == "coarse"
        else "best_model_bar.pt"
    )
    run_checkpoint_path = run_dir / "checkpoint.pt"
    legacy_checkpoint_path = data_dir / (
        checkpoint_name.replace(".pt", f"{artifact_suffix}.pt")
    )
    best_selection_score = -1.0
    best_val_metrics = {}
    best_epoch = 0
    patience_counter = 0
    training_history = []
    stopped_early = False
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep", colour="green")
    for epoch in pbar:
        train_loss = train_one_epoch(
            model,
            train_dl,
            optimizer,
            device,
            coarse_loss_weight=args.coarse_loss_weight,
            target_level=args.target_level,
        )
        if args.target_level == "coarse":
            val_metrics = evaluate_coarse_song_collection(
                model,
                bar_features,
                bar_labels,
                val_ids,
                device,
                downbeats=all_downbeats,
                gt_segments=all_gt_segments,
                postprocess_config=postprocess_config,
            )
            selection_bar_f1 = val_metrics["macro_f1"]
            selection_accuracy = val_metrics["accuracy"]
            selection_segment_f1 = val_metrics["macro_seg_f1_mean"]
            selection_boundary_f1 = val_metrics["boundary_f1_3s_mean"]
        else:
            val_metrics = evaluate_song_collection(
                model,
                bar_features,
                bar_labels,
                val_ids,
                device,
                coarse_weight=args.coarse_inference_weight,
                downbeats=all_downbeats,
                gt_segments=all_gt_segments,
            )
            selection_bar_f1 = val_metrics["coarse_macro_f1"]
            selection_accuracy = val_metrics["coarse_accuracy"]
            selection_segment_f1 = val_metrics["coarse_macro_seg_f1_mean"]
            selection_boundary_f1 = val_metrics["coarse_boundary_f1_3s_mean"]

        selection_score = checkpoint_selection_score(
            val_metrics,
            target_level=args.target_level,
            bar_weight=args.selection_bar_weight,
            segment_weight=args.selection_segment_weight,
            boundary_weight=args.selection_boundary_weight,
        )
        pbar.set_postfix(
            loss=f"{train_loss:.3f}",
            val_acc=f"{selection_accuracy:.3f}",
            val_f1=f"{selection_bar_f1:.3f}",
            val_seg=f"{selection_segment_f1:.3f}",
            val_b3=f"{selection_boundary_f1:.3f}",
            score=f"{selection_score:.3f}",
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "selection_score": round(selection_score, 6),
            "validation": val_metrics,
        })
        _write_json(run_dir / "training_history.json", training_history)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0
            best_checkpoint_config = {
                **checkpoint_config,
                "best_epoch": best_epoch,
                "best_validation_score": round(best_selection_score, 6),
                "best_validation_metrics": best_val_metrics,
            }
            save_checkpoint(
                run_checkpoint_path, model, best_checkpoint_config
            )
            # Root-level artifacts remain convenient aliases for the latest run.
            if args.write_legacy_artifacts:
                save_checkpoint(
                    legacy_checkpoint_path, model, best_checkpoint_config
                )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                pbar.set_description(f"Training [stop@e{best_epoch}]")
                stopped_early = True
                break

    state_dict, _ = load_checkpoint(run_checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    best_bar_f1 = (
        best_val_metrics["macro_f1"]
        if args.target_level == "coarse"
        else best_val_metrics["coarse_macro_f1"]
    )
    best_accuracy = (
        best_val_metrics["accuracy"]
        if args.target_level == "coarse"
        else best_val_metrics["coarse_accuracy"]
    )
    best_segment_f1 = (
        best_val_metrics["macro_seg_f1_mean"]
        if args.target_level == "coarse"
        else best_val_metrics["coarse_macro_seg_f1_mean"]
    )
    best_boundary_f1 = (
        best_val_metrics["boundary_f1_3s_mean"]
        if args.target_level == "coarse"
        else best_val_metrics["coarse_boundary_f1_3s_mean"]
    )
    print(
        f"\nBest epoch: {best_epoch}, "
        f"validation selection score: {best_selection_score:.4f}"
    )
    print(
        "Best validation: "
        f"bar_acc={best_accuracy:.4f}  "
        f"bar_f1={best_bar_f1:.4f}  "
        f"segment_f1={best_segment_f1:.4f}  "
        f"boundary@3s={best_boundary_f1:.4f}"
    )

    # ---- Test evaluation ----
    print(f"\n=== Test ({len(test_ids)} songs) ===")
    all_test_metrics = []
    run_preds_dir = run_dir / "predictions"
    run_preds_dir.mkdir(exist_ok=True)
    legacy_preds_dir = data_dir / (
        f"predictions_bar_coarse{artifact_suffix}"
        if args.target_level == "coarse"
        else f"predictions_bar{artifact_suffix}"
    )
    if args.write_legacy_artifacts:
        legacy_preds_dir.mkdir(exist_ok=True)

    for test_sid in tqdm(test_ids, desc="Evaluating", unit="song", colour="yellow"):
        gt_segs = all_gt_segments[test_sid]
        downbeats = all_downbeats[test_sid]
        if args.target_level == "coarse":
            full_preds = predict_coarse_sequence(
                model,
                bar_features[test_sid],
                device,
                postprocess_config=postprocess_config,
            )
            metrics = compute_coarse_metrics(
                full_preds,
                bar_labels[test_sid],
                downbeats=downbeats,
                gt_segments=gt_segs,
            )
        else:
            predictions = predict_hierarchical_sequence(
                model,
                bar_features[test_sid],
                device,
                coarse_weight=args.coarse_inference_weight,
            )
            full_preds = predictions["fine"]
            metrics = compute_metrics(
                full_preds,
                bar_labels[test_sid],
                all_coarse_preds=predictions["coarse_direct"],
                downbeats=downbeats,
                gt_segments=gt_segs,
            )
        metrics["test_song"] = test_sid
        all_test_metrics.append(metrics)

        # Export predictions
        export_preds = full_preds.clone()
        export_preds[bar_labels[test_sid] < 0] = -1
        if args.target_level == "coarse":
            coarse_gt_segs = segments_to_coarse(gt_segs)
            coarse_pred_segs = coarse_bars_to_segments(
                export_preds, downbeats
            )
            export_payload = {
                "song_id": test_sid,
                "target_level": "coarse",
                "ground_truth_fine": [
                    {
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "label": label,
                        "coarse_label": coarse_label_for(label),
                    }
                    for s, e, label in gt_segs
                ],
                "ground_truth": [
                    {"start": round(s, 2), "end": round(e, 2), "label": label}
                    for s, e, label in coarse_gt_segs
                ],
                "predicted": [
                    {"start": round(s, 2), "end": round(e, 2), "label": label}
                    for s, e, label in coarse_pred_segs
                ],
            }
        else:
            pred_segs = bars_to_segments(export_preds, downbeats)
            coarse_gt_segs = segments_to_coarse(gt_segs)
            coarse_pred_segs = segments_to_coarse(pred_segs)
            export_payload = {
                "song_id": test_sid,
                "target_level": "hierarchical",
                "ground_truth": [
                    {
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "label": lb,
                        "coarse_label": coarse_label_for(lb),
                    }
                    for s, e, lb in gt_segs
                ],
                "predicted": [
                    {
                        "start": round(s, 2),
                        "end": round(e, 2),
                        "label": lb,
                        "coarse_label": coarse_label_for(lb),
                    }
                    for s, e, lb in pred_segs
                ],
                "ground_truth_coarse": [
                    {"start": round(s, 2), "end": round(e, 2), "label": lb}
                    for s, e, lb in coarse_gt_segs
                ],
                "predicted_coarse": [
                    {"start": round(s, 2), "end": round(e, 2), "label": lb}
                    for s, e, lb in coarse_pred_segs
                ],
            }
        prediction_name = f"{test_sid}.prediction.json"
        _write_json(run_preds_dir / prediction_name, export_payload)
        if args.write_legacy_artifacts:
            _write_json(legacy_preds_dir / prediction_name, export_payload)

    # ---- Summary ----
    print("\n=== Test Summary ===")
    accs = [m["accuracy"] for m in all_test_metrics]
    f1s = [m["macro_f1"] for m in all_test_metrics]
    print(f"Bar Accuracy:     mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"Bar Macro F1:     mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    if args.target_level == "hierarchical":
        coarse_accs = [m["coarse_accuracy"] for m in all_test_metrics]
        coarse_f1s = [m["coarse_macro_f1"] for m in all_test_metrics]
        print(f"Coarse Accuracy:  mean={np.mean(coarse_accs):.4f}  std={np.std(coarse_accs):.4f}")
        print(f"Coarse Macro F1:  mean={np.mean(coarse_f1s):.4f}  std={np.std(coarse_f1s):.4f}")

    for test_sid, m in zip(test_ids, all_test_metrics):
        seg_str = (
            f"  seg_f1={m.get('macro_seg_f1', 0):.4f}"
            f"  bnd@0.5={m.get('boundary_f1_0_5s', 0):.4f}"
            f"  bnd@3={m.get('boundary_f1_3s', 0):.4f}"
        )
        print(f"  {test_sid}: acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}{seg_str}")

    if all_test_metrics and 'macro_seg_f1' in all_test_metrics[0]:
        seg_f1s = [m['macro_seg_f1'] for m in all_test_metrics]
        bnd_f1s = [m['boundary_f1_0_5s'] for m in all_test_metrics]
        bnd_f1s_3s = [m['boundary_f1_3s'] for m in all_test_metrics]
        print(f"\nSegment Macro F1: mean={np.mean(seg_f1s):.4f}")
        print(f"Boundary F1 @0.5s: mean={np.mean(bnd_f1s):.4f}")
        print(f"Boundary F1 @3.0s: mean={np.mean(bnd_f1s_3s):.4f}")

    print(f"\nPer-class {'Coarse ' if args.target_level == 'coarse' else ''}Bar F1:")
    class_f1s = defaultdict(list)
    for m in all_test_metrics:
        for lb, f1 in m["per_class_f1"].items():
            class_f1s[lb].append(f1)
    report_labels = COARSE_LABELS if args.target_level == "coarse" else LABELS
    for lb in report_labels:
        if lb in class_f1s:
            print(f"  {lb:22s}  mean={np.mean(class_f1s[lb]):.4f}")

    results = {
        "run_id": run_id,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "target_level": args.target_level,
        "backbone": args.backbone,
        "backbone_model": MUQ_MODEL_NAME if args.backbone == "muq" else MERT_MODEL_NAME,
        "feature_layers": feature_layers,
        "muq_chunk_seconds": args.muq_chunk_seconds,
        "muq_overlap_seconds": args.muq_overlap_seconds,
        "pool_mode": args.pool_mode, "input_dim": input_dim,
        "bar_context_features": args.bar_context_features,
        "bar_context_feature_dim": bar_context_feature_dim(args.bar_context_features),
        "mert_layers": args.mert_layers,
        "muq_layers": args.muq_layers,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(training_history),
        "stopped_early": stopped_early,
        "coarse_loss_weight": args.coarse_loss_weight,
        "coarse_inference_weight": args.coarse_inference_weight,
        "postprocess": postprocess_config.as_dict(),
        "tail_downbeat_extension": _tail_extension_config(args),
        "tail_downbeat_extension_summary": tail_summary,
        "fine_labels": LABELS,
        "coarse_labels": COARSE_LABELS,
        "selection_weights": checkpoint_config["selection_weights"],
        "best_validation_score": round(best_selection_score, 4),
        "best_validation_metrics": best_val_metrics,
        "best_val_accuracy": round(best_accuracy, 4),
        "best_val_macro_f1": round(best_bar_f1, 4),
        "best_val_segment_macro_f1": round(best_segment_f1, 4),
        "best_val_boundary_f1_3s": round(best_boundary_f1, 4),
        "best_epoch": best_epoch,
        "test_bar_acc_mean": round(np.mean(accs), 4),
        "test_bar_acc_std": round(np.std(accs), 4),
        "test_bar_macro_f1_mean": round(np.mean(f1s), 4),
        "test_bar_macro_f1_std": round(np.std(f1s), 4),
        "per_class_bar_f1": {
            lb: round(np.mean(class_f1s.get(lb, [0])), 4)
            for lb in report_labels
        },
        "per_song": all_test_metrics,
    }
    if args.target_level == "hierarchical":
        results.update({
            "test_coarse_acc_mean": round(np.mean(coarse_accs), 4),
            "test_coarse_acc_std": round(np.std(coarse_accs), 4),
            "test_coarse_macro_f1_mean": round(np.mean(coarse_f1s), 4),
            "test_coarse_macro_f1_std": round(np.std(coarse_f1s), 4),
        })
    if all_test_metrics and 'macro_seg_f1' in all_test_metrics[0]:
        results["test_seg_macro_f1_mean"] = round(np.mean(seg_f1s), 4)
        results["test_boundary_f1_0_5s_mean"] = round(np.mean(bnd_f1s), 4)
        results["test_boundary_f1_3s_mean"] = round(np.mean(bnd_f1s_3s), 4)
    results_name = (
        "results_bar_coarse.json"
        if args.target_level == "coarse"
        else "results_bar.json"
    )
    if artifact_suffix:
        results_name = results_name.replace(".json", f"{artifact_suffix}.json")
    _write_json(run_dir / "metrics.json", results)
    if args.write_legacy_artifacts:
        _write_json(data_dir / results_name, results)
    completed_at = datetime.now().astimezone().isoformat()
    _write_json(run_dir / "config.json", {
        **run_config,
        "status": "completed",
        "completed_at": completed_at,
        "best_epoch": best_epoch,
        "best_validation_score": round(best_selection_score, 6),
    })
    latest_run_name = "latest_run.json" if not artifact_suffix else f"latest_run{artifact_suffix}.json"
    if args.write_legacy_artifacts:
        _write_json(data_dir / latest_run_name, {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "checkpoint": str(run_checkpoint_path),
            "metrics": str(run_dir / "metrics.json"),
            "target_level": args.target_level,
            "backbone": args.backbone,
            "completed_at": completed_at,
        })
    print(f"\nRun saved to {run_dir}")
    print(f"Results saved to {run_dir / 'metrics.json'}")
    print(f"Predictions saved to {run_preds_dir}")


if __name__ == "__main__":
    main()
