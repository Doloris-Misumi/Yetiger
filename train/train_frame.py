"""
YesTiger MERT Structure Segmentation — Training Script
========================================================
Pipeline:
  MP3 → MERT-v1-95M (frozen) → frame embeddings (cache)
  Annotations → time→frame label mapping
  BiLSTM head → fixed song-level Train/Val/Test split → metrics

Usage:
  python train_frame.py --data-dir . --epochs 25 --batch-size 16
"""

import argparse
import json
import math
from collections import defaultdict
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
    chunk_starts,
    coarse_label_for,
    fine_labels_to_coarse,
    load_checkpoint,
    predict_hierarchical_sequence,
    save_checkpoint,
    segments_to_coarse,
    set_seed,
)
from segmentation_metrics import segment_level_metrics as compute_segment_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000  # MERT native sample rate
# MERT (wav2vec2) CNN downsamples by factor 320 → ~75 Hz frame rate
# Actual frame_hop is computed per-song from waveform_length / num_mert_frames
LABELS = MODEL_LABELS
LABEL2ID = {lb: i for i, lb in enumerate(LABELS)}
ID2LABEL = {i: lb for i, lb in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
NUM_COARSE_CLASSES = len(COARSE_LABELS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotation(
    ann_path: Path,
    include_annotation_markers: bool = False,
) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, music_label)."""
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    segs = ann.get("segments", [])
    return [
        (s["start"], s["end"], s["music_label"])
        for s in segs
        if include_annotation_markers
        or s["music_label"] not in ANNOTATION_ONLY_LABELS
    ]


def segments_to_frame_labels(
    segments: List[Tuple[float, float, str]],
    num_frames: int,
    audio_duration_s: float,
) -> torch.Tensor:
    """
    Convert (start, end, label) → per-frame label tensor.
    Computes frame_hop from actual audio duration / num_mert_frames.
    """
    frame_hop = audio_duration_s / max(num_frames, 1)
    labels = torch.full((num_frames,), -1, dtype=torch.long)
    for start, end, label in segments:
        if label not in LABEL2ID:
            continue
        lid = LABEL2ID[label]
        f0 = max(0, int(start / frame_hop))
        f1 = min(num_frames, int(math.ceil(end / frame_hop)))
        labels[f0:f1] = lid
    return labels


def _load_with_ffmpeg(audio_path: Path, target_sr: int) -> torch.Tensor:
    """Fallback: decode MP3 via ffmpeg → WAV pipe → torchaudio."""
    import subprocess, tempfile, os

    # Find ffmpeg - check common locations
    ffmpeg = None
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.venv', 'Scripts', 'ffmpeg.exe'),
        'ffmpeg',
        'ffmpeg.exe',
    ]:
        try:
            subprocess.run([candidate, '-version'], capture_output=True, timeout=5)
            ffmpeg = candidate
            break
        except Exception:
            continue

    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found - install it to decode MP3 files")

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [ffmpeg, '-y', '-i', str(audio_path), '-ar', str(target_sr),
             '-ac', '1', '-f', 'wav', tmp_path],
            capture_output=True, timeout=60, check=True
        )
        import torchaudio
        waveform, sr = torchaudio.load(tmp_path)
        return waveform.squeeze(0)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def load_audio(audio_path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio, resample to target_sr, return mono waveform."""
    import torchaudio
    import torchaudio.functional as AF

    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        # Fallback to ffmpeg for files with codec issues
        waveform, sr = _load_with_ffmpeg(audio_path, target_sr).unsqueeze(0), target_sr

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # (samples,)


# ---------------------------------------------------------------------------
# MERT feature extraction
# ---------------------------------------------------------------------------

class MERTFeatureExtractor:
    """Extract frame-level embeddings from frozen MERT."""

    def __init__(self, device: str = "cuda"):
        import os
        # Workaround for transformers strict torch version check
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        from transformers import Wav2Vec2Model, Wav2Vec2Config

        self.device = device
        try:
            self.model = Wav2Vec2Model.from_pretrained(
                MERT_MODEL_NAME, use_safetensors=True
            ).to(device)
        except Exception:
            # Fallback: allow torch.load with weights_only=False for older formats
            import torch as _torch
            _torch_orig = _torch.load
            _torch.load = lambda *a, **kw: _torch_orig(*a, **{**kw, 'weights_only': False})
            self.model = Wav2Vec2Model.from_pretrained(MERT_MODEL_NAME).to(device)
            _torch.load = _torch_orig

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (samples,) at 24kHz
        returns: (num_frames, hidden_dim)  [768 for MERT-v1-95M]
        """
        wf = waveform.unsqueeze(0).to(self.device)  # (1, samples)
        outputs = self.model(wf, output_hidden_states=True)
        # Use last hidden state: (1, time_steps, 768)
        hidden = outputs.last_hidden_state.squeeze(0).cpu()  # (time_steps, 768)
        return hidden

    def extract_all(self, audio_path: Path, cache_dir: Path) -> torch.Tensor:
        """Extract and cache embeddings for one song."""
        cache_path = cache_dir / f"{audio_path.stem}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        waveform = load_audio(audio_path)
        embeddings = self.extract(waveform)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        return embeddings


# ---------------------------------------------------------------------------
# BiLSTM Head
# ---------------------------------------------------------------------------

class StructureBiLSTM(nn.Module):
    """Shared BiLSTM with fine-label and coarse-family classification heads."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        num_coarse_classes: int = NUM_COARSE_CLASSES,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fine_classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.coarse_classifier = nn.Linear(
            hidden_dim * 2, num_coarse_classes
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, time_steps, input_dim)
        returns fine and coarse logits
        """
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return {
            "fine": self.fine_classifier(out),
            "coarse": self.coarse_classifier(out),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Yields (features, labels) chunks for training."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 512,
        stride: int = 256,
    ):
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
        # Pad if needed
        if x.shape[0] < self.chunk_size:
            pad = self.chunk_size - x.shape[0]
            x = F.pad(x, (0, 0, 0, pad))
            y = F.pad(y, (0, pad), value=-1)
        return x, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def median_filter_labels(
    labels: torch.Tensor,
    window: int = 11,
    ignore_label: int = -1,
) -> torch.Tensor:
    """
    Apply majority-vote filter to frame labels for temporal smoothing.
    Uses box convolution over one-hot encodings → argmax (fast).
    Window size should be odd; typical value 11 ≈ 0.15s at 75Hz.
    """
    if window <= 1:
        return labels.clone()

    # Build one-hot: (N, C)
    arr = labels.numpy()
    N = len(arr)
    valid_mask = arr != ignore_label
    onehot = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    onehot[valid_mask, arr[valid_mask]] = 1.0

    # Box convolution → count of each class in sliding window
    kernel = np.ones(window, dtype=np.float32)
    # correlate1d for each class (fast C implementation)
    from scipy.ndimage import correlate1d
    counts = np.zeros_like(onehot)
    for c in range(NUM_CLASSES):
        counts[:, c] = correlate1d(onehot[:, c], kernel, mode='nearest')

    # Argmax → most frequent class
    smoothed = np.argmax(counts, axis=1).astype(np.int64)

    # Restore frames where all neighbors were -1
    all_invalid = counts.sum(axis=1) == 0
    smoothed[all_invalid] = ignore_label

    return torch.from_numpy(smoothed)


def frames_to_segments(
    frame_labels: torch.Tensor,
    frame_hop_s: float,
    ignore_label: int = -1,
    smooth_window: int = 0,
) -> List[Tuple[float, float, str]]:
    """Merge consecutive same-label frames into (start_s, end_s, label).
    If smooth_window > 0, apply median filter first.
    """
    if smooth_window > 1:
        frame_labels = median_filter_labels(frame_labels, smooth_window, ignore_label)

    segs = []
    labels = frame_labels.tolist()
    if not labels:
        return segs

    prev = labels[0]
    start = 0
    for i, lb in enumerate(labels):
        if lb != prev:
            end = i
            if prev >= 0:
                segs.append((start * frame_hop_s, end * frame_hop_s, ID2LABEL[prev]))
            start = i
            prev = lb
    # Last segment
    if prev >= 0:
        segs.append((start * frame_hop_s, len(labels) * frame_hop_s, ID2LABEL[prev]))
    return segs


def segment_level_metrics(
    pred_segs: List[Tuple[float, float, str]],
    gt_segs: List[Tuple[float, float, str]],
) -> Dict:
    return compute_segment_metrics(pred_segs, gt_segs, LABELS)


def compute_metrics(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    all_coarse_preds: Optional[torch.Tensor] = None,
    frame_hop_s: Optional[float] = None,
    gt_segments: Optional[List[Tuple[float, float, str]]] = None,
    smooth_window: int = 0,
) -> Dict:
    """Compute frame accuracy, per-class F1, and optionally segment-level metrics."""
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
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        per_class_f1[ID2LABEL[lid]] = round(f1, 4)

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

    # Segment-level: convert frame preds to segments
    if frame_hop_s is not None and gt_segments is not None:
        segment_preds = all_preds.reshape(-1).clone()
        segment_preds[all_labels.reshape(-1) < 0] = -1
        pred_segs = frames_to_segments(segment_preds, frame_hop_s,
                                       smooth_window=smooth_window)
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    coarse_loss_weight: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        fine_logits = outputs["fine"].reshape(-1, NUM_CLASSES)
        coarse_logits = outputs["coarse"].reshape(-1, NUM_COARSE_CLASSES)
        y_flat = y.reshape(-1)
        mask = y_flat >= 0
        if mask.sum() == 0:
            continue
        coarse_targets = fine_labels_to_coarse(y_flat)
        fine_loss = F.cross_entropy(fine_logits[mask], y_flat[mask])
        coarse_loss = F.cross_entropy(
            coarse_logits[mask], coarse_targets[mask]
        )
        loss = fine_loss + coarse_loss_weight * coarse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate_song_collection(
    model: nn.Module,
    features: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    song_ids: List[str],
    device: str,
    chunk_size: int,
    coarse_weight: float = 0.5,
) -> Dict:
    """Evaluate every frame once while keeping song sequences separate."""
    all_preds = []
    all_coarse_preds = []
    all_labels = []
    for song_id in song_ids:
        predictions = predict_hierarchical_sequence(
            model,
            features[song_id],
            device,
            chunk_size=chunk_size,
            stride=max(1, chunk_size // 2),
            coarse_weight=coarse_weight,
        )
        all_preds.append(predictions["fine"])
        all_coarse_preds.append(predictions["coarse_direct"])
        all_labels.append(labels[song_id])
    return compute_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        all_coarse_preds=torch.cat(all_coarse_preds),
    )


# ---------------------------------------------------------------------------
# Main fixed-split pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YesTiger MERT structure segmenter")
    parser.add_argument("--data-dir", type=Path, default=Path("."),
                        help="train/ directory containing songs/ and annotations/")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory to cache MERT embeddings (default: data-dir/cache)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--smooth-window", type=int, default=75,
                        help="Median filter window size for segment smoothing (0=off, odd number)")
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5,
                        help="Weight of the coarse-family auxiliary loss")
    parser.add_argument("--coarse-inference-weight", type=float, default=0.5,
                        help="Parent-family score weight during fine prediction")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--songs", type=str, nargs="*", default=None,
                        help="Specific song IDs to use (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"
    print(f"Device: {device}")

    data_dir = args.data_dir.resolve()
    cache_dir = (args.cache_dir or data_dir / "cache").resolve()
    songs_dir = data_dir / "songs"
    ann_dir = data_dir / "annotations"

    # ---- Collect songs ----
    song_ids = []
    if args.songs:
        song_ids = args.songs
    else:
        for d in sorted(ann_dir.iterdir()):
            if d.is_dir() and (songs_dir / f"{d.name}.mp3").exists():
                if list(d.glob("*.annotation.json")):
                    song_ids.append(d.name)

    print(f"Songs: {len(song_ids)}")
    if len(song_ids) < 2:
        print("Need at least 2 songs")
        return

    # ---- Step 1: Extract MERT embeddings (cache) ----
    print("\n=== Step 1: MERT embedding extraction ===")
    extractor = MERTFeatureExtractor(device=device)
    features: Dict[str, torch.Tensor] = {}
    labels_dict: Dict[str, torch.Tensor] = {}

    for sid in tqdm(song_ids, desc="Extracting MERT embeddings", unit="song", colour="cyan"):
        audio_path = songs_dir / f"{sid}.mp3"
        ann_path = ann_dir / sid / f"{sid}.annotation.json"
        waveform = load_audio(audio_path)
        audio_dur = len(waveform) / SAMPLE_RATE
        emb = extractor.extract_all(audio_path, cache_dir)
        segs = load_annotation(ann_path)
        lab = segments_to_frame_labels(segs, emb.shape[0], audio_dur)
        min_len = min(emb.shape[0], lab.shape[0])
        features[sid] = emb[:min_len]
        labels_dict[sid] = lab[:min_len]
        unlabeled = (lab < 0).sum().item()

    # ---- Step 2: Train/Val/Test split ----
    available = set(features.keys())
    train_ids = [s for s in TRAIN_IDS if s in available]
    val_ids = [s for s in VAL_IDS if s in available]
    test_ids = [s for s in TEST_IDS if s in available]
    print(f"\n=== Step 2: Train {len(train_ids)} / Val {len(val_ids)} / Test {len(test_ids)} ===")
    if not train_ids or not val_ids or not test_ids:
        raise RuntimeError(
            "The fixed split is incomplete. Check songs/ and annotations/, or "
            "do not use --songs for a full fixed-split experiment."
        )

    train_ds = ConcatDataset([
        FrameDataset(
            features[s],
            labels_dict[s],
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
    model = StructureBiLSTM(
        input_dim=768, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sw = args.smooth_window  # median filter window for segment predictions
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    checkpoint_config = {
        "task": "frame",
        "input_dim": 768,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "chunk_size": args.chunk_size,
        "smooth_window": args.smooth_window,
        "coarse_loss_weight": args.coarse_loss_weight,
        "coarse_inference_weight": args.coarse_inference_weight,
        "seed": args.seed,
        "fine_labels": LABELS,
        "coarse_labels": COARSE_LABELS,
        "fine_to_coarse": {
            label: coarse_label_for(label) for label in LABELS
        },
    }

    # Train with progress bar + early stopping on val
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep", colour="green")
    for epoch in pbar:
        train_loss = train_one_epoch(
            model,
            train_dl,
            optimizer,
            device,
            coarse_loss_weight=args.coarse_loss_weight,
        )
        val_metrics = evaluate_song_collection(
            model,
            features,
            labels_dict,
            val_ids,
            device,
            chunk_size=args.chunk_size,
            coarse_weight=args.coarse_inference_weight,
        )
        pbar.set_postfix(
            loss=f"{train_loss:.3f}",
            val_acc=f"{val_metrics['accuracy']:.3f}",
            val_f1=f"{val_metrics['macro_f1']:.3f}",
            coarse_f1=f"{val_metrics['coarse_macro_f1']:.3f}",
        )

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(data_dir / "best_model.pt", model, checkpoint_config)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                pbar.set_description(f"Training [stop@e{best_epoch}]")
                break

    # Load best model
    state_dict, _ = load_checkpoint(data_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    print(f"\nBest epoch: {best_epoch}, Val macro_f1: {best_val_f1:.4f}")

    # ---- Test evaluation ----
    print(f"\n=== Test ({len(test_ids)} songs) ===")
    all_test_metrics = []
    for test_sid in tqdm(test_ids, desc="Evaluating", unit="song", colour="yellow"):
        waveform = load_audio(songs_dir / f"{test_sid}.mp3")
        audio_dur = len(waveform) / SAMPLE_RATE
        frame_hop = audio_dur / max(labels_dict[test_sid].shape[0], 1)
        gt_segs = load_annotation(ann_dir / test_sid / f"{test_sid}.annotation.json")
        predictions = predict_hierarchical_sequence(
            model,
            features[test_sid],
            device,
            chunk_size=args.chunk_size,
            stride=max(1, args.chunk_size // 2),
            coarse_weight=args.coarse_inference_weight,
        )
        full_preds = predictions["fine"]
        metrics = compute_metrics(
            full_preds,
            labels_dict[test_sid],
            all_coarse_preds=predictions["coarse_direct"],
            frame_hop_s=frame_hop,
            gt_segments=gt_segs,
            smooth_window=sw,
        )
        metrics["test_song"] = test_sid
        all_test_metrics.append(metrics)

        # Export predictions
        preds_dir = data_dir / "predictions"
        preds_dir.mkdir(exist_ok=True)
        export_preds = full_preds.clone()
        export_preds[labels_dict[test_sid] < 0] = -1
        pred_segs = frames_to_segments(export_preds, frame_hop, smooth_window=sw)
        coarse_gt_segs = segments_to_coarse(gt_segs)
        coarse_pred_segs = segments_to_coarse(pred_segs)
        (preds_dir / f"{test_sid}.prediction.json").write_text(
            json.dumps({
                "song_id": test_sid,
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
            }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Summary ----
    print("\n=== Test Summary ===")
    accs = [m["accuracy"] for m in all_test_metrics]
    f1s = [m["macro_f1"] for m in all_test_metrics]
    print(f"Frame Accuracy:  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"Frame Macro F1:  mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    coarse_accs = [m["coarse_accuracy"] for m in all_test_metrics]
    coarse_f1s = [m["coarse_macro_f1"] for m in all_test_metrics]
    print(f"Coarse Accuracy: mean={np.mean(coarse_accs):.4f}  std={np.std(coarse_accs):.4f}")
    print(f"Coarse Macro F1: mean={np.mean(coarse_f1s):.4f}  std={np.std(coarse_f1s):.4f}")

    for test_sid, m in zip(test_ids, all_test_metrics):
        seg_str = (
            f"  seg_f1={m['macro_seg_f1']:.4f}"
            f"  bnd@0.5={m['boundary_f1_0_5s']:.4f}"
            f"  bnd@3={m['boundary_f1_3s']:.4f}"
            if 'macro_seg_f1' in m else ""
        )
        print(f"  {test_sid}: acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}{seg_str}")

    if all_test_metrics and 'macro_seg_f1' in all_test_metrics[0]:
        seg_f1s = [m['macro_seg_f1'] for m in all_test_metrics]
        bnd_f1s = [m['boundary_f1_0_5s'] for m in all_test_metrics]
        bnd_f1s_3s = [m['boundary_f1_3s'] for m in all_test_metrics]
        print(f"\nSegment Macro F1: mean={np.mean(seg_f1s):.4f}")
        print(f"Boundary F1 @0.5s: mean={np.mean(bnd_f1s):.4f}")
        print(f"Boundary F1 @3.0s: mean={np.mean(bnd_f1s_3s):.4f}")

    print("\nPer-class Frame F1:")
    class_f1s = defaultdict(list)
    for m in all_test_metrics:
        for lb, f1 in m["per_class_f1"].items():
            class_f1s[lb].append(f1)
    for lb in LABELS:
        if lb in class_f1s:
            print(f"  {lb:22s}  mean={np.mean(class_f1s[lb]):.4f}")

    # Save
    results = {
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "seed": args.seed,
        "coarse_loss_weight": args.coarse_loss_weight,
        "coarse_inference_weight": args.coarse_inference_weight,
        "fine_labels": LABELS,
        "coarse_labels": COARSE_LABELS,
        "best_val_macro_f1": round(best_val_f1, 4), "best_epoch": best_epoch,
        "test_frame_acc_mean": round(np.mean(accs), 4),
        "test_frame_acc_std": round(np.std(accs), 4),
        "test_frame_macro_f1_mean": round(np.mean(f1s), 4),
        "test_frame_macro_f1_std": round(np.std(f1s), 4),
        "test_coarse_acc_mean": round(np.mean(coarse_accs), 4),
        "test_coarse_acc_std": round(np.std(coarse_accs), 4),
        "test_coarse_macro_f1_mean": round(np.mean(coarse_f1s), 4),
        "test_coarse_macro_f1_std": round(np.std(coarse_f1s), 4),
        "per_class_frame_f1": {lb: round(np.mean(class_f1s.get(lb, [0])), 4) for lb in LABELS},
        "per_song": all_test_metrics,
    }
    if all_test_metrics and 'macro_seg_f1' in all_test_metrics[0]:
        results["test_seg_macro_f1_mean"] = round(np.mean(seg_f1s), 4)
        results["test_boundary_f1_0_5s_mean"] = round(np.mean(bnd_f1s), 4)
        results["test_boundary_f1_3s_mean"] = round(np.mean(bnd_f1s_3s), 4)
    (data_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {data_dir / 'results.json'}")


if __name__ == "__main__":
    main()
