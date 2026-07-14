"""
Test best model on held-out test set.
Usage: python test_frame.py --data-dir .
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Import shared pieces from training script
from train_frame import (
    COARSE_LABELS, LABELS, SAMPLE_RATE,
    MERTFeatureExtractor, StructureBiLSTM,
    load_audio, load_annotation, segments_to_frame_labels,
    frames_to_segments, compute_metrics,
)
from research_utils import (
    TEST_IDS,
    coarse_label_for,
    load_checkpoint,
    predict_hierarchical_sequence,
    segments_to_coarse,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Test MERT structure segmenter")
    parser.add_argument("--data-dir", type=Path, default=Path("."),
                        help="train/ directory containing songs/ and annotations/")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to best_model.pt (default: data-dir/best_model.pt)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory for MERT embeddings")
    parser.add_argument("--smooth-window", type=int, default=75,
                        help="Median filter window size for segment smoothing (0=off)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Inference chunk size (default: checkpoint value or 512)")
    parser.add_argument("--coarse-inference-weight", type=float, default=None,
                        help="Override checkpoint parent-family score weight")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    songs_dir = data_dir / "songs"
    ann_dir = data_dir / "annotations"
    cache_dir = args.cache_dir or data_dir / "cache"
    model_path = args.model_path or data_dir / "best_model.pt"
    preds_dir = data_dir / "predictions"
    preds_dir.mkdir(exist_ok=True)
    sw = args.smooth_window
    print(f"Smooth window: {sw} frames ({sw*0.013:.1f}s at 75Hz)")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python train_frame.py --data-dir .")
        sys.exit(1)
    state_dict, checkpoint_config = load_checkpoint(model_path, map_location=DEVICE)
    if "fine_classifier.weight" not in state_dict:
        raise ValueError(
            "This is a legacy flat-label checkpoint. Retrain with the "
            "hierarchical fine/coarse model before testing."
        )
    checkpoint_classes = state_dict["fine_classifier.weight"].shape[0]
    if checkpoint_classes != len(LABELS):
        raise ValueError(
            f"Checkpoint predicts {checkpoint_classes} classes, but the current "
            f"fine schema has {len(LABELS)} labels. Retrain the model."
        )
    checkpoint_coarse_classes = state_dict["coarse_classifier.weight"].shape[0]
    if checkpoint_coarse_classes != len(COARSE_LABELS):
        raise ValueError(
            f"Checkpoint predicts {checkpoint_coarse_classes} coarse classes, "
            f"but the current schema has {len(COARSE_LABELS)}."
        )
    chunk_size = args.chunk_size or checkpoint_config.get("chunk_size", 512)
    coarse_inference_weight = (
        args.coarse_inference_weight
        if args.coarse_inference_weight is not None
        else checkpoint_config.get("coarse_inference_weight", 0.5)
    )

    # ---- Step 1: Extract MERT features (cached) ----
    print("=== Step 1: Extract MERT embeddings ===")
    extractor = MERTFeatureExtractor(device=DEVICE)
    features = {}
    labels_dict = {}

    available = set()
    for d in sorted(ann_dir.iterdir()):
        if d.is_dir() and (songs_dir / f"{d.name}.mp3").exists():
            if list(d.glob("*.annotation.json")):
                available.add(d.name)

    test_ids = [s for s in TEST_IDS if s in available]
    if not test_ids:
        raise RuntimeError("No held-out test songs with audio and annotations were found")
    all_needed = set(test_ids)
    print(f"Test songs: {len(test_ids)}")

    for sid in tqdm(sorted(all_needed), desc="Extracting", unit="song", colour="cyan"):
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

    # ---- Step 2: Load model ----
    print(f"\n=== Step 2: Load model from {model_path} ===")
    model = StructureBiLSTM(
        input_dim=checkpoint_config.get(
            "input_dim", state_dict["lstm.weight_ih_l0"].shape[1]
        ),
        hidden_dim=checkpoint_config.get(
            "hidden_dim", state_dict["lstm.weight_hh_l0"].shape[1]
        ),
        num_layers=checkpoint_config.get(
            "num_layers",
            len([
                key for key in state_dict
                if key.startswith("lstm.weight_ih_l") and not key.endswith("_reverse")
            ]),
        ),
        dropout=checkpoint_config.get("dropout", 0.5),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Step 3: Test evaluation ----
    print(f"\n=== Step 3: Test ({len(test_ids)} songs) ===")
    all_metrics = []

    for test_sid in tqdm(test_ids, desc="Evaluating", unit="song", colour="yellow"):
        waveform = load_audio(songs_dir / f"{test_sid}.mp3")
        audio_dur = len(waveform) / SAMPLE_RATE
        frame_hop = audio_dur / max(labels_dict[test_sid].shape[0], 1)
        gt_segs = load_annotation(ann_dir / test_sid / f"{test_sid}.annotation.json")
        predictions = predict_hierarchical_sequence(
            model,
            features[test_sid],
            DEVICE,
            chunk_size=chunk_size,
            stride=max(1, chunk_size // 2),
            coarse_weight=coarse_inference_weight,
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
        all_metrics.append(metrics)

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
    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]
    print(f"Frame Accuracy:  mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"Frame Macro F1:  mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    coarse_accs = [m["coarse_accuracy"] for m in all_metrics]
    coarse_f1s = [m["coarse_macro_f1"] for m in all_metrics]
    print(f"Coarse Accuracy: mean={np.mean(coarse_accs):.4f}  std={np.std(coarse_accs):.4f}")
    print(f"Coarse Macro F1: mean={np.mean(coarse_f1s):.4f}  std={np.std(coarse_f1s):.4f}")

    for test_sid, m in zip(test_ids, all_metrics):
        seg_str = ""
        if 'macro_seg_f1' in m:
            seg_str = (
                f"  seg_f1={m['macro_seg_f1']:.4f}"
                f"  bnd@0.5={m['boundary_f1_0_5s']:.4f}"
                f"  bnd@3={m['boundary_f1_3s']:.4f}"
            )
        print(f"  {test_sid}: acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}{seg_str}")

    if all_metrics and 'macro_seg_f1' in all_metrics[0]:
        seg_f1s = [m['macro_seg_f1'] for m in all_metrics]
        bnd_f1s = [m['boundary_f1_0_5s'] for m in all_metrics]
        bnd_f1s_3s = [m['boundary_f1_3s'] for m in all_metrics]
        print(f"\nSegment Macro F1: mean={np.mean(seg_f1s):.4f}")
        print(f"Boundary F1 @0.5s: mean={np.mean(bnd_f1s):.4f}")
        print(f"Boundary F1 @3.0s: mean={np.mean(bnd_f1s_3s):.4f}")

    print("\nPer-class Frame F1:")
    class_f1s = defaultdict(list)
    for m in all_metrics:
        for lb, f1 in m["per_class_f1"].items():
            class_f1s[lb].append(f1)
    for lb in LABELS:
        if lb in class_f1s:
            print(f"  {lb:22s}  mean={np.mean(class_f1s[lb]):.4f}")

    # Save test results
    results = {
        "n_test": len(test_ids),
        "chunk_size": chunk_size,
        "coarse_inference_weight": coarse_inference_weight,
        "test_frame_acc_mean": round(np.mean(accs), 4),
        "test_frame_acc_std": round(np.std(accs), 4),
        "test_frame_macro_f1_mean": round(np.mean(f1s), 4),
        "test_frame_macro_f1_std": round(np.std(f1s), 4),
        "test_coarse_acc_mean": round(np.mean(coarse_accs), 4),
        "test_coarse_acc_std": round(np.std(coarse_accs), 4),
        "test_coarse_macro_f1_mean": round(np.mean(coarse_f1s), 4),
        "test_coarse_macro_f1_std": round(np.std(coarse_f1s), 4),
        "per_class_frame_f1": {lb: round(np.mean(class_f1s.get(lb, [0])), 4) for lb in LABELS},
        "per_song": all_metrics,
    }
    if all_metrics and 'macro_seg_f1' in all_metrics[0]:
        results["test_seg_macro_f1_mean"] = round(np.mean(seg_f1s), 4)
        results["test_boundary_f1_0_5s_mean"] = round(np.mean(bnd_f1s), 4)
        results["test_boundary_f1_3s_mean"] = round(np.mean(bnd_f1s_3s), 4)
    (data_dir / "test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {data_dir / 'test_results.json'}")
    print(f"Predictions saved to {preds_dir}/")


if __name__ == "__main__":
    main()
