"""
Test bar-level model on held-out evaluation splits.
Usage: python test_bar.py --data-dir .
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from train_bar import (
    COARSE_LABELS, LABELS, SAMPLE_RATE,
    MERT_MODEL_NAME, MUQ_MODEL_NAME,
    StructureBiLSTM, create_feature_extractor,
    backbone_artifact_suffix,
    load_audio, load_annotation, load_struct,
    BAR_CONTEXT_FEATURE_CHOICES,
    append_bar_context_features, bar_context_feature_dim,
    bar_pooling, segments_to_bar_labels, bars_to_segments,
    coarse_bars_to_segments, compute_coarse_metrics, compute_metrics,
)
from research_utils import (
    PAPER_TEST_IDS,
    TEST_IDS,
    annotation_end_time,
    coarse_label_for,
    extend_tail_downbeats as extend_tail_downbeat_grid,
    load_checkpoint,
    predict_coarse_sequence,
    predict_hierarchical_sequence,
    segments_to_coarse,
)
from postprocess import config_from_mapping, make_postprocess_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tail_extension_summary(stats_by_song):
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


def main():
    parser = argparse.ArgumentParser(description="Test bar-level frozen-backbone segmenter")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--backbone",
        choices=["mert", "muq"],
        default=None,
        help="Frozen audio representation backbone. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="HuggingFace/model cache root for downloaded backbones.",
    )
    parser.add_argument("--pool-mode", type=str, default=None,
                        choices=["mean", "meanmax", "meanmaxstd"])
    parser.add_argument(
        "--bar-context-features",
        choices=BAR_CONTEXT_FEATURE_CHOICES,
        default=None,
        help="Override checkpoint per-bar timing feature mode.",
    )
    parser.add_argument("--mert-layers", type=int, nargs="*", default=None,
                        help="MERT layers (must match training)")
    parser.add_argument("--muq-layers", type=int, nargs="*", default=None,
                        help="MuQ layers (must match training)")
    parser.add_argument("--muq-chunk-seconds", type=float, default=None,
                        help="Override MuQ forward chunk seconds.")
    parser.add_argument("--muq-overlap-seconds", type=float, default=None,
                        help="Override MuQ chunk overlap seconds.")
    parser.add_argument("--coarse-inference-weight", type=float, default=None,
                        help="Override checkpoint parent-family score weight")
    parser.add_argument(
        "--eval-split",
        type=str,
        default="dev_test",
        choices=["dev_test", "paper_test", "all"],
        help=(
            "Evaluation split. dev_test is the historical 5-song test set; "
            "paper_test is the frozen 10-song blind set."
        ),
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "smooth", "merge", "full"],
        default=None,
        help=(
            "Override checkpoint postprocess mode for coarse models. "
            "If omitted, uses checkpoint metadata or none. merge only "
            "applies short-segment merging."
        ),
    )
    parser.add_argument(
        "--postprocess-smooth-window",
        type=int,
        default=None,
        help="Override postprocess smoothing window.",
    )
    parser.add_argument(
        "--postprocess-transition-penalty",
        type=float,
        default=None,
        help="Override postprocess transition penalty.",
    )
    parser.add_argument(
        "--postprocess-min-bars",
        type=str,
        default=None,
        help="Override min-duration map, e.g. verse=4,chorus=4.",
    )
    parser.set_defaults(extend_tail_downbeats=None)
    parser.add_argument(
        "--extend-tail-downbeats",
        dest="extend_tail_downbeats",
        action="store_true",
        help="Extend final downbeat grid to annotation end.",
    )
    parser.add_argument(
        "--no-extend-tail-downbeats",
        dest="extend_tail_downbeats",
        action="store_false",
        help="Disable tail downbeat extension.",
    )
    parser.add_argument(
        "--tail-extension-lookback",
        type=int,
        default=None,
        help="Recent bar intervals used to extrapolate tail downbeats.",
    )
    parser.add_argument(
        "--tail-extension-tolerance",
        type=float,
        default=None,
        help="Do not extend when annotation end is within this many seconds.",
    )
    parser.add_argument(
        "--tail-extension-max-bars",
        type=int,
        default=None,
        help="Safety cap for newly added tail downbeats.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    songs_dir = data_dir / "songs"
    ann_dir = data_dir / "annotations"
    struct_dir = data_dir / "struct"
    cache_dir = args.cache_dir or data_dir / "cache"
    model_cache_dir = (args.model_cache_dir or data_dir.parent / ".hf").resolve()
    requested_backbone = args.backbone
    requested_suffix = backbone_artifact_suffix(requested_backbone) if requested_backbone else ""
    if args.model_path:
        model_path = args.model_path
    elif requested_suffix and (data_dir / f"best_model_bar_coarse{requested_suffix}.pt").exists():
        model_path = data_dir / f"best_model_bar_coarse{requested_suffix}.pt"
    elif requested_suffix and (data_dir / f"best_model_bar{requested_suffix}.pt").exists():
        model_path = data_dir / f"best_model_bar{requested_suffix}.pt"
    elif (data_dir / "best_model_bar_coarse.pt").exists():
        model_path = data_dir / "best_model_bar_coarse.pt"
    else:
        model_path = data_dir / "best_model_bar.pt"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python train_bar.py --data-dir .")
        sys.exit(1)

    state_dict, checkpoint_config = load_checkpoint(model_path, map_location=DEVICE)
    checkpoint_postprocess = config_from_mapping(
        checkpoint_config.get("postprocess")
    )
    use_checkpoint_postprocess_values = (
        args.postprocess is None
        or (
            checkpoint_postprocess.enabled
            and args.postprocess == checkpoint_postprocess.mode
        )
    )
    try:
        postprocess_config = make_postprocess_config(
            mode=(
                args.postprocess
                if args.postprocess is not None
                else checkpoint_postprocess.mode
            ),
            smoothing_window=(
                args.postprocess_smooth_window
                if args.postprocess_smooth_window is not None
                else (
                    checkpoint_postprocess.smoothing_window
                    if use_checkpoint_postprocess_values
                    else 3
                )
            ),
            transition_penalty=(
                args.postprocess_transition_penalty
                if args.postprocess_transition_penalty is not None
                else (
                    checkpoint_postprocess.transition_penalty
                    if use_checkpoint_postprocess_values
                    else 0.12
                )
            ),
            min_bars_spec=args.postprocess_min_bars,
            min_bars=checkpoint_postprocess.min_bars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    checkpoint_tail_config = checkpoint_config.get("tail_downbeat_extension", {})
    use_tail_downbeat_extension = (
        args.extend_tail_downbeats
        if args.extend_tail_downbeats is not None
        else checkpoint_tail_config.get("enabled", True)
    )
    tail_extension_lookback = (
        args.tail_extension_lookback
        if args.tail_extension_lookback is not None
        else checkpoint_tail_config.get("lookback_bars", 8)
    )
    tail_extension_tolerance = (
        args.tail_extension_tolerance
        if args.tail_extension_tolerance is not None
        else checkpoint_tail_config.get("tolerance_s", 0.5)
    )
    tail_extension_max_bars = (
        args.tail_extension_max_bars
        if args.tail_extension_max_bars is not None
        else checkpoint_tail_config.get("max_new_downbeats", 64)
    )
    if tail_extension_lookback < 1:
        parser.error("--tail-extension-lookback must be >= 1")
    if tail_extension_tolerance < 0:
        parser.error("--tail-extension-tolerance must be >= 0")
    if tail_extension_max_bars < 1:
        parser.error("--tail-extension-max-bars must be >= 1")
    muq_chunk_seconds = (
        args.muq_chunk_seconds
        if args.muq_chunk_seconds is not None
        else checkpoint_config.get("muq_chunk_seconds", 30.0)
    )
    muq_overlap_seconds = (
        args.muq_overlap_seconds
        if args.muq_overlap_seconds is not None
        else checkpoint_config.get("muq_overlap_seconds", 1.0)
    )
    if muq_chunk_seconds < 0:
        parser.error("--muq-chunk-seconds must be >= 0")
    if muq_overlap_seconds < 0:
        parser.error("--muq-overlap-seconds must be >= 0")
    tail_extension_config = {
        "enabled": bool(use_tail_downbeat_extension),
        "target": "annotation_end",
        "lookback_bars": tail_extension_lookback,
        "tolerance_s": tail_extension_tolerance,
        "max_new_downbeats": tail_extension_max_bars,
    }
    target_level = checkpoint_config.get("target_level", "hierarchical")
    if target_level not in {"coarse", "hierarchical"}:
        raise ValueError(f"Unknown checkpoint target_level: {target_level}")
    if target_level == "hierarchical":
        if "fine_classifier.weight" not in state_dict:
            raise ValueError(
                "This is a legacy flat-label checkpoint. Retrain with the "
                "current coarse or hierarchical model before testing."
            )
        checkpoint_classes = state_dict["fine_classifier.weight"].shape[0]
        if checkpoint_classes != len(LABELS):
            raise ValueError(
                f"Checkpoint predicts {checkpoint_classes} classes, but the "
                f"current fine schema has {len(LABELS)} labels."
            )
    checkpoint_coarse_classes = state_dict["coarse_classifier.weight"].shape[0]
    if checkpoint_coarse_classes != len(COARSE_LABELS):
        raise ValueError(
            f"Checkpoint predicts {checkpoint_coarse_classes} coarse classes, "
            f"but the current schema has {len(COARSE_LABELS)}."
        )
    pool_mode = args.pool_mode or checkpoint_config.get("pool_mode") or "mean"
    bar_context_features = (
        args.bar_context_features
        if args.bar_context_features is not None
        else checkpoint_config.get("bar_context_features", "none")
    )
    backbone = args.backbone or checkpoint_config.get("backbone", "mert")
    if backbone not in {"mert", "muq"}:
        raise ValueError(f"Unknown backbone: {backbone}")
    mert_layers = (
        args.mert_layers
        if args.mert_layers is not None
        else checkpoint_config.get("mert_layers")
    )
    muq_layers = (
        args.muq_layers
        if args.muq_layers is not None
        else checkpoint_config.get("muq_layers")
    )
    feature_layers = (
        muq_layers
        if backbone == "muq"
        else mert_layers
    )
    if feature_layers is None:
        feature_layers = checkpoint_config.get("feature_layers")
    print(f"Pool mode: {pool_mode}")
    print(
        f"Bar context features: {bar_context_features} "
        f"(dim={bar_context_feature_dim(bar_context_features)})"
    )
    print(f"Backbone: {backbone}")
    print(f"Backbone model: {MUQ_MODEL_NAME if backbone == 'muq' else MERT_MODEL_NAME}")
    print(f"Feature layers: {feature_layers or 'last hidden state only'}")
    print(f"Target level: {target_level}")
    print(f"Postprocess: {postprocess_config.summary()}")
    print(
        "Tail downbeat extension: "
        f"{'on' if use_tail_downbeat_extension else 'off'} "
        f"(target=annotation_end)"
    )
    coarse_inference_weight = (
        args.coarse_inference_weight
        if args.coarse_inference_weight is not None
        else checkpoint_config.get("coarse_inference_weight", 0.5)
    )
    print(f"Coarse inference weight: {coarse_inference_weight}")

    # ---- Step 1: Extract frozen-backbone features ----
    print(f"=== Step 1: Extract {backbone.upper()} embeddings ===")
    if backbone == "muq":
        print(f"Model cache: {model_cache_dir}")
        print(f"MuQ chunking: {muq_chunk_seconds}s chunks, {muq_overlap_seconds}s overlap")
    extractor = create_feature_extractor(
        backbone,
        device=DEVICE,
        feature_layers=feature_layers,
        model_cache_dir=model_cache_dir,
        muq_chunk_seconds=muq_chunk_seconds,
        muq_overlap_seconds=muq_overlap_seconds,
    )
    frame_features = {}
    all_downbeats = {}

    available = set()
    for d in sorted(ann_dir.iterdir()):
        if d.is_dir() and (songs_dir / f"{d.name}.mp3").exists():
            if list(d.glob("*.annotation.json")):
                available.add(d.name)
    split_ids = {
        "dev_test": TEST_IDS,
        "paper_test": PAPER_TEST_IDS,
        "all": TEST_IDS + PAPER_TEST_IDS,
    }[args.eval_split]
    test_ids = [s for s in split_ids if s in available]
    if not test_ids:
        raise RuntimeError(
            f"No {args.eval_split} songs with audio and annotations were found"
        )
    all_needed = set(test_ids)
    print(f"Evaluation split: {args.eval_split}")
    print(f"Test songs: {len(test_ids)}")

    for sid in tqdm(sorted(all_needed), desc=f"Extracting {backbone.upper()}", unit="song", colour="cyan"):
        audio_path = songs_dir / f"{sid}.mp3"
        emb = extractor.extract_all(audio_path, cache_dir)
        frame_features[sid] = emb

    # ---- Step 2: Bar pooling ----
    print("\n=== Step 2: Bar pooling ===")
    bar_features = {}
    bar_labels_dict = {}
    tail_extension_stats = {}

    for sid in test_ids:
        audio_path = songs_dir / f"{sid}.mp3"
        ann_path = ann_dir / sid / f"{sid}.annotation.json"
        struct_path = struct_dir / f"{sid}.json"

        waveform = load_audio(audio_path)
        audio_dur = len(waveform) / SAMPLE_RATE
        struct = load_struct(struct_path)
        gt_segs = load_annotation(ann_path)
        downbeats = struct["downbeats"]
        if use_tail_downbeat_extension:
            downbeats, tail_stats = extend_tail_downbeat_grid(
                downbeats,
                target_end_s=annotation_end_time(gt_segs),
                lookback_bars=tail_extension_lookback,
                tolerance_s=tail_extension_tolerance,
                max_new_downbeats=tail_extension_max_bars,
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

        bar_emb = bar_pooling(frame_features[sid], struct["beats"], downbeats,
                              audio_dur, pool_mode=pool_mode)
        bar_emb = append_bar_context_features(
            bar_emb,
            struct.get("beats", []),
            downbeats,
            audio_dur,
            mode=bar_context_features,
        )
        bar_lab = segments_to_bar_labels(gt_segs, downbeats)
        if len(bar_emb) != len(bar_lab):
            raise ValueError(
                f"{sid}: pooled bars ({len(bar_emb)}) != labels ({len(bar_lab)})"
            )

        bar_features[sid] = bar_emb
        bar_labels_dict[sid] = bar_lab
        all_downbeats[sid] = downbeats
        tail_note = (
            f", +{int(tail_stats.get('added_downbeats', 0))} tail downbeats"
            if int(tail_stats.get("added_downbeats", 0)) > 0
            else ""
        )
        print(f"  {sid}: {bar_emb.shape[0]} bars, {bar_emb.shape[1]} dim{tail_note}")

    tail_summary = _tail_extension_summary(tail_extension_stats)
    if use_tail_downbeat_extension:
        print(
            "Tail downbeat extension: "
            f"{tail_summary['songs_extended']} songs, "
            f"+{tail_summary['total_added_downbeats']} downbeats"
        )

    # ---- Step 3: Load model ----
    print(f"\n=== Step 3: Load model from {model_path} ===")
    input_dim = bar_features[test_ids[0]].shape[1]
    checkpoint_input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
    if input_dim != checkpoint_input_dim:
        raise ValueError(
            f"Feature dim {input_dim} does not match checkpoint dim "
            f"{checkpoint_input_dim}. Pass the training-time --pool-mode and "
            "--mert-layers values."
        )
    model = StructureBiLSTM(
        input_dim=input_dim,
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
        target_level=target_level,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dim: {input_dim}")

    # ---- Step 4: Test evaluation ----
    print(f"\n=== Step 4: Test ({len(test_ids)} songs) ===")
    all_metrics = []
    split_suffix = "" if args.eval_split == "dev_test" else f"_{args.eval_split}"
    artifact_suffix = backbone_artifact_suffix(backbone)
    preds_dir = data_dir / (
        f"predictions_bar_coarse{artifact_suffix}{split_suffix}"
        if target_level == "coarse"
        else f"predictions_bar{artifact_suffix}{split_suffix}"
    )
    preds_dir.mkdir(exist_ok=True)

    for test_sid in tqdm(test_ids, desc="Evaluating", unit="song", colour="yellow"):
        gt_segs = load_annotation(ann_dir / test_sid / f"{test_sid}.annotation.json")
        downbeats = all_downbeats[test_sid]
        if target_level == "coarse":
            full_preds = predict_coarse_sequence(
                model,
                bar_features[test_sid],
                DEVICE,
                postprocess_config=postprocess_config,
            )
            metrics = compute_coarse_metrics(
                full_preds,
                bar_labels_dict[test_sid],
                downbeats=downbeats,
                gt_segments=gt_segs,
            )
        else:
            predictions = predict_hierarchical_sequence(
                model,
                bar_features[test_sid],
                DEVICE,
                coarse_weight=coarse_inference_weight,
            )
            full_preds = predictions["fine"]
            metrics = compute_metrics(
                full_preds,
                bar_labels_dict[test_sid],
                all_coarse_preds=predictions["coarse_direct"],
                downbeats=downbeats,
                gt_segments=gt_segs,
            )
        metrics["test_song"] = test_sid
        all_metrics.append(metrics)

        # Export predictions
        export_preds = full_preds.clone()
        export_preds[bar_labels_dict[test_sid] < 0] = -1
        coarse_gt_segs = segments_to_coarse(gt_segs)
        if target_level == "coarse":
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

        (preds_dir / f"{test_sid}.prediction.json").write_text(
            json.dumps(export_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ---- Summary ----
    print("\n=== Test Summary ===")
    accs = [m["accuracy"] for m in all_metrics]
    f1s = [m["macro_f1"] for m in all_metrics]
    print(f"Bar Accuracy:     mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"Bar Macro F1:     mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    if target_level == "hierarchical":
        coarse_accs = [m["coarse_accuracy"] for m in all_metrics]
        coarse_f1s = [m["coarse_macro_f1"] for m in all_metrics]
        print(f"Coarse Accuracy:  mean={np.mean(coarse_accs):.4f}  std={np.std(coarse_accs):.4f}")
        print(f"Coarse Macro F1:  mean={np.mean(coarse_f1s):.4f}  std={np.std(coarse_f1s):.4f}")

    for test_sid, m in zip(test_ids, all_metrics):
        seg_str = (
            f"  seg_f1={m.get('macro_seg_f1', 0):.4f}"
            f"  bnd@0.5={m.get('boundary_f1_0_5s', 0):.4f}"
            f"  bnd@3={m.get('boundary_f1_3s', 0):.4f}"
        )
        print(f"  {test_sid}: acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}{seg_str}")

    if all_metrics and 'macro_seg_f1' in all_metrics[0]:
        seg_f1s = [m['macro_seg_f1'] for m in all_metrics]
        bnd_f1s = [m['boundary_f1_0_5s'] for m in all_metrics]
        bnd_f1s_3s = [m['boundary_f1_3s'] for m in all_metrics]
        print(f"\nSegment Macro F1: mean={np.mean(seg_f1s):.4f}")
        print(f"Boundary F1 @0.5s: mean={np.mean(bnd_f1s):.4f}")
        print(f"Boundary F1 @3.0s: mean={np.mean(bnd_f1s_3s):.4f}")

    print(f"\nPer-class {'Coarse ' if target_level == 'coarse' else ''}Bar F1:")
    class_f1s = defaultdict(list)
    for m in all_metrics:
        for lb, f1 in m["per_class_f1"].items():
            class_f1s[lb].append(f1)
    report_labels = COARSE_LABELS if target_level == "coarse" else LABELS
    for lb in report_labels:
        if lb in class_f1s:
            print(f"  {lb:22s}  mean={np.mean(class_f1s[lb]):.4f}")

    results = {
        "eval_split": args.eval_split,
        "n_test": len(test_ids),
        "target_level": target_level,
        "backbone": backbone,
        "backbone_model": MUQ_MODEL_NAME if backbone == "muq" else MERT_MODEL_NAME,
        "pool_mode": pool_mode,
        "bar_context_features": bar_context_features,
        "bar_context_feature_dim": bar_context_feature_dim(bar_context_features),
        "feature_layers": feature_layers,
        "muq_chunk_seconds": muq_chunk_seconds,
        "muq_overlap_seconds": muq_overlap_seconds,
        "mert_layers": mert_layers,
        "muq_layers": muq_layers,
        "coarse_inference_weight": coarse_inference_weight,
        "postprocess": postprocess_config.as_dict(),
        "tail_downbeat_extension": tail_extension_config,
        "tail_downbeat_extension_summary": tail_summary,
        "input_dim": input_dim,
        "test_bar_acc_mean": round(np.mean(accs), 4),
        "test_bar_acc_std": round(np.std(accs), 4),
        "test_bar_macro_f1_mean": round(np.mean(f1s), 4),
        "test_bar_macro_f1_std": round(np.std(f1s), 4),
        "per_class_bar_f1": {
            lb: round(np.mean(class_f1s.get(lb, [0])), 4)
            for lb in report_labels
        },
        "per_song": all_metrics,
    }
    if target_level == "hierarchical":
        results.update({
            "test_coarse_acc_mean": round(np.mean(coarse_accs), 4),
            "test_coarse_acc_std": round(np.std(coarse_accs), 4),
            "test_coarse_macro_f1_mean": round(np.mean(coarse_f1s), 4),
            "test_coarse_macro_f1_std": round(np.std(coarse_f1s), 4),
        })
    if all_metrics and 'macro_seg_f1' in all_metrics[0]:
        results["test_seg_macro_f1_mean"] = round(np.mean(seg_f1s), 4)
        results["test_boundary_f1_0_5s_mean"] = round(np.mean(bnd_f1s), 4)
        results["test_boundary_f1_3s_mean"] = round(np.mean(bnd_f1s_3s), 4)
    results_name = (
        f"test_results_bar_coarse{artifact_suffix}{split_suffix}.json"
        if target_level == "coarse"
        else f"test_results_bar{artifact_suffix}{split_suffix}.json"
    )
    (data_dir / results_name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {data_dir / results_name}")
    print(f"Predictions saved to {preds_dir}/")


if __name__ == "__main__":
    main()
