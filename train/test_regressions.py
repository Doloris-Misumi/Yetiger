"""Fast regression tests that do not require audio files or MERT downloads."""

import tempfile
import unittest
from pathlib import Path

import torch

from research_utils import (
    ANNOTATION_ONLY_LABELS,
    COARSE_LABELS,
    FINE_TO_COARSE,
    MODEL_LABELS,
    chunk_starts,
    extend_tail_downbeats,
    fine_labels_to_coarse,
    load_checkpoint,
    predict_coarse_sequence,
    predict_hierarchical_sequence,
    predict_sequence,
    save_checkpoint,
    segments_to_coarse,
)
from postprocess import make_postprocess_config, postprocess_coarse_logits
from train_bar import (
    LABEL2ID,
    BarDataset,
    StructureBiLSTM,
    _add_song_metric_means,
    bar_pooling,
    checkpoint_selection_score,
    compute_coarse_metrics,
    create_run_directory,
    segment_level_metrics,
    segments_to_bar_labels,
)


class PipelineRegressionTests(unittest.TestCase):
    def test_end_is_annotation_only(self):
        self.assertIn("end", ANNOTATION_ONLY_LABELS)
        self.assertNotIn("end", MODEL_LABELS)
        self.assertNotIn("chant", MODEL_LABELS)
        self.assertNotIn("end", LABEL2ID)
        labels = segments_to_bar_labels(
            [(0.0, 1.0, "verse"), (1.0, 2.0, "end")],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(labels.tolist(), [LABEL2ID["verse"], -1])

    def test_two_level_label_hierarchy(self):
        self.assertEqual(FINE_TO_COARSE["instrumental_break"], "instrumental")
        self.assertEqual(FINE_TO_COARSE["solo"], "instrumental")
        self.assertEqual(FINE_TO_COARSE["post_chorus"], "chorus")
        self.assertEqual(FINE_TO_COARSE["pre_chorus_build"], "pre_chorus")
        self.assertIn("instrumental", COARSE_LABELS)

        fine = torch.tensor([
            LABEL2ID["instrumental_break"],
            LABEL2ID["solo"],
            -1,
        ])
        coarse = fine_labels_to_coarse(fine)
        self.assertEqual(coarse[0].item(), coarse[1].item())
        self.assertEqual(coarse[2].item(), -1)

        coarse_segments = segments_to_coarse([
            (0.0, 4.0, "instrumental_break"),
            (4.0, 8.0, "solo"),
            (8.0, 12.0, "chorus"),
            (12.0, 16.0, "post_chorus"),
        ])
        self.assertEqual(coarse_segments, [
            (0.0, 8.0, "instrumental"),
            (8.0, 16.0, "chorus"),
        ])

    def test_coarse_task_removes_fine_only_boundaries(self):
        fine_labels = torch.tensor([
            LABEL2ID["chorus"],
            LABEL2ID["post_chorus"],
            LABEL2ID["pre_chorus"],
            LABEL2ID["pre_chorus_build"],
        ])
        coarse_predictions = fine_labels_to_coarse(fine_labels)
        metrics = compute_coarse_metrics(
            coarse_predictions,
            fine_labels,
            downbeats=[0.0, 4.0, 8.0, 12.0, 16.0],
            gt_segments=[
                (0.0, 4.0, "chorus"),
                (4.0, 8.0, "post_chorus"),
                (8.0, 12.0, "pre_chorus"),
                (12.0, 16.0, "pre_chorus_build"),
            ],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["boundary_f1_0_5s"], 1.0)
        self.assertEqual(metrics["macro_seg_f1"], round(2 / 7, 4))

        model = StructureBiLSTM(
            input_dim=4,
            hidden_dim=2,
            num_layers=1,
            target_level="coarse",
        )
        predictions = predict_coarse_sequence(
            model, torch.randn(9, 4), "cpu", chunk_size=4, stride=2
        )
        self.assertEqual(tuple(predictions.shape), (9,))
        self.assertFalse(hasattr(model, "fine_classifier"))

    def test_coarse_postprocess_removes_interior_spike(self):
        label_to_id = {label: i for i, label in enumerate(COARSE_LABELS)}
        raw_ids = [
            label_to_id["intro"],
            label_to_id["verse"],
            label_to_id["intro"],
            label_to_id["verse"],
            label_to_id["pre_chorus"],
            label_to_id["chorus"],
        ]
        logits = torch.full((len(raw_ids), len(COARSE_LABELS)), -3.0)
        for bar_index, label_id in enumerate(raw_ids):
            logits[bar_index, label_id] = 3.0

        config = make_postprocess_config(
            mode="full",
            smoothing_window=1,
            transition_penalty=0.5,
            min_bars_spec="verse=1,pre_chorus=1,chorus=1",
        )
        predictions = postprocess_coarse_logits(logits, COARSE_LABELS, config)
        self.assertEqual(tuple(predictions.shape), (len(raw_ids),))
        self.assertEqual(predictions[0].item(), label_to_id["intro"])
        self.assertNotEqual(predictions[2].item(), label_to_id["intro"])

    def test_merge_postprocess_is_min_duration_only(self):
        config = make_postprocess_config(mode="merge")
        self.assertTrue(config.use_min_duration)
        self.assertFalse(config.use_transition_grammar)
        self.assertEqual(config.smoothing_window, 1)
        self.assertEqual(config.transition_penalty, 0.0)

    def test_tail_downbeat_extension_covers_annotation_end(self):
        extended, stats = extend_tail_downbeats(
            [0.0, 4.0, 8.0],
            target_end_s=17.5,
            lookback_bars=2,
            tolerance_s=0.5,
        )
        self.assertEqual(extended, [0.0, 4.0, 8.0, 12.0, 16.0, 17.5])
        self.assertEqual(stats["added_downbeats"], 3)

        unchanged, stats = extend_tail_downbeats(
            [0.0, 4.0, 8.0],
            target_end_s=8.2,
            tolerance_s=0.5,
        )
        self.assertEqual(unchanged, [0.0, 4.0, 8.0])
        self.assertEqual(stats["added_downbeats"], 0)

    def test_chunks_cover_tail(self):
        self.assertEqual(chunk_starts(106, 32, 32), [0, 32, 64, 74])
        dataset = BarDataset(
            torch.randn(106, 4),
            torch.zeros(106, dtype=torch.long),
            chunk_size=32,
            stride=32,
        )
        self.assertEqual(dataset.starts, [0, 32, 64, 74])

    def test_bar_pooling_preserves_short_bars_without_nan(self):
        pooled = bar_pooling(
            torch.randn(10, 3),
            beats=[],
            downbeats=[0.0, 0.01, 1.0],
            audio_duration_s=1.0,
            pool_mode="meanmaxstd",
        )
        self.assertEqual(tuple(pooled.shape), (2, 9))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_mir_boundary_tolerance(self):
        metrics = segment_level_metrics(
            [(0.0, 10.4, "intro"), (10.4, 20.0, "verse")],
            [(0.0, 10.0, "intro"), (10.0, 20.0, "verse")],
        )
        self.assertEqual(metrics["boundary_f1_exact"], 0.0)
        self.assertEqual(metrics["boundary_f1_0_5s"], 1.0)
        self.assertEqual(metrics["boundary_f1_3s"], 1.0)

    def test_checkpoint_metadata_and_complete_prediction(self):
        model = StructureBiLSTM(input_dim=4, hidden_dim=2, num_layers=1)
        predictions = predict_sequence(
            model, torch.randn(106, 4), "cpu", chunk_size=32, stride=32
        )
        self.assertEqual(tuple(predictions.shape), (106,))
        hierarchy = predict_hierarchical_sequence(
            model, torch.randn(106, 4), "cpu", chunk_size=32, stride=32
        )
        self.assertEqual(tuple(hierarchy["fine"].shape), (106,))
        self.assertEqual(tuple(hierarchy["coarse"].shape), (106,))
        self.assertTrue(torch.equal(
            hierarchy["coarse"],
            fine_labels_to_coarse(hierarchy["fine"]),
        ))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, {"input_dim": 4, "seed": 42})
            state_dict, config = load_checkpoint(path)
        self.assertEqual(config["seed"], 42)
        self.assertIn("lstm.weight_ih_l0", state_dict)
        self.assertIn("fine_classifier.weight", state_dict)
        self.assertIn("coarse_classifier.weight", state_dict)

    def test_validation_selection_score_and_song_means(self):
        metrics = {
            "macro_f1": 0.6,
        }
        _add_song_metric_means(metrics, [
            {
                "macro_seg_f1": 0.4,
                "boundary_f1_0_5s": 0.3,
                "boundary_f1_3s": 0.5,
            },
            {
                "macro_seg_f1": 0.8,
                "boundary_f1_0_5s": 0.5,
                "boundary_f1_3s": 0.7,
            },
        ])
        self.assertEqual(metrics["macro_seg_f1_mean"], 0.6)
        self.assertEqual(metrics["boundary_f1_3s_mean"], 0.6)
        score = checkpoint_selection_score(metrics, "coarse")
        self.assertAlmostEqual(score, 0.6)

    def test_run_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_directory(
                Path(directory),
                target_level="coarse",
                pool_mode="meanmaxstd",
                mert_layers=[4, 8, 12],
                seed=42,
                run_name="baseline",
            )
            second = create_run_directory(
                Path(directory),
                target_level="coarse",
                pool_mode="meanmaxstd",
                mert_layers=[4, 8, 12],
                seed=42,
                run_name="baseline",
            )
        self.assertEqual(first.name, "baseline")
        self.assertEqual(second.name, "baseline_2")


if __name__ == "__main__":
    unittest.main()
