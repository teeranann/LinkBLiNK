"""Judge feature construction and the disabled/missing-model paths."""

import numpy as np
import pandas as pd
import pytest

from linkblink.config import JudgeConfig
from linkblink.linking import apply_judge, build_segments, parse_embedding, rf_features


class TestParseEmbedding:
    def test_passes_through_arrays(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        assert parse_embedding(arr).tolist() == [1.0, 2.0]

    def test_parses_the_csv_string_form(self):
        assert parse_embedding("[0.5 1.5 2.5]").tolist() == [0.5, 1.5, 2.5]

    def test_parses_comma_separated(self):
        assert parse_embedding("(0.5, 1.5)").tolist() == [0.5, 1.5]

    def test_missing_values_give_an_empty_array(self):
        assert parse_embedding(None).size == 0
        assert parse_embedding(float("nan")).size == 0

    def test_garbage_gives_an_empty_array(self):
        assert parse_embedding("not numbers").size == 0


class TestRFFeatures:
    def make_row(self, **kwargs):
        base = {
            "x": 0.0, "y": 0.0, "frame": 0, "area": 20,
            "Ibcnt": 100.0, "fwhm_avg_pixels": 3.0,
            "siamese_embedding": np.array([1.0, 0.0], dtype=np.float32),
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_distances_and_gap(self):
        end = self.make_row(x=0.0, y=0.0, frame=10)
        start = self.make_row(x=3.0, y=4.0, frame=15)

        features = rf_features(end, start)
        assert features["position_euclidean_distance"] == pytest.approx(5.0)
        assert features["time_difference"] == 5
        assert features["siamese_euclidean_distance"] == pytest.approx(0.0)

    def test_missing_embedding_gives_infinite_distance(self):
        end = self.make_row(siamese_embedding=None)
        start = self.make_row()
        assert rf_features(end, start)["siamese_euclidean_distance"] == float("inf")

    def test_mismatched_embedding_length_gives_infinite_distance(self):
        end = self.make_row(siamese_embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        start = self.make_row()
        assert rf_features(end, start)["siamese_euclidean_distance"] == float("inf")

    def test_ratios_and_differences(self):
        end = self.make_row(area=20, Ibcnt=100.0)
        start = self.make_row(area=10, Ibcnt=50.0)

        features = rf_features(end, start)
        assert features["area_difference"] == pytest.approx(10.0)
        assert features["area_ratio"] == pytest.approx(2.0)
        assert features["Ibcnt_ratio"] == pytest.approx(2.0)

    def test_zero_denominator_does_not_raise(self):
        end = self.make_row(area=5)
        start = self.make_row(area=0)
        assert np.isfinite(rf_features(end, start)["area_ratio"])


class TestSegments:
    def test_endpoints_are_taken_in_frame_order(self):
        df = pd.DataFrame({
            "frame": [3, 1, 2],
            "x": [30.0, 10.0, 20.0],
            "y": [0.0, 0.0, 0.0],
            "particle": [0, 0, 0],
        })
        (segment,) = build_segments(df)

        assert (segment.start_frame, segment.end_frame) == (1, 3)
        assert segment.first_row["x"] == 10.0
        assert segment.last_row["x"] == 30.0


class TestApplyJudge:
    def frame(self):
        return pd.DataFrame({
            "frame": [1, 2, 10, 11],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "particle": [0, 0, 1, 1],
        })

    def test_disabled_judge_returns_input_unchanged(self):
        judged, log, count = apply_judge(self.frame(), JudgeConfig(enabled=False))
        assert count == 0
        assert log.empty
        assert judged["particle"].tolist() == [0, 0, 1, 1]

    def test_missing_model_is_a_skip_not_a_crash(self, tmp_path):
        config = JudgeConfig(enabled=True, model_path=tmp_path / "absent.pkl")
        judged, log, count = apply_judge(self.frame(), config)
        assert count == 0
        assert log.empty
        assert len(judged) == 4
