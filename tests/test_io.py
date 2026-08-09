"""Frame naming, mask-cache detection, and DataFrame hygiene."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from linkblink.config import MaskCacheConfig
from linkblink.io.images import (
    list_frames,
    load_normalised_frame,
    mask_name_for,
    parse_frame_number,
)
from linkblink.io.paths import find_video_dirs, mask_cache_is_usable
from linkblink.utils.dataframe import ensure_frame_is_column, normalise_track_frame


class TestFrameNaming:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Real_1_frame_0042.tif", 42),
            ("frame_0000.tif", 0),
            ("video_frame_1.tif", 1),
            ("noNumbersHere.tif", 0),
        ],
    )
    def test_parse_frame_number(self, name, expected):
        assert parse_frame_number(Path(name)) == expected

    def test_mask_name_follows_the_convention(self):
        assert mask_name_for(Path("Real_1_frame_0007.tif")) == "Real_1_frame_0007_predict_mask.png"


class TestImageLoading:
    def test_8bit_image_normalises_by_255(self, tmp_path):
        path = tmp_path / "f.png"
        Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(path)
        assert load_normalised_frame(path) == pytest.approx(np.ones((4, 4)))

    def test_16bit_tiff_normalises_by_65535(self, tmp_path):
        path = tmp_path / "f.tif"
        Image.fromarray(np.full((4, 4), 65535, dtype=np.uint16)).save(path)
        loaded = load_normalised_frame(path)
        assert loaded.dtype == np.float32
        assert loaded == pytest.approx(np.ones((4, 4)))

    def test_list_frames_is_sorted_and_filtered(self, tmp_path):
        for name in ["b.tif", "a.tif", "notes.txt"]:
            (tmp_path / name).write_bytes(b"x")
        assert [p.name for p in list_frames(tmp_path)] == ["a.tif", "b.tif"]


class TestMaskCache:
    def make_video(self, tmp_path, n_frames, n_masks):
        frames = tmp_path / "frames"
        masks = tmp_path / "masks"
        frames.mkdir()
        masks.mkdir()
        for i in range(n_frames):
            (frames / f"frame_{i:04d}.tif").write_bytes(b"x")
        for i in range(n_masks):
            (masks / f"frame_{i:04d}_predict_mask.png").write_bytes(b"x")
        return frames, masks

    def test_complete_cache_is_usable(self, tmp_path):
        frames, masks = self.make_video(tmp_path, 5, 5)
        assert mask_cache_is_usable(frames, masks, MaskCacheConfig()) is True

    def test_partial_cache_is_rejected(self, tmp_path):
        frames, masks = self.make_video(tmp_path, 5, 3)
        assert mask_cache_is_usable(frames, masks, MaskCacheConfig()) is False

    def test_partial_cache_accepted_when_full_match_not_required(self, tmp_path):
        frames, masks = self.make_video(tmp_path, 5, 3)
        config = MaskCacheConfig(require_full_match=False)
        assert mask_cache_is_usable(frames, masks, config) is True

    def test_missing_directory_is_not_usable(self, tmp_path):
        frames, _ = self.make_video(tmp_path, 3, 0)
        assert mask_cache_is_usable(frames, tmp_path / "absent", MaskCacheConfig()) is False


class TestFindVideoDirs:
    def test_only_folders_containing_tiffs_are_returned(self, tmp_path):
        (tmp_path / "good").mkdir()
        (tmp_path / "good" / "a.tif").write_bytes(b"x")
        (tmp_path / "empty").mkdir()
        (tmp_path / "notes.txt").write_bytes(b"x")

        assert [d.name for d in find_video_dirs(tmp_path)] == ["good"]

    def test_missing_input_dir_gives_empty_list(self, tmp_path):
        assert find_video_dirs(tmp_path / "absent") == []


class TestDataFrameHygiene:
    def test_frame_is_promoted_off_the_index(self):
        df = pd.DataFrame({"frame": [1, 2], "x": [0.0, 1.0]}).set_index("frame")
        out = ensure_frame_is_column(df)
        assert "frame" in out.columns
        assert out["frame"].tolist() == [1, 2]

    def test_duplicate_frame_column_is_deduplicated(self):
        df = pd.DataFrame({"frame": [1, 2], "x": [0.0, 1.0]})
        df = pd.concat([df, df["frame"]], axis=1)  # frame appears twice
        out = normalise_track_frame(df)
        assert (out.columns == "frame").sum() == 1

    def test_frame_and_particle_are_coerced_to_int(self):
        df = pd.DataFrame({"frame": ["1", "2"], "particle": [3.0, 4.0], "x": [0.0, 1.0]})
        out = normalise_track_frame(df)
        assert out["frame"].tolist() == [1, 2]
        assert out["particle"].tolist() == [3, 4]
