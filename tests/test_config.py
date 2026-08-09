"""Config defaults, YAML overrides, and path resolution."""

from pathlib import Path

import pytest

from linkblink.config import Config


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "test.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_match_published_values():
    config = Config.load()
    assert config.camera.pixelsize_nm == 35.9
    assert config.unet.threshold == 0.5
    assert config.linking.search_range_px == 5.0
    assert config.judge.min_confidence == 0.50
    assert config.filtering.disabled is True


def test_yaml_overrides_only_named_keys(tmp_path):
    path = write_yaml(tmp_path, "camera:\n  pixelsize_nm: 100.0\n")
    config = Config.load(path)

    assert config.camera.pixelsize_nm == 100.0
    assert config.camera.frame_rate_hz == 100.0  # untouched default


def test_unknown_key_raises_rather_than_being_ignored(tmp_path):
    path = write_yaml(tmp_path, "camera:\n  pixelsize_typo: 1.0\n")
    with pytest.raises(ValueError, match="pixelsize_typo"):
        Config.load(path)


def test_unknown_section_raises(tmp_path):
    path = write_yaml(tmp_path, "nonexistent_section:\n  key: 1\n")
    with pytest.raises(ValueError, match="nonexistent_section"):
        Config.load(path)


def test_relative_paths_resolve_against_project_root(tmp_path):
    path = write_yaml(tmp_path, "paths:\n  input_dir: some/where\n")
    config = Config.load(path, project_root=tmp_path)
    assert config.paths.input_dir == (tmp_path / "some" / "where").resolve()


def test_absolute_paths_are_left_alone(tmp_path):
    absolute = (tmp_path / "elsewhere").resolve()
    path = write_yaml(tmp_path, f"paths:\n  input_dir: {absolute.as_posix()}\n")
    config = Config.load(path, project_root=tmp_path)
    assert config.paths.input_dir == absolute


def test_string_paths_become_path_objects(tmp_path):
    path = write_yaml(tmp_path, "unet:\n  model_path: checkpoints/other.pth\n")
    config = Config.load(path, project_root=tmp_path)
    assert isinstance(config.unet.model_path, Path)
    assert config.unet.model_path.name == "other.pth"
