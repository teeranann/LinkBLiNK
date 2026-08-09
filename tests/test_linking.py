"""Nearest-neighbour linking and stub removal."""

import numpy as np
import pandas as pd
import pytest

from linkblink.linking import drop_short_trajectories, link_nearest_neighbour


def detections(rows):
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def test_single_particle_moving_slowly_keeps_one_id():
    df = detections([(1, 10.0, 10.0), (2, 11.0, 10.0), (3, 12.0, 10.0)])
    linked = link_nearest_neighbour(df, max_displacement_px=5.0)
    assert linked["particle"].nunique() == 1


def test_jump_beyond_search_range_starts_a_new_track():
    df = detections([(1, 10.0, 10.0), (2, 50.0, 50.0)])
    linked = link_nearest_neighbour(df, max_displacement_px=5.0)
    assert linked["particle"].nunique() == 2


def test_two_particles_stay_separate():
    df = detections([
        (1, 10.0, 10.0), (1, 40.0, 40.0),
        (2, 11.0, 10.0), (2, 41.0, 40.0),
    ])
    linked = link_nearest_neighbour(df, max_displacement_px=5.0)
    assert linked["particle"].nunique() == 2
    # Each track spans both frames.
    assert (linked.groupby("particle").size() == 2).all()


def test_closest_candidate_wins_when_two_compete():
    # The frame-2 detection at x=10.5 is nearest the track from x=10.
    df = detections([(1, 10.0, 10.0), (2, 10.5, 10.0), (2, 13.0, 10.0)])
    linked = link_nearest_neighbour(df, max_displacement_px=5.0)

    track_of_first = linked.loc[(linked["frame"] == 1), "particle"].iloc[0]
    continued = linked[(linked["frame"] == 2) & (linked["particle"] == track_of_first)]
    assert continued["x"].iloc[0] == pytest.approx(10.5)


def test_missing_intermediate_frame_breaks_the_track():
    # No frame 2, so the frame-3 detection cannot continue the frame-1 track.
    df = detections([(1, 10.0, 10.0), (3, 10.5, 10.0)])
    linked = link_nearest_neighbour(df, max_displacement_px=5.0)
    assert linked["particle"].nunique() == 2


def test_empty_input_is_returned_untouched():
    df = detections([])
    assert link_nearest_neighbour(df, 5.0).empty


def test_drop_short_trajectories_removes_stubs():
    df = pd.DataFrame({
        "frame": [1, 2, 3, 1, 2],
        "x": np.zeros(5),
        "y": np.zeros(5),
        "particle": [0, 0, 0, 1, 1],
    })
    kept = drop_short_trajectories(df, min_length=3)
    assert set(kept["particle"]) == {0}
