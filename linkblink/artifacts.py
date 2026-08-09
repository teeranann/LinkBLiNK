"""Names of the files each run writes.

Centralised because these names are the pipeline's public contract: published
result folders and any downstream tool read them by name, so they must not
drift between the writer and the readers.
"""

FILTERED_PARTICLES_CSV = "filtered_particle_data_for_tracking.csv"
"""Per-detection measurements, before linking."""

LINKED_RAW_CSV = "linked_particle_trajectories_raw.csv"
"""Trajectories from nearest-neighbour linking, before the judge."""

LINKED_JUDGED_CSV = "linked_particle_trajectories_judged.csv"
"""Trajectories after re-identification merges. The primary result."""

JUDGE_MERGE_LOG_CSV = "judge_merge_log.csv"
"""One row per merge the judge performed, with its confidence and features."""

MSD_RESULTS_CSV = "msd_results.csv"
EVALUATION_METRICS_CSV = "evaluation_metrics.csv"

MASK_SUFFIX = "_predict_mask.png"
FILTERED_MASK_SUFFIX = "_filtered_mask.png"
