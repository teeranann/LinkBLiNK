"""
Pairs generator (prefers judged CSVs).

Looks for, in each video subfolder:
  1) linked_particle_trajectories_judged.csv   (preferred)
  2) linked_particle_trajectories.csv          (fallback)

Builds positive/negative pairs for RF judge training.
"""

import json
from pathlib import Path
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================
# ======= USER SETTINGS ======
# ============================
CSV_DIR = r"D:\Works\AI_Project\Process\05_Judge\Brownian\results_pipeline_Final"    # parent folder with subfolders per video
OUTPUT_DIR = r"D:\Works\AI_Project\Process\05_Judge\Brownian\results_pipeline_Final"
NUM_POSITIVE = 6000
NUM_NEGATIVE = 6000
MIN_K = 5
MAX_K = 80
SEED = 42

# hard-negative spice (optional but recommended)
CLOSE_NEGATIVE_FRAC = 0.15      # 15% of negatives will be "close"
CLOSE_NEGATIVE_MAX_DIST = 6.0   # px
CLOSE_NEGATIVE_MAX_K = 15       # frames

# clip distance so one feature doesn't dominate (match pipeline inference)
POS_DIST_CLIP = 10.0
# ============================

REQUIRED_COLS = [
    "particle", "frame", "x", "y",
    "area", "Ibcnt", "fwhm_avg_pixels",
    "siamese_embedding"
]

def parse_embedding_string(s: str) -> np.ndarray:
    if not isinstance(s, str):
        return np.array([], dtype=np.float32)
    try:
        t = s.strip()
        if len(t) >= 2 and t[0] in "[(" and t[-1] in "])":
            t = t[1:-1]
        t = t.replace(",", " ")
        vals = [float(p) for p in t.split() if p]
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File '{path}' missing required column(s): {missing}")

def find_source_csvs(base_dir: Path) -> List[Path]:
    """
    For each immediate/recursive subfolder, pick judged if present, else raw.
    Returns a list of selected CSV paths (one per video folder).
    """
    # find all folders that contain either csv
    candidates = set()
    for p in base_dir.rglob("linked_particle_trajectories_judged.csv"):
        candidates.add(p.parent)
    for p in base_dir.rglob("linked_particle_trajectories.csv"):
        candidates.add(p.parent)

    chosen = []
    for folder in sorted(candidates):
        judged = folder / "linked_particle_trajectories_judged.csv"
        raw    = folder / "linked_particle_trajectories.csv"
        if judged.exists():
            chosen.append(judged)
        elif raw.exists():
            chosen.append(raw)
    return chosen

def euclidean(a: float, b: float) -> float:
    return float(abs(a - b))

def generate_pairs(
    df: pd.DataFrame,
    k_range: Tuple[int, int],
    num_pairs_to_generate: int,
    is_positive: bool,
    rng: random.Random
) -> List[dict]:
    pairs = []

    unique_particles = df["particle"].unique()
    if is_positive and len(unique_particles) < 1:
        return pairs
    if (not is_positive) and len(unique_particles) < 2:
        return pairs

    df_pf = df.set_index(["particle", "frame"], drop=False)

    min_k, max_k = k_range
    if min_k > max_k:
        min_k, max_k = max_k, min_k

    max_retries = max(1000, num_pairs_to_generate * 10)
    retries = 0

    while len(pairs) < num_pairs_to_generate and retries < max_retries:
        retries += 1

        pid1 = rng.choice(unique_particles)
        traj1 = df[df["particle"] == pid1]
        if len(traj1) < 2:
            continue

        k = rng.randint(min_k, max_k)
        start_frame = int(traj1["frame"].min())
        end_frame = int(traj1["frame"].max())
        t_max = end_frame - k
        if start_frame >= t_max:
            continue

        t = rng.randint(start_frame, t_max)

        try:
            lost_row = df_pf.loc[(pid1, t)]
            if isinstance(lost_row, pd.DataFrame):
                lost_row = lost_row.iloc[0]
        except KeyError:
            continue

        if is_positive:
            pid2 = pid1
        else:
            # NEGATIVE: prefer a hard-close negative sometimes
            others = [p for p in unique_particles if p != pid1]
            if not others:
                continue

            try_close = rng.random() < CLOSE_NEGATIVE_FRAC
            if try_close:
                k_local = rng.randint(min(min_k, CLOSE_NEGATIVE_MAX_K),
                                      min(max_k, CLOSE_NEGATIVE_MAX_K))
                cand: List[Tuple[int, pd.Series, int, float]] = []
                for pid2_try in others:
                    try:
                        new_row_try = df_pf.loc[(pid2_try, t + k_local)]
                        if isinstance(new_row_try, pd.DataFrame):
                            new_row_try = new_row_try.iloc[0]
                    except KeyError:
                        continue
                    d = float(np.hypot(
                        float(lost_row['x']) - float(new_row_try['x']),
                        float(lost_row['y']) - float(new_row_try['y'])
                    ))
                    if d <= CLOSE_NEGATIVE_MAX_DIST:
                        cand.append((pid2_try, new_row_try, k_local, d))
                if cand:
                    cand.sort(key=lambda z: z[3])
                    pid2, new_row, k, _ = cand[0]
                else:
                    pid2 = rng.choice(others)
                    try:
                        new_row = df_pf.loc[(pid2, t + k)]
                        if isinstance(new_row, pd.DataFrame):
                            new_row = new_row.iloc[0]
                    except KeyError:
                        continue
            else:
                pid2 = rng.choice(others)
                try:
                    new_row = df_pf.loc[(pid2, t + k)]
                    if isinstance(new_row, pd.DataFrame):
                        new_row = new_row.iloc[0]
                except KeyError:
                    continue

        if is_positive:
            try:
                new_row = df_pf.loc[(pid2, t + k)]
                if isinstance(new_row, pd.DataFrame):
                    new_row = new_row.iloc[0]
            except KeyError:
                continue

        emb1 = parse_embedding_string(lost_row["siamese_embedding"])
        emb2 = parse_embedding_string(new_row["siamese_embedding"])
        if emb1.size == 0 or emb2.size == 0 or emb1.shape != emb2.shape:
            continue

        siamese_dist = float(np.linalg.norm(emb1 - emb2))
        pos_dist = float(np.hypot(lost_row["x"] - new_row["x"], lost_row["y"] - new_row["y"]))
        pos_dist = min(pos_dist, POS_DIST_CLIP)  # clip

        time_diff = int(new_row["frame"] - lost_row["frame"])
        area_diff = euclidean(lost_row["area"], new_row["area"])
        ibcnt_diff = euclidean(lost_row["Ibcnt"], new_row["Ibcnt"])
        fwhm_diff = euclidean(lost_row["fwhm_avg_pixels"], new_row["fwhm_avg_pixels"])

        def safe_ratio(a, b):
            b = float(b) if float(b) != 0 else 1e-6
            return float(a) / b

        area_ratio = safe_ratio(lost_row["area"], new_row["area"])
        ibcnt_ratio = safe_ratio(lost_row["Ibcnt"], new_row["Ibcnt"])
        fwhm_ratio = safe_ratio(lost_row["fwhm_avg_pixels"], new_row["fwhm_avg_pixels"])

        pairs.append({
            "siamese_euclidean_distance": siamese_dist,
            "position_euclidean_distance": pos_dist,
            "time_difference": time_diff,
            "area_difference": area_diff,
            "Ibcnt_difference": ibcnt_diff,
            "fwhm_avg_difference": fwhm_diff,
            "area_ratio": area_ratio,
            "Ibcnt_ratio": ibcnt_ratio,
            "fwhm_avg_ratio": fwhm_ratio,
            "label": 1 if is_positive else 0,
            "video_id": str(df.attrs.get("video_id", "")),
        })

    return pairs

def main():
    rng = random.Random(SEED)
    outdir = Path(OUTPUT_DIR); outdir.mkdir(parents=True, exist_ok=True)

    csv_files = find_source_csvs(Path(CSV_DIR))
    if not csv_files:
        print(f"[!] No trajectories CSVs found under {CSV_DIR}")
        return

    # Load + validate
    dataframes: List[Tuple[Path, pd.DataFrame]] = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            ensure_required_columns(df, f)
            df = df.dropna(subset=["siamese_embedding"]).copy()
            if len(df) == 0:
                continue
            # attach video_id as an attribute so generate_pairs can include it
            df.attrs["video_id"] = f.parent.name
            dataframes.append((f, df))
            print(f"[+] Using: {f.name} in {f.parent.name}")
        except Exception as e:
            print(f"[!] Skipping '{f}': {e}")

    if not dataframes:
        print("[!] No usable CSVs after validation.")
        return

    pos_per = max(1, NUM_POSITIVE // len(dataframes))
    neg_per = max(1, NUM_NEGATIVE // len(dataframes))

    all_pairs = []
    print("\n--- Generating positive pairs ---")
    for f, df in tqdm(dataframes, desc="Positive"):
        all_pairs.extend(generate_pairs(df, (MIN_K, MAX_K), pos_per, True, rng))
    pos_count = len(all_pairs)
    print(f"[✓] Generated positive pairs: {pos_count}")

    print("\n--- Generating negative pairs ---")
    for f, df in tqdm(dataframes, desc="Negative"):
        all_pairs.extend(generate_pairs(df, (MIN_K, MAX_K), neg_per, False, rng))
    neg_count = len(all_pairs) - pos_count
    print(f"[✓] Generated negative pairs: {neg_count}")

    if not all_pairs:
        print("[!] No pairs generated.")
        return

    all_df = pd.DataFrame(all_pairs).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # stratified split (label) but keep video_id for later grouped evaluation if you want
    def stratified_split(df: pd.DataFrame, train=0.7, val=0.15, test=0.15, seed=42):
        rs = np.random.RandomState(seed)
        labels = df["label"].values
        unique = np.unique(labels)
        if len(unique) < 2:
            df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(df_shuf)
            i1 = int(train * n); i2 = int((train + val) * n)
            return df_shuf.iloc[:i1], df_shuf.iloc[i1:i2], df_shuf.iloc[i2:]
        train_parts, val_parts, test_parts = [], [], []
        for y in unique:
            sub = df[df["label"] == y].sample(frac=1, random_state=seed)
            n = len(sub); i1 = int(train * n); i2 = int((train + val) * n)
            train_parts.append(sub.iloc[:i1]); val_parts.append(sub.iloc[i1:i2]); test_parts.append(sub.iloc[i2:])
        return (
            pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
            pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
            pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
        )

    train_df, val_df, test_df = stratified_split(all_df, seed=SEED)

    (outdir / "train.csv").write_text(train_df.to_csv(index=False))
    (outdir / "val.csv").write_text(val_df.to_csv(index=False))
    (outdir / "test.csv").write_text(test_df.to_csv(index=False))

    meta = {
        "csv_dir": str(CSV_DIR),
        "output_dir": str(outdir),
        "num_positive_requested": NUM_POSITIVE,
        "num_negative_requested": NUM_NEGATIVE,
        "num_positive_generated": int((train_df["label"]==1).sum() + (val_df["label"]==1).sum() + (test_df["label"]==1).sum()),
        "num_negative_generated": int((train_df["label"]==0).sum() + (val_df["label"]==0).sum() + (test_df["label"]==0).sum()),
        "min_k": MIN_K, "max_k": MAX_K, "seed": SEED,
        "pos_dist_clip": POS_DIST_CLIP,
        "close_negative_frac": CLOSE_NEGATIVE_FRAC,
        "close_negative_max_dist": CLOSE_NEGATIVE_MAX_DIST,
        "close_negative_max_k": CLOSE_NEGATIVE_MAX_K,
        "source_csv_count": len(dataframes)
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n[Done]")
    print(f"  Output folder : {outdir}")
    print(f"  Train size    : {len(train_df)}")
    print(f"  Val size      : {len(val_df)}")
    print(f"  Test size     : {len(test_df)}")
    print(f"  Positives     : {meta['num_positive_generated']}")
    print(f"  Negatives     : {meta['num_negative_generated']}")

if __name__ == "__main__":
    main()
