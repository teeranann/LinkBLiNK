# LinkBLiNK Tracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20806664.svg)](https://doi.org/10.5281/zenodo.20806664)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Linking Blinking Localizations in Nanoscopic Kinetics Tracker** — a
machine-learning-assisted particle tracking pipeline for single-molecule
localization microscopy (SMLM).

LinkBLiNK Tracker addresses two failure modes that limit classical
single-particle trackers when fluorophores blink or temporarily defocus near
other particles: **trajectory fragmentation** (the gap is left open) and
**false linkage** (two distinct particles are merged into one). The pipeline
couples proximity-based linking with three machine-learning modules:

1. **U-Net** segments each frame into per-particle binary masks.
2. **Siamese network** maps each detection into a 128-D appearance embedding
   so that particles of similar appearance lie close together.
3. **Random Forest "Judge"** decides whether two trajectory fragments across a
   temporal gap belong to the same particle, using spatial, temporal,
   photometric, and embedding features.

> **Versions.** This branch is **v2.0.0-dev**, a modular reimplementation of
> the pipeline as the `linkblink` Python package. All results in the
> manuscript were produced with the v1.0.x pipeline, which remains permanently
> archived: code as **v1.0.2**
> ([10.5281/zenodo.20996425](https://doi.org/10.5281/zenodo.20996425)) and the
> identical code plus the full example datasets as **v1.0.3**
> ([10.5281/zenodo.21863739](https://doi.org/10.5281/zenodo.21863739)).
> The v2 pipeline is verified to reproduce v1.0.x output exactly — see
> [Verification](#verification). The original v1.0.x scripts are kept
> unmodified in [`legacy/`](legacy/).

> **Status.** Code released alongside a manuscript in preparation for
> journal submission. Pretrained model weights are bundled in
> `checkpoints/`. The software is registered as a copyrighted work in Thailand
> under Application No. 465770 (Burapha University). Source code is released
> under the MIT License (see [License](#license)).

---

## Quick start

```bash
pip install -r requirements.txt
python run.py --config configs/default.yaml
```

That processes every video folder under `data/input/` (the shipped `demo/`
folder has 100 frames) and writes to `outputs/results/<video>/`.

```bash
python run.py                              # batch, built-in defaults
python run.py --single                     # pick one video via a file dialog
python run.py --single --input path/to/video.tif
python run.py --input-dir data/other --no-judge
python run.py --evaluate                   # score against data/ground_truth/
python run.py --help
```

As a library:

```python
from linkblink import Config, Pipeline

config = Config.load("configs/default.yaml")
config.judge.min_confidence = 0.7
Pipeline(config).run_batch()
```

Individual stages are importable on their own:

```python
from linkblink.detection import UNetDetector
from linkblink.features import extract_frame_particles
from linkblink.linking import apply_judge, link_nearest_neighbour
from linkblink.analysis import calculate_msd
```

---

## The pipeline

```
frames ──[U-Net]──> masks ──[measure + filter]──> detections
       ──[nearest-neighbour]──> tracks ──[RF judge]──> trajectories
       ──[MSD / plots / evaluation]
```

| Stage | Module | What it does |
|---|---|---|
| 1 (optional) | `io/seq.py` | `.seq` → TIFF via MATLAB. Skipped for TIFF input. |
| 2 | `detection/predict.py` | U-Net inference, one binary mask per frame. |
| 3 | `features/` | Per-particle shape, focus, Gaussian PSF fit, photometry. |
| 3b | `filtering/defocus.py` | Rejects defocused / malformed detections. |
| 3c | `embedding/` | Siamese appearance vector per surviving detection. |
| 4a | `linking/nearest_neighbor.py` | Frame-to-frame linking. No gap bridging. |
| 4b | `linking/judge.py` | Random forest merges tracks split by blinking. |
| 5 | `analysis/`, `viz/` | MSD, diffusion fits, figures, ground-truth scoring. |

### Layout

```
run.py                     entry point
configs/default.yaml       all tunable parameters
checkpoints/               R2G3B3.pth (U-Net), S1.pth (Siamese), F5.pkl (RF)
data/input/<video>/        TIFF frames, one folder per video
data/ground_truth/         optional <video>_ground_truth.csv
examples/                  full example videos + reference results (below)
matlab_scripts/            .seq readers
outputs/                   masks and results (git-ignored, regenerated)
tests/                     unit tests

linkblink/                 the pipeline package
  config.py                typed dataclasses + YAML loading
  cli.py                   argument parsing
  pipeline.py              stage wiring — the only orchestration code
  artifacts.py             output filenames (single source of truth)
  io/ models/ detection/ features/ filtering/ embedding/ linking/
  analysis/ viz/ utils/

simulator/                 Scenario A and B benchmark-video synthesis
preprocessing/             training-data preparation (masks, filters, pairs)
training/                  U-Net / Siamese / Random Forest training scripts
benchmark/                 TrackMate baseline scripts (Fiji/Jython)
legacy/                    the original v1.0.x scripts, unmodified
```

### Outputs

Written to `outputs/results/<video>/`:

| File | Contents |
|---|---|
| `filtered_particle_data_for_tracking.csv` | Per-detection measurements, pre-linking |
| `linked_particle_trajectories_raw.csv` | Trajectories before the judge |
| `linked_particle_trajectories_judged.csv` | **Primary result** — after re-identification |
| `judge_merge_log.csv` | One row per merge, with confidence and features |
| `msd_results.csv` | MSD vs lag, per particle |
| `evaluation_metrics.csv` | Only with `--evaluate` |
| `*.png` | MSD curves, photon decay, trajectory overview |

Reference outputs for the shipped demo video are committed under
`examples/reference_results/`, so a fresh install can be checked against the
published numbers without a GPU run.

---

## Configuration

Everything lives in `configs/default.yaml`; anything omitted falls back to the
defaults in `linkblink/config.py`. Relative paths resolve against the project
root, so a config file is portable between machines. Unknown keys raise an
error instead of being silently ignored.

Values that matter most:

- `unet.threshold` (0.5) — detection sensitivity.
- `unet.norm_mean` / `norm_std` — **must** match the values `R2G3B3.pth` was
  trained with. Changing them invalidates the checkpoint.
- `filtering.disabled` (true) — when true, every detection is kept, but all the
  filter criteria are still measured into the CSV, so you can re-threshold
  offline without re-running detection.
- `linking.search_range_px` (5.0) — max centroid movement between frames.
- `judge.min_confidence` (0.50) — how sure the forest must be before merging.
- `camera.*` — detector constants; wrong values give wrong photon numbers.

### `.seq` (StreamPix) input

The pipeline reads 16-bit TIFF sequences directly. Raw StreamPix `.seq`
recordings are converted to TIFF first via a bundled MATLAB script
(`matlab_scripts/seq_to_tif.m`, MATLAB R2021b or newer): set
`matlab.exe_path` in `configs/default.yaml` to your MATLAB executable
(default is the Windows R2021b path — edit for your machine), then run with
`python run.py --single --input path/to/video.seq`. TIFF input never invokes
MATLAB.

---

## Verification

The v2 pipeline was checked against the archived v1.0.x pipeline
([v1.0.2, 10.5281/zenodo.20996425](https://doi.org/10.5281/zenodo.20996425))
on an identical 100-frame input (the shipped demo video), on **both** Python
environments. Within each environment v2 reproduces v1.0.x exactly:

| Output | Result |
|---|---|
| 100 U-Net mask PNGs | **byte-identical** |
| 100 filtered mask PNGs | **byte-identical** |
| `filtered_particle_data_for_tracking.csv` | 112 rows × 28 cols, all columns bit-identical except one (below) |
| `linked_particle_trajectories_raw.csv` | 100 rows × 29 cols, identical |
| `linked_particle_trajectories_judged.csv` | 100 rows × 29 cols, identical |
| `msd_results.csv` | 30 rows, **exact** to 0.0e+00 |
| Diffusion fits (D, V, Z) | identical to all printed digits |

**The one difference** is text precision in the `laplacian_variance` column:
v1.0.x wrote a NumPy `float32` at ~7 significant digits; v2 writes the same
value at full Python-float precision. `float32(old) == float32(new)` holds for
every row — the computation is identical, only the recorded precision changed.

### Python 3.12 vs 3.14 — results differ, and it is not the rewrite

The paper's results were produced on Python 3.12.10 (`torch 2.7.1+cu126`).
Running *both* codebases on *both* interpreters gives four runs, and the mask
outputs group by **interpreter, not by code**: v1.0.x and v2 are identical to
each other on 3.12, and identical to each other on 3.14, but the two
torch/cuDNN versions produce slightly different convolution outputs, which
shifts pixels across the 0.5 sigmoid threshold at mask boundaries.

Downstream effect of 3.12 → 3.14, measured on the demo video: detection and
trajectory counts unchanged; centroids shift ≤ 0.17 px; per-particle
photometry (`Ibcnt`, photons) changes by up to 36 % on individual detections;
MSD ≤ 0.46 % relative. **To reproduce the published numbers, use the Python
3.12 environment.** Python 3.14 is a fine baseline for new work — re-derive
reference values rather than comparing against 3.12-era outputs.

Run the unit tests with:

```bash
python -m pytest tests -q      # 50 tests, no GPU or checkpoints needed
```

---

## Reproducing the paper benchmarks

1. Generate simulated videos with the config-driven scripts in `simulator/`
   (`VideoSynthesis_ScenarioA_withGTMask.py`,
   `VideoSynthesis_ScenarioB_withGTMask.py`). Set the output/background paths
   and `NUM_VIDEOS_TO_GENERATE` (100 in the paper) at the top of each script.
   Each difficulty level is one group of constants — e.g.
   `SCENARIO_A_CLOSE_PASS_DISTANCE_PIXELS` for Scenario A (A1–A7), the decoy
   intensity/PSF/ellipticity for Scenario B (B1–B8) and the BI/BP/BE
   decomposition, and `SCENARIO_B_GAP_LENGTH_RANGE` for the BX duration
   series. The exact values for every level are tabulated in Supplementary
   Tables S2–S5 of the manuscript. Acquisition constants already match the
   paper (`PIXEL_SIZE_UM = 0.0359`, `NUM_FRAMES = 500`, `BIT_DEPTH = 16`).
2. Run the baseline trackers (Crocker-Grier via TrackPy, Simple LAP and full
   LAP via Fiji TrackMate) on each video. Detection and tracking templates are
   in `benchmark/`; these run inside the Fiji Jython console.
3. Run LinkBLiNK on the same videos (`data/input/` + `python run.py`).
4. Aggregate the per-condition metrics (fragmentation rate, false linkage
   rate, completeness, association precision/recall/F1) and compare.

Full example videos (both real 1000-frame recordings and one full 500-frame
video per scenario, with ground truth) ship in `examples/` — see
[`examples/README.md`](examples/README.md).

## Data preparation

The scripts in `preprocessing/` turn raw frames and trajectories into the
exact inputs used to train each module. Paths are set in the configuration
block at the top of each script.

| Script | Role |
|--------|------|
| `ParticleMaskGeneration.m` | Interactive labelling: click particles and fit a 2-D Gaussian to generate the binary U-Net training masks (MATLAB; Optimization Toolbox required). |
| `DefocusingFilter.py` | Removes out-of-focus detections from the U-Net masks by Gaussian-fit quality. |
| `SiameseDataPrep_Batch.py` | Crops particle patches and builds the same/different pairs used to train the Siamese appearance embedding. |
| `RandomForestGen.py` | Builds the positive/negative fragment pairs used to train the Random Forest Judge. |
| `id_corrector.py` | Aligns predicted trajectory IDs to the ground truth (Hungarian assignment) for evaluation. |

## Retraining

- **U-Net.** `training/unet/train.py` retrains the segmentation model.
- **Siamese network.** `training/siamese/SiameseTrain.py` retrains the
  appearance embedding from real videos of stationary particles.
- **Random Forest Judge.** `training/random_forest/RandomForestTrain.py`
  retrains the Judge from fragment pairs labelled by ground truth.

These scripts are shipped as used for the paper: they were written against the
v1.0.x workspace layout and import the modules now kept in `legacy/`
(`data_loading.py`, etc.), so run them with `legacy/` on `PYTHONPATH` and
adjust the data paths at the top of each script. Install
`requirements-dev.txt` first.

---

## Requirements

Python 3.12 is the published-results environment (see
[Verification](#verification)); the code also runs on Python 3.14. A
CUDA-capable GPU is used when visible, otherwise CPU.

See `requirements.txt`. `scikit-learn` is required even though nothing imports
it directly — the judge is a pickled sklearn estimator (`F5.pkl`, pickled with
scikit-learn 1.7.0; pin `scikit-learn==1.7.*` to match the published
environment exactly). MATLAB R2021b is only needed for `.seq` input.

## Citation

If you use LinkBLiNK Tracker in academic work, please cite the manuscript
(in preparation). Until publication, cite the
software via the `CITATION.cff` file (GitHub's "Cite this repository" widget)
or the Zenodo concept DOI
[10.5281/zenodo.20806664](https://doi.org/10.5281/zenodo.20806664).

## License

The source code in this repository is released under the MIT License (see
`LICENSE`). The pretrained model weights in `checkpoints/` are distributed
under the same terms.

The underlying software, registered as *"Software for Tracking and Linking
Blinking Fluorescent Particle Positions in Nanoscale Kinetics (LinkBLiNK
Tracker)"*, is a registered copyrighted work in Thailand under **Copyright
Application No. 465770** (filed 19 December 2025 with the Department of
Intellectual Property), with **Burapha University** as the copyright holder.

- **Inventors:** Teeranan Nongnual, Kanoksak Saelee
- **Co-inventors:** Papichaya Pooldee, Sitti Buathong, Supranee Kaewpirom

## Authors

Department of Chemistry and Department of Physics, Faculty of Science,
Burapha University, Chonburi 20131, Thailand.

- Kanoksak Saelee
- Papichaya Pooldee
- Sitti Buathong
- Supranee Kaewpirom
- Teeranan Nongnual ([teeranan.no@buu.ac.th](mailto:teeranan.no@buu.ac.th)) — corresponding author
