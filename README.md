# LinkBLiNK Tracker

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20806665.svg)](https://doi.org/10.5281/zenodo.20806665)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Linking Blinking Localizations in Nanoscopic Kinetics Tracker**, a
machine-learning-assisted particle tracking pipeline for single-molecule
localization microscopy (SMLM).

LinkBLiNK Tracker addresses two failure modes that limit classical
single-particle trackers when fluorophores blink or temporarily defocus
near other particles: **trajectory fragmentation** (the gap is left
open) and **false linkage** (two distinct particles are merged into
one). The pipeline couples proximity-based linking with three
machine-learning modules:

1. **U-Net** segments each frame into per-particle binary masks.
2. **Siamese network** maps each detection into a 128-D appearance
   embedding so that particles of similar appearance lie close together.
3. **Random Forest "Judge"** decides whether two trajectory fragments
   across a temporal gap belong to the same particle, using spatial,
   temporal, photometric, and embedding features.

Across controlled benchmarks against the Crocker-Grier, Simple LAP,
and full LAP trackers, the Judge reduced the fragmentation rate to
1.5%, ten times lower than the 14.5% to 47.0% reached by the
baselines, and held zero false linkage at moderate impostor difficulty
where the baselines reached 12% to 49%.

> **Status.** Code released alongside a manuscript currently in
> preparation. Pretrained model weights are bundled in
> `checkpoints/`. The software is registered as a copyrighted work in
> Thailand on 19 December 2025 under Application No. 465770
> (Burapha University; authors: Kanoksak Saelee and Teeranan
> Nongnual). Source code is released under the MIT License (see
> `LICENSE`).

## Installation

Python 3.12 is recommended (the pipeline was developed on 3.12.10).
A CUDA-capable GPU is recommended for inference but not required.

```bash
git clone https://github.com/teeranann/LinkBLiNK.git
cd LinkBLiNK
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

To retrain the deep-learning modules or to regenerate simulated
benchmark videos, install the development extras as well:

```bash
pip install -r requirements-dev.txt
```

## Quick start on the bundled example

The repository ships with a small real F8BT example in
`examples/real/`. To run the full pipeline on it:

1. Copy or symlink an example video folder into `input_files/` at the
   repository root (the pipeline expects `<repo>/input_files/<video>/`
   containing a TIFF sequence):

   ```bash
   mkdir input_files
   cp -r "examples/real/Real Particle 1" input_files/
   ```

2. Run the main pipeline:

   ```bash
   python LinkBLiNK.py
   ```

Outputs are written to:

- `unet_masks/<video>/` — U-Net binary masks per frame.
- `filtered_masks/<video>/` — masks after the defocusing filter.
- `results/<video>/` — linked trajectories, MSD, and diagnostic plots.

The `batch_mode_enabled` flag in `LinkBLiNK.py` controls whether the
pipeline iterates over every subfolder of `input_files/` or opens a
file-dialog GUI.

## Repository layout

```
LinkBLiNK/
├── LinkBLiNK.py             main pipeline
├── Illustration.py          plotting and MSD diffusion fits
├── SiameseNet.py            Siamese network architecture
├── unet_model.py            U-Net (reduced-depth) architecture
├── unet_parts.py            U-Net building blocks
├── data_loading.py          PyTorch dataset and image I/O
├── checkpoints/             pretrained weights
│   ├── R2G3B3.pth           U-Net weights
│   ├── S1.pth               Siamese weights
│   └── F5.pkl               Random Forest Judge
├── matlab_scripts/          optional .m helpers (.seq -> .tif)
├── examples/                example data (first 50 frames per video)
│   ├── real/                F8BT particle videos (Real Particle 1, 2)
│   └── synthetic/           Scenario A and Scenario B examples
├── training/                training scripts
│   ├── unet/                U-Net training and evaluation
│   ├── siamese/             Siamese network training
│   └── random_forest/       Random Forest Judge training
├── benchmark/               TrackMate-based baseline benchmarks (Fiji/Jython)
├── simulator/               Scenario A and Scenario B video synthesis
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

Runtime directories (`temp_video_frames/`, `unet_masks/`,
`filtered_masks/`, `results/`, `ground_truth/`) are created on demand
and are ignored by git.

## Reproducing the paper benchmarks

1. Generate simulated videos for Scenario A and Scenario B with the
   scripts in `simulator/`. Adjust the difficulty parameters in the
   script header to sweep the full A1–A7, B1–B8, BI/BP/BE/BX grids.
2. Run the baseline trackers (Crocker-Grier via TrackPy, Simple LAP
   and full LAP via Fiji TrackMate) on each video. Detection and
   tracking templates are in `benchmark/`. These scripts run inside
   the Fiji Jython console.
3. Run LinkBLiNK Tracker on the same videos by placing each video
   folder under `input_files/` and running `python LinkBLiNK.py`.
4. Aggregate the per-condition metrics (fragmentation rate, false
   linkage rate, completeness, association precision/recall/F1) and
   compare across methods.

## Retraining

- **U-Net.** `training/unet/train.py` retrains the segmentation model.
  The script expects a workspace layout with `data/imgs/` and
  `data/masks/`; adjust paths at the top of the script.
- **Siamese network.** `training/siamese/SiameseTrain.py` retrains
  the appearance embedding from real videos of stationary particles.
- **Random Forest Judge.** `training/random_forest/RandomForestTrain.py`
  retrains the Judge from fragment pairs labeled by ground truth.

## Inputs

LinkBLiNK accepts:

- 16-bit TIFF image sequences (one folder per video, files named in
  ascending frame order).
- StreamPix `.seq` files (the pipeline calls a bundled MATLAB script
  to extract TIFFs first; MATLAB R2021b or newer required). Set
  `input_type` to `'seq'` in `LinkBLiNK.py` and adjust the MATLAB
  path in `CONFIG['matlab_exe_path']`.

## Citation

If you use LinkBLiNK Tracker in academic work, please cite the
manuscript (in preparation). Until publication, please cite the
software directly via the `CITATION.cff` file at the top of this
repository, which is rendered as a "Cite this repository" widget by
GitHub.

## License

The source code in this repository is released under the MIT License
(see `LICENSE`). The pretrained model weights in `checkpoints/` are
distributed under the same terms.

The underlying software, registered as
*"Software for Tracking and Linking Blinking Fluorescent Particle
Positions in Nanoscale Kinetics (LinkBLiNK Tracker)"*, is a registered
copyrighted work in Thailand under **Copyright Application No. 465770**
(filed 19 December 2025 with the Department of Intellectual Property),
with **Burapha University** as the copyright holder.

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
