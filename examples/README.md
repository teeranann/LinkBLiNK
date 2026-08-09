# Example data

This folder ships the complete example videos used in the paper: the full
experimental SMLM recordings of F8BT nanoparticles (1000 frames each) and one
full simulated video per benchmark scenario (500 frames each, with ground
truth). They are sufficient to run the LinkBLiNK pipeline end-to-end and to
reproduce the real-data demonstration.

```
real/Real Particle 1/   real F8BT nanoparticle video, 1000 frames (16-bit TIFF)
real/Real Particle 2/   real F8BT nanoparticle video, 1000 frames (16-bit TIFF)
synthetic/ScenarioA/    simulated reconnection scenario, 500 frames
synthetic/ScenarioB/    simulated impostor-rejection scenario, 500 frames
                        + scenario_meta.json (ground truth)
```

The full benchmark sets (100 videos per condition across Scenarios A, B and
the BI/BP/BE/BX series) can be regenerated with the config-driven scripts in
`simulator/` using the difficulty-level values tabulated in Supplementary
Tables S2–S5 of the manuscript, and are also available from the corresponding
author on reasonable request.
