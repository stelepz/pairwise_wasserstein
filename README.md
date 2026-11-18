## Automated Equilibration Detection for MD Simulations

**Early version** This repository contains an early version of an algorithm developed during my PhD at TU Berlin (group of Prof. Maria A. Mroginski).
The method has since been substantially reworked; the current version is part of a manuscript in preparation and will be released once the paper is published.

## The problem

Every molecular dynamics (MD) simulation starts from a structure that is not in thermal equilibrium with the simulated conditions, for example a crystal structure placed in water at 300 K. 
The first part of the trajectory is therefore a relaxation phase that has to be discarded before any equilibrium property is computed. 
In practice, the end of this phase is usually judged by eye from an RMSD plot, which is subjective and makes fully automated MD pipelines impossible.

This algorithm estimates the equilibration point automatically as the onset of marginal stationarity from a single observable time series (e.g. the RMSD), without assuming a functional form 
for the relaxation and without user-chosen tolerances.
A trajectory can be considered equilibrated from the frame on where the distribution of the observable no longer changes, i.e. where the trajectory is stationary in its marginal distribution. 
To test this, the algorithm compares distributions of different parts of the trajectory with the Wasserstein distance and derives the threshold for "no longer changes" from the data itself.

## How it works

For a time series of N frames:

1. **Slicing.** For every slice size `a` from `slice_start` up to N/3, the trajectory is cut into consecutive, non-overlapping slices of `a` frames. Leftover frames are cut from the start,
   so the last slice always ends at the last frame.

3. **Pairwise distances.** For each slice size, the 1D Wasserstein distance between the value distributions of all pairs of slices is computed and scaled by `sqrt(a)`. The scaling compensates
   for the fact that the lower bound of distances between finite samples scales as `1/sqrt(a)` and makes distances comparable across slice sizes.

5. **Threshold per slice size.** The distribution of all pairwise distances of one slice size is estimated with a Gaussian kernel density estimate (KDE); its mode is taken as the typical
   distance for that slice size.

7. **Global threshold.** The per-slice-size values are combined into one global threshold. Five variants are returned: `min`, `max`, `mean`, `median`, and `kde` (the mode of a KDE over all
   per-slice-size values).

9. **Convergence point per slice size.** Starting from the last slice and moving backwards, the algorithm finds the earliest slice from which every pair of slices up to the end of the trajectory
    lies below the threshold. The first frame of that slice is the convergence estimate for this slice size.

11. **Final estimate.** The estimates of all slice sizes are averaged. If no slice size gives an estimate, the trajectory is reported as not converged.

Averaging over many slice sizes makes the result robust: small slices resolve the convergence point finely but are noisy, large slices are stable but coarse.

## Performance

For slice size `a` there are N/a slices and about (N/a)² / 2 pairs, so the workload varies by orders of magnitude between small and large slice sizes. The distance computation therefore 
switches between two parallelization strategies (wasserstein_dictionary_parallel):

- **Small slices** (below 0.5 % of N): millions of cheap comparisons, so the pairwise distances of one slice size are distributed across all cores in batches (chunksize=1000) to keep
  inter-process overhead low.
- **Large slices**: only a few pairs per size, so whole slice sizes are distributed across the cores instead.

`slice_start` has the largest influence on runtime, since the smallest slices produce by far the most pairs.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy` (≥ 2.0), `scipy`, `pandas`, `tqdm`.


### Returns

| Value | Content |
|---|---|
| `conv_points` | Estimated first converged frame for each threshold method (`min`, `max`, `mean`, `median`, `kde`). Equal to `N` if not converged. |
| `thresholds` | The global threshold for each method. |
| `details` | Per method, a list of `[slice_size, convergence_frame]` pairs for all slice sizes that gave an estimate. |
| `runtime` | `(hours, minutes, seconds)` |

## Example data

`trpzip_unfolding_RMSD.csv` contains a block-averaged RMSD time series (20,000 frames) of the unfolding of the tryptophan zipper 2 (trpzip2) β-hairpin, simulated with NAMD.

