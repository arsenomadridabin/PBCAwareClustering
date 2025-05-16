# PBC-Aware Regional Composition Analysis

This repository provides a complete pipeline to analyze Fe-rich, Fe-poor, and boundary regions from molecular dynamics simulations (e.g., LAMMPS `dump` files), while fully respecting periodic boundary conditions (PBC).

## Features

* **Fe-density estimation** using 3D histogram + Gaussian smoothing with PBC tiling
* **Region segmentation** via KMeans clustering of Fe density
* **Connected component labeling** that merges components across periodic box boundaries
* **Asymmetric boundary detection** using distance transforms
* **Per-region atomic composition** and **weight percent** calculation
* **Support for multiple snapshots**
* **Visualizations** in XY, YZ, and ZX planes
* **Excel reports**: per-snapshot + average weight % and atom counts

## Output Files

* `plots_snapshots/` — 6-panel plots for each snapshot (classification + Fe density)
* `weight_percent_summary.xlsx` — weight % + atom counts (one sheet per snapshot)
* `average_weight_percent.xlsx` — averaged weight % across all snapshots

## How It Works

1. **Fe positions** are replicated in 3×3×3 PBC-aware tiles
2. 3D Fe density histogram is computed and smoothed with a Gaussian kernel
3. KMeans clustering (n=2) separates high vs. low density voxels
4. Fe-rich component is extracted using **PBC-aware connected components**
5. Fe-poor regions are cleaned with **PBC-aware small patch removal**
6. Boundary voxels are defined via asymmetric distance thresholds to Fe-rich and Fe-poor
7. Atom positions are voxelized, and per-region compositions are computed
8. Slices and summary stats are saved

## Requirements

* Python 3.8+
* NumPy, Pandas, Matplotlib, SciPy, scikit-learn

---

## HPC Resources at LSU SMIC

#SBATCH -N 20                    # SMIC allows at max 20 nodes at a time (although 86 overall)
#SBATCH -n 20                   # SMIC allows number of MPI 20 per node  
##SBATCH -c 12                  # 6 threads per MPI process
#SBATCH -t 15:00:00
#SBATCH -p checkpt
#SBATCH -A hpc_bb_karki3
#SBATCH -o  gb.out
#SBATCH -e  err.out
