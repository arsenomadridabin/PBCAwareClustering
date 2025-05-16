import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from sklearn.cluster import KMeans
from multiprocessing import Pool
from collections import defaultdict
import shutil

# --- Config ---
DUMP_FILE = "out_1000_small.dump"
PLOT_DIR = "plots_snapshots"
TEMP_EXCEL_DIR = "temp_excel"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(TEMP_EXCEL_DIR, exist_ok=True)

sigma = 7
rich_distance = 5
poor_distance = 2
num_bins = (100, 100, 100)
atomic_weights = {'Fe': 55.845, 'Mg': 24.305, 'Si': 28.085, 'O': 15.999, 'N': 14.007}
type_map = {1: 'Fe', 2: 'Mg', 3: 'Si', 4: 'O', 5: 'N'}

# --- Helper Functions ---
def get_largest_component(mask):
    padded = np.pad(mask, pad_width=1, mode='wrap')
    labeled, _ = label(padded)
    cropped = labeled[1:-1, 1:-1, 1:-1]

    face_pairs = []
    shape = mask.shape
    for y in range(shape[1]):
        for z in range(shape[2]):
            l1 = labeled[1, y+1, z+1]
            l2 = labeled[-2, y+1, z+1]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))
    for x in range(shape[0]):
        for z in range(shape[2]):
            l1 = labeled[x+1, 1, z+1]
            l2 = labeled[x+1, -2, z+1]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))
    for x in range(shape[0]):
        for y in range(shape[1]):
            l1 = labeled[x+1, y+1, 1]
            l2 = labeled[x+1, y+1, -2]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))

    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(x, y):
        xr, yr = find(x), find(y)
        if xr != yr:
            parent[yr] = xr
    for a, b in face_pairs:
        union(a, b)

    label_map = defaultdict(list)
    for val in np.unique(cropped):
        if val:
            label_map[find(val)].append(val)
    label_counts = {root: np.count_nonzero(np.isin(cropped, members)) for root, members in label_map.items()}
    if not label_counts:
        return np.zeros_like(cropped, dtype=bool)
    largest_root = max(label_counts, key=label_counts.get)
    return np.isin(cropped, label_map[largest_root])

def remove_small_components(mask, min_voxels=100):
    padded = np.pad(mask, pad_width=1, mode='wrap')
    labeled, _ = label(padded)
    cropped = labeled[1:-1, 1:-1, 1:-1]

    face_pairs = []
    shape = mask.shape
    for y in range(shape[1]):
        for z in range(shape[2]):
            l1 = labeled[1, y+1, z+1]
            l2 = labeled[-2, y+1, z+1]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))
    for x in range(shape[0]):
        for z in range(shape[2]):
            l1 = labeled[x+1, 1, z+1]
            l2 = labeled[x+1, -2, z+1]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))
    for x in range(shape[0]):
        for y in range(shape[1]):
            l1 = labeled[x+1, y+1, 1]
            l2 = labeled[x+1, y+1, -2]
            if l1 and l2 and l1 != l2:
                face_pairs.append((l1, l2))

    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def union(x, y):
        xr, yr = find(x), find(y)
        if xr != yr:
            parent[yr] = xr
    for a, b in face_pairs:
        union(a, b)

    label_map = defaultdict(list)
    for val in np.unique(cropped):
        if val:
            label_map[find(val)].append(val)

    keep_labels = []
    for root, members in label_map.items():
        if np.count_nonzero(np.isin(cropped, members)) >= min_voxels:
            keep_labels.extend(members)
    return np.isin(cropped, keep_labels)

def extend_periodic_positions(positions, box_length):
    shifts = [-1, 0, 1]
    shift_vectors = np.array([[i, j, k] for i in shifts for j in shifts for k in shifts])
    return np.vstack([positions + shift * box_length for shift in shift_vectors])

def compute_weight_percent(atom_data):
    atom_types = atom_data[:, 1].astype(int)
    positions = atom_data[:, 2:5]
    fe_positions = positions[atom_types == 1]
    box_min, box_max = positions.min(0), positions.max(0)
    box_length = box_max - box_min
    fe_positions_wrapped = extend_periodic_positions(fe_positions, box_length)

    fe_hist, edges = np.histogramdd(fe_positions_wrapped, bins=num_bins, range=[[box_min[0], box_max[0]],
                                                                                 [box_min[1], box_max[1]],
                                                                                 [box_min[2], box_max[2]]])
    fe_density = gaussian_filter(fe_hist, sigma=sigma)

    flat = fe_density.flatten()
    valid = flat >= 0
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat[valid].reshape(-1, 1))
    label_map = dict(zip(np.argsort(kmeans.cluster_centers_.flatten()), [0, 1]))

    binary_mask = np.full(flat.shape, -1)
    binary_mask[valid] = np.vectorize(label_map.get)(labels)
    binary_mask = binary_mask.reshape(fe_density.shape)

    rich_mask = get_largest_component(binary_mask == 1)
    poor_mask = remove_small_components((binary_mask == 0) & (~rich_mask))
    dist_rich = distance_transform_edt(~rich_mask)
    dist_poor = distance_transform_edt(~poor_mask)
    boundary_mask = (dist_rich < rich_distance) & (dist_poor < poor_distance)

    region_labels = np.full(fe_density.shape, -1)
    region_labels[boundary_mask] = 2
    region_labels[rich_mask & ~boundary_mask] = 1
    region_labels[poor_mask & ~boundary_mask] = 0
    unlabeled = region_labels == -1
    region_labels[unlabeled & (dist_rich < dist_poor)] = 1
    region_labels[unlabeled & (dist_rich >= dist_poor)] = 0

    x_edges, y_edges, z_edges = edges
    x_bin = np.clip(np.digitize(positions[:, 0], x_edges) - 1, 0, num_bins[0]-1)
    y_bin = np.clip(np.digitize(positions[:, 1], y_edges) - 1, 0, num_bins[1]-1)
    z_bin = np.clip(np.digitize(positions[:, 2], z_edges) - 1, 0, num_bins[2]-1)
    atom_region_labels = region_labels[x_bin, y_bin, z_bin]

    region_counts = {0: {e: 0 for e in atomic_weights}, 1: {e: 0 for e in atomic_weights}, 2: {e: 0 for e in atomic_weights}}
    for t, r in zip(atom_types, atom_region_labels):
        el = type_map.get(t)
        if el and r in region_counts:
            region_counts[r][el] += 1
    counts_df = pd.DataFrame(region_counts).rename(columns={0: 'Fe-poor', 1: 'Fe-rich', 2: 'Boundary'})

    weight_percent = {}
    for region in ['Fe-rich', 'Fe-poor']:
        total_mass = sum(counts_df[region][el] * atomic_weights[el] for el in atomic_weights)
        weight_percent[region] = {
            el: (counts_df[region][el] * atomic_weights[el]) / total_mass * 100 if total_mass > 0 else 0.0
            for el in atomic_weights
        }

    return pd.DataFrame(weight_percent).round(2), counts_df, region_labels, fe_density

def process_snapshot(idx_snap):
    idx, snap = idx_snap
    wp_df, count_df, region_labels, fe_density = compute_weight_percent(snap)
    x, y, z = region_labels.shape

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(region_labels[:, :, z // 2], cmap='jet', origin='lower')
    axes[0, 1].imshow(region_labels[:, y // 2, :], cmap='jet', origin='lower')
    axes[0, 2].imshow(region_labels[x // 2, :, :], cmap='jet', origin='lower')
    axes[1, 0].imshow(fe_density[:, :, z // 2], cmap='hot', origin='lower')
    axes[1, 1].imshow(fe_density[:, y // 2, :], cmap='hot', origin='lower')
    axes[1, 2].imshow(fe_density[x // 2, :, :], cmap='hot', origin='lower')
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/snapshot_{idx+1:03d}_combined.png", dpi=300)
    plt.close()

    temp_excel = f"{TEMP_EXCEL_DIR}/snapshot_{idx+1:03d}.xlsx"
    with pd.ExcelWriter(temp_excel) as writer:
        wp_df.to_excel(writer, sheet_name="WeightPercent", index=True)
        count_df.to_excel(writer, sheet_name="Counts", index=True)

    return wp_df

if __name__ == '__main__':
    with open(DUMP_FILE, "r") as f:
        lines = f.readlines()

    snapshots = []
    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            n_atoms = int(lines[i + 3])
            data = lines[i + 9 : i + 9 + n_atoms]
            atom_data = np.genfromtxt(data, invalid_raise=False)
            if atom_data.shape[0] == n_atoms:
                snapshots.append(atom_data)
            i += 9 + n_atoms
        else:
            i += 1

    with Pool() as pool:
        weight_percent_dfs = pool.map(process_snapshot, list(enumerate(snapshots)))

    with pd.ExcelWriter("weight_percent_summary.xlsx") as writer:
        for idx in range(len(weight_percent_dfs)):
            snap_file = f"{TEMP_EXCEL_DIR}/snapshot_{idx+1:03d}.xlsx"
            temp_wb = pd.read_excel(snap_file, sheet_name=None)
            for sheet_name, df in temp_wb.items():
                df.to_excel(writer, sheet_name=f"Snapshot_{idx+1}_{sheet_name}", index=True)

    avg_df = sum(weight_percent_dfs) / len(weight_percent_dfs)
    avg_df.round(2).to_excel("average_weight_percent.xlsx")

    shutil.rmtree(TEMP_EXCEL_DIR)
