#Bijaya sir recommended PBC aware union find (Small Cluster using Kmeans PBC Aware)
#Try this
# This is specially designed for single snapshot (the last one)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Config
DUMP_FILE = "last_snapshot_3000k.dump"
PLOT_DIR = "plots_snapshots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Change arguments here
sigma = 7
rich_distance = 5
poor_distance = 2
num_bins = (100, 100, 100)
atomic_weights = {'Fe': 55.845, 'Mg': 24.305, 'Si': 28.085, 'O': 15.999, 'N': 14.007}
type_map = {1: 'Fe', 2: 'Mg', 3: 'Si', 4: 'O', 5: 'N'}
elem_list = list(atomic_weights.keys())

#SOmeUtilities
def get_largest_component(mask):
    from scipy.ndimage import label
    from collections import defaultdict

    padded = np.pad(mask, pad_width=1, mode='wrap')
    labeled, _ = label(padded)
    cropped = labeled[1:-1, 1:-1, 1:-1]

    # Track face-connected label pairs
    face_pairs = []
    shape = mask.shape

    # X-faces
    for y in range(shape[1]):
        for z in range(shape[2]):
            l1 = labeled[1, y+1, z+1]
            l2 = labeled[-2, y+1, z+1]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    # Y-faces
    for x in range(shape[0]):
        for z in range(shape[2]):
            l1 = labeled[x+1, 1, z+1]
            l2 = labeled[x+1, -2, z+1]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    # Z-faces
    for x in range(shape[0]):
        for y in range(shape[1]):
            l1 = labeled[x+1, y+1, 1]
            l2 = labeled[x+1, y+1, -2]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    # Union-Find structure
    parent = dict()
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
        if val != 0:
            label_map[find(val)].append(val)

    #Count total voxels for each unified label group
    label_counts = {}
    for root, members in label_map.items():
        mask_union = np.isin(cropped, members)
        label_counts[root] = np.count_nonzero(mask_union)

    if not label_counts:
        return np.zeros_like(cropped, dtype=bool)

    largest_root = max(label_counts, key=label_counts.get)
    largest_labels = label_map[largest_root]
    return np.isin(cropped, largest_labels)

def remove_small_components(mask, min_voxels=100):
    from scipy.ndimage import label
    from collections import defaultdict

    padded = np.pad(mask, pad_width=1, mode='wrap')
    labeled, _ = label(padded)
    cropped = labeled[1:-1, 1:-1, 1:-1]

    shape = mask.shape
    face_pairs = []

    for y in range(shape[1]):
        for z in range(shape[2]):
            l1 = labeled[1, y+1, z+1]
            l2 = labeled[-2, y+1, z+1]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    for x in range(shape[0]):
        for z in range(shape[2]):
            l1 = labeled[x+1, 1, z+1]
            l2 = labeled[x+1, -2, z+1]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    for x in range(shape[0]):
        for y in range(shape[1]):
            l1 = labeled[x+1, y+1, 1]
            l2 = labeled[x+1, y+1, -2]
            if l1 != 0 and l2 != 0 and l1 != l2:
                face_pairs.append((l1, l2))

    parent = dict()
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
        if val != 0:
            label_map[find(val)].append(val)

    keep_labels = []
    for root, members in label_map.items():
        mask_union = np.isin(cropped, members)
        if np.count_nonzero(mask_union) >= min_voxels:
            keep_labels.extend(members)

    return np.isin(cropped, keep_labels)

def extend_periodic_positions(fe_positions, box_length):
    shifts = [-1, 0, 1]
    shift_vectors = np.array([[i, j, k] for i in shifts for j in shifts for k in shifts])
    all_fe = [fe_positions + shift * box_length for shift in shift_vectors]
    return np.vstack(all_fe)

def compute_weight_percent(atom_data):
    atom_types = atom_data[:, 1].astype(int)
    positions = atom_data[:, 2:5]
    fe_positions = positions[atom_types == 1]

    box_min = positions.min(0)
    box_max = positions.max(0)
    box_length = box_max - box_min

    fe_positions_wrapped = extend_periodic_positions(fe_positions, box_length)

    fe_hist, edges = np.histogramdd(fe_positions_wrapped, bins=num_bins, range=[[box_min[0], box_max[0]],
                                                                                 [box_min[1], box_max[1]],
                                                                                 [box_min[2], box_max[2]]])
    fe_density = gaussian_filter(fe_hist, sigma=sigma)

    flat_density = fe_density.flatten()
    valid = flat_density >= 0
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat_density[valid].reshape(-1, 1))
    sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
    label_map = {sorted_centers[0]: 0, sorted_centers[1]: 1}
    binary_mask_flat = np.full(flat_density.shape, -1)
    binary_mask_flat[valid] = np.vectorize(label_map.get)(labels)
    binary_mask = binary_mask_flat.reshape(fe_density.shape)

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

def plot_voxels_with_box(fe_rich_voxels, box_size=100, save_path=None):
    fig = plt.figure(figsize=(8, 8), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # === Plot the voxel data ===
    ax.voxels(fe_rich_voxels, facecolors='crimson', edgecolor='black', linewidth=0.3, alpha=0.7)

    # === Add simulation box ===
    edges = [
        [(0,0,0), (box_size,0,0)], [(0,0,0), (0,box_size,0)], [(0,0,0), (0,0,box_size)],
        [(box_size,0,0), (box_size,box_size,0)], [(box_size,0,0), (box_size,0,box_size)],
        [(0,box_size,0), (box_size,box_size,0)], [(0,box_size,0), (0,box_size,box_size)],
        [(0,0,box_size), (0,box_size,box_size)], [(0,0,box_size), (box_size,0,box_size)],
        [(box_size,box_size,0), (box_size,box_size,box_size)],
        [(box_size,0,box_size), (box_size,box_size,box_size)],
        [(0,box_size,box_size), (box_size,box_size,box_size)]
    ]
    ax.add_collection3d(Line3DCollection(edges, colors='gray', linewidths=0.5, alpha=0.5))

    # === Axes styling ===
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()

if __name__=='__main__':
    #Load dump
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
                #Break (Because we are doing for only one snapshot)
                break

    # Process and plot
    weight_percent_dfs = []
    region_counts_dfs = []
    idx = 0
    snap = snapshots[0]
    wp_df, count_df, region_labels, fe_density = compute_weight_percent(snap)
    weight_percent_dfs.append(wp_df)
    region_counts_dfs.append(count_df)

    x, y, z = region_labels.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Classification
    axes[0, 0].imshow(region_labels[:, :, z // 2], cmap='jet', origin='lower')
    axes[0, 1].imshow(region_labels[:, y // 2, :], cmap='jet', origin='lower')
    axes[0, 2].imshow(region_labels[x // 2, :, :], cmap='jet', origin='lower')

    axes[0, 0].set_title("XY - Region")
    axes[0, 1].set_title("YZ - Region")
    axes[0, 2].set_title("ZX - Region")

    # Density
    axes[1, 0].imshow(fe_density[:, :, z // 2], cmap='hot', origin='lower')
    axes[1, 1].imshow(fe_density[:, y // 2, :], cmap='hot', origin='lower')
    axes[1, 2].imshow(fe_density[x // 2, :, :], cmap='hot', origin='lower')

    axes[1, 0].set_title("XY - Fe Density")
    axes[1, 1].set_title("YZ - Fe Density")
    axes[1, 2].set_title("ZX - Fe Density")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/snapshot_{idx+1:03d}_combined.png", dpi=300)
    plt.close()

    # Save Excel with multiple sheets
    with pd.ExcelWriter("weight_percent_summary.xlsx") as writer:
        for idx in range(len(weight_percent_dfs)):
            weight_percent_dfs[idx].to_excel(writer, sheet_name=f"Snapshot_{idx+1}_wt%", index=True)
            region_counts_dfs[idx].to_excel(writer, sheet_name=f"Snapshot_{idx+1}_counts", index=True)

    # Save average weight percent
    avg_df = sum(weight_percent_dfs) / len(weight_percent_dfs)
    avg_df.round(2).to_excel("average_weight_percent.xlsx")



    # If alongside the Excel output and plots of middle slice you also need "3D voexl structure" which is computationally little expensive then, uncomment below line.
    # (region_labels==2) filters all the Fe rich region
    plot_voxels_with_box(region_labels==2)
