import numpy as np

def unwrap_and_center_all_snapshots(input_file="out.dump",
                                    output_file="com_centered_unwrapped.dump",
                                    box_size=17.0):
    with open(input_file, "r") as f:
        lines = f.readlines()

    snapshot_starts = [i for i, line in enumerate(lines) if "ITEM: TIMESTEP" in line]
    with open(output_file, "w") as out:
        for snap_idx, start in enumerate(snapshot_starts):
            try:
                num_atoms = int(lines[start + 3].strip())
                header = lines[start:start + 9]
                atom_lines = lines[start + 9:start + 9 + num_atoms]
                atom_data = np.genfromtxt(atom_lines)

                ids = atom_data[:, 0].astype(int)
                types = atom_data[:, 1].astype(int)
                positions = atom_data[:, 2:5]

                fe_positions = positions[types == 1]
                if len(fe_positions) == 0:
                    print(f"Snapshot {snap_idx}: No Fe atoms found. Skipping.")
                    continue

                # Unwrap Fe atoms
                unwrapped = fe_positions.copy()
                ref = unwrapped[0]
                for i in range(1, len(unwrapped)):
                    delta = unwrapped[i] - ref
                    delta -= box_size * np.round(delta / box_size)
                    unwrapped[i] = ref + delta

                fe_com_unwrapped = unwrapped.mean(axis=0)
                box_center = np.array([box_size / 2.0] * 3)
                shift = box_center - fe_com_unwrapped

                shifted_positions = (positions + shift) % box_size

                # Write centered snapshot to output
                out.writelines(header)
                for i in range(num_atoms):
                    out.write(f"{ids[i]:.0f} {types[i]:.0f} " +
                              f"{shifted_positions[i, 0]:.6f} {shifted_positions[i, 1]:.6f} {shifted_positions[i, 2]:.6f}\n")

                print(f"Snapshot {snap_idx} centered and written.")
            except Exception as e:
                print(f"Failed to process snapshot at line {start}: {e}")

# Here output_n_skipped.dump is a dump file generated by uniformly sampling ourput.dump
# For uniform sampling check skip_outdump.py
unwrap_and_center_all_snapshots("output_n_skipped.dump")
