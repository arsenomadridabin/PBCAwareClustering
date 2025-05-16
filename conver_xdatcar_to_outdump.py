import numpy as np

def read_poscar(poscar_path):
    with open(poscar_path, 'r') as f:
        lines = f.readlines()
    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    elements = lines[5].split()
    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)
    return lattice, elements, counts, total_atoms

def convert_xdatcar_to_lammps_centered(xdatcar_path, poscar_path, output_path="out_centered.dump", center_element="Fe"):
    lattice, elements, counts, total_atoms = read_poscar(poscar_path)
    atom_types = []
    element_map = []
    for i, (el, count) in enumerate(zip(elements, counts)):
        atom_types += [i + 1] * count
        element_map += [el] * count

    lattice_inv = np.linalg.inv(lattice)

    with open(xdatcar_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as out:
        line_ptr = 0
        snapshot_count = 0
        print(len(lines))
        while line_ptr < len(lines):
            if "Direct configuration" in lines[line_ptr]:
                print("11111")
                coords = []
                line_ptr += 1
                while len(coords) < total_atoms and line_ptr < len(lines):
                    line = lines[line_ptr].strip()
                    if line and not line.startswith("Direct"):
                        parts = line.split()
                        if len(parts) == 3:
                            try:
                                coords.append([float(x) for x in parts])
                            except ValueError:
                                pass
                    line_ptr += 1

                if len(coords) != total_atoms:
                    print(f" Skipping snapshot {snapshot_count}: expected {total_atoms}, got {len(coords)}")
                    continue

                # Convert and center
                frac_coords = np.array(coords)
                cart_coords = frac_coords @ lattice

                center_indices = [i for i, el in enumerate(element_map) if el == center_element]
                ref = cart_coords[center_indices[0]]
                unwrapped = cart_coords.copy()
                for idx_i in center_indices[1:]:
                    delta = cart_coords[idx_i] - ref
                    delta -= lattice @ np.round(lattice_inv @ delta)
                    unwrapped[idx_i] = ref + delta

                center_of_mass = unwrapped[center_indices].mean(axis=0)
                box_center = np.mean(lattice, axis=0)
                shift = box_center - center_of_mass
                shifted_coords = (cart_coords + shift) % np.diag(lattice)

                # Write to LAMMPS dump
                out.write("ITEM: TIMESTEP\n")
                out.write(f"{snapshot_count}\n")
                out.write("ITEM: NUMBER OF ATOMS\n")
                out.write(f"{total_atoms}\n")
                out.write("ITEM: BOX BOUNDS pp pp pp\n")
                for i in range(3):
                    out.write(f"0.0 {lattice[i, i]:.6f}\n")
                out.write("ITEM: ATOMS id type x y z\n")
                for i in range(total_atoms):
                    out.write(f"{i+1} {atom_types[i]} {shifted_coords[i,0]:.6f} {shifted_coords[i,1]:.6f} {shifted_coords[i,2]:.6f}\n")

                snapshot_count += 1
            else:
                line_ptr += 1

    print(f"Finished: {snapshot_count} snapshots written to '{output_path}'.")

convert_xdatcar_to_lammps_centered("XDATCAR", "POSCAR", "out_centered.dump", center_element="Fe")
