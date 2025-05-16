def skip_snapshots_from_dump(input_file="out.dump", skip=10, output_file="output_10_skipped.dump"):
    with open(input_file, "r") as f:
        lines = f.readlines()

    frame_id = 0
    line_idx = 0
    output_lines = []

    while line_idx < len(lines):
        if not lines[line_idx].startswith("ITEM: TIMESTEP"):
            line_idx += 1
            continue

        # Start of a snapshot
        timestep = lines[line_idx + 1].strip()
        num_atoms = int(lines[line_idx + 3].strip())
        atom_start = line_idx + 9
        atom_end = atom_start + num_atoms

        snapshot_lines = lines[line_idx:atom_end]

        if frame_id % skip == 0:
            output_lines.extend(snapshot_lines)

        line_idx = atom_end
        frame_id += 1

    with open(output_file, "w") as out_f:
        out_f.writelines(output_lines)

    print(f"Saved skipped output to: {output_file}")


# Run the function
skip_snapshots_from_dump("out.dump", skip=25, output_file="output_25_skipped.dump")
