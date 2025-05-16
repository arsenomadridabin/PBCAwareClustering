def extract_last_snapshot(input_file="out.dump", output_file="last_snapshot.dump"):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    snapshot_starts = []
    line_idx = 0

    # Find all starting indices of snapshots
    while line_idx < len(lines):
        if lines[line_idx].startswith("ITEM: TIMESTEP"):
            snapshot_starts.append(line_idx)
        line_idx += 1

    if not snapshot_starts:
        print("No snapshots found in the file.")
        return

    # Use the last snapshot
    last_start = snapshot_starts[-1]
    num_atoms = int(lines[last_start + 3].strip())
    last_end = last_start + 9 + num_atoms

    last_snapshot_lines = lines[last_start:last_end]

    with open(output_file, "w") as out:
        out.writelines(last_snapshot_lines)

    print(f"Last snapshot written to: {output_file}")


# Run the function
extract_last_snapshot("small_centered_all.dump", "last_snapshot.dump")
