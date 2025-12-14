import os
import random
import json
import zipfile
from math import ceil


def generate_assignments(frames_dir, output_dir, batch_name, annotators):
    """
    Splits frames among annotators, creates zip files, and saves an assignment map.
    Returns the path to the assignment map.
    """
    print(f"--- Starting Assignment Generation for {batch_name} ---")

    # 1. Get all frame files
    all_frames = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_frames = len(all_frames)

    if total_frames == 0:
        print("No frames found in frames_dir.")
        return None

    # 2. Shuffle and Split
    random.shuffle(all_frames)

    # Calculate split logic
    n_annotators = len(annotators)
    k, m = divmod(total_frames, n_annotators)
    splits = [all_frames[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_annotators)]

    assignment_map = {}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. Create Zips
    for idx, annotator in enumerate(annotators):
        assigned_frames = splits[idx]
        assignment_map[annotator] = assigned_frames

        # Define zip name
        zip_filename = f"{batch_name}_{annotator}_frames.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        print(f"  -> Zipping {len(assigned_frames)} frames for {annotator}...")

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for frame_file in assigned_frames:
                src_path = os.path.join(frames_dir, frame_file)
                zipf.write(src_path, arcname=frame_file)

    # 4. Save Assignment Map
    map_path = os.path.join(output_dir, "assignment_map.json")
    with open(map_path, 'w') as f:
        json.dump(assignment_map, f, indent=4)

    print(f"Assignments complete. Map saved to: {map_path}\n")
    return map_path