import json
import os

# --- CONFIG ---
# List all your annotation files here
FILES_TO_FIX = [
    "ppe_coco_format/ppe_coco_format/train/_annotations.coco.json",
    "ppe_coco_format/ppe_coco_format/valid/_annotations.coco.json",
    "ppe_coco_format/ppe_coco_format/test/_annotations.coco.json"
]


def fix_json(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (Not found)")
        return

    print(f"Processing {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. Check if fix is needed
    # If the first category is already 1, we might not need to do anything.
    # But we check if '0' exists to be safe.
    category_ids = [c['id'] for c in data['categories']]
    if 0 not in category_ids:
        print(f"  -> No ID 0 found. Looks safe. Skipping.")
        return

    print(f"  -> Found ID 0. shifting IDs by +1...")

    # 2. Create a mapping (Old ID -> New ID)
    id_map = {}
    for cat in data['categories']:
        old_id = cat['id']
        new_id = old_id + 1
        cat['id'] = new_id
        id_map[old_id] = new_id

    # 3. Update Annotations
    count = 0
    for ann in data['annotations']:
        old_cat = ann['category_id']
        if old_cat in id_map:
            ann['category_id'] = id_map[old_cat]
            count += 1

    # 4. Save
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print(f"  -> Fixed {count} annotations. Saved.")


if __name__ == "__main__":
    for p in FILES_TO_FIX:
        fix_json(p)