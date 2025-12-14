import json
import os

PATHS = [
    "/workspace/dataset/ppe_coco_format/train/_annotations.coco.json",
    "/workspace/dataset/ppe_coco_format/valid/_annotations.coco.json"
]


def fix_json(path):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return

    print(f"🔧 Fixing: {path}")
    with open(path, 'r') as f:
        data = json.load(f)

    # 1. Sort categories by ID to ensure consistent mapping
    # (This assumes your categories are consistent across train/valid)
    categories = sorted(data['categories'], key=lambda x: x['id'])

    print("   Original IDs found:", [c['id'] for c in categories])

    # 2. Create Mapping (Old ID -> New 0-indexed ID)
    id_map = {}
    new_categories = []
    for new_id, cat in enumerate(categories):
        old_id = cat['id']
        id_map[old_id] = new_id

        # Update category definition
        cat['id'] = new_id
        new_categories.append(cat)
        # print(f"   Mapped {old_id} -> {new_id} ({cat['name']})")

    data['categories'] = new_categories

    # 3. Update Annotations
    for ann in data['annotations']:
        if ann['category_id'] in id_map:
            ann['category_id'] = id_map[ann['category_id']]
        else:
            print(f"⚠️ Warning: Found unknown category ID {ann['category_id']}! Deleting annotation.")
            data['annotations'].remove(ann)

    # 4. Save
    with open(path, 'w') as f:
        json.dump(data, f)
    print("   ✅ Fixed and saved.")


if __name__ == "__main__":
    for p in PATHS:
        fix_json(p)