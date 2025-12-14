import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_xml_tree(batch_name, annotator_name, filenames, proposals):
    root = ET.Element("annotations")

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = f"{batch_name}_{annotator_name}"

    for filename in filenames:
        # 1. Get the data block for this file
        frame_data = proposals.get(filename)

        if not frame_data:
            # If no data exists for this file in JSON, just add an empty image tag
            ET.SubElement(root, "image", id="0", name=filename)
            continue

        # 2. Extract dimensions (Safe fallback to empty string if missing)
        height = str(frame_data.get("image_height", ""))
        width = str(frame_data.get("image_width", ""))

        # 3. Create the image tag with dimensions
        image_elem = ET.SubElement(root, "image", id="0", name=filename, width=width, height=height)

        # 4. Get the list of annotations
        # The key in your JSON is "annotations", not the root object
        detections = frame_data.get("annotations", [])

        for det in detections:
            label = det.get('label')
            bbox = det.get('bbox')
            score = det.get('score', 0.0)

            if label and bbox:
                # Ensure float conversion
                xtl, ytl, xbr, ybr = map(float, bbox)

                box_elem = ET.SubElement(image_elem, "box")
                box_elem.set("label", label)
                box_elem.set("xtl", f"{xtl:.2f}")
                box_elem.set("ytl", f"{ytl:.2f}")
                box_elem.set("xbr", f"{xbr:.2f}")
                box_elem.set("ybr", f"{ybr:.2f}")
                box_elem.set("occluded", "0")
                box_elem.set("source", "manual")

                attr_elem = ET.SubElement(box_elem, "attribute")
                attr_elem.set("name", "confidence")
                attr_elem.text = f"{score:.2f}"
    return root


def convert_proposals_to_xml(proposals_file, assignment_map_path, output_dir, batch_name):
    """
    Generates CVAT XML files based on the assignment map and proposals.
    """
    print(f"--- Starting XML Generation for {batch_name} ---")

    if not os.path.exists(proposals_file) or not os.path.exists(assignment_map_path):
        print("Error: Missing proposals file or assignment map.")
        return

    with open(proposals_file, 'r') as f:
        proposals = json.load(f)

    with open(assignment_map_path, 'r') as f:
        assignments = json.load(f)

    for annotator, frames in assignments.items():
        print(f"  -> Creating XML for {annotator}...")

        xml_root = create_xml_tree(batch_name, annotator, frames, proposals)
        xml_str = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="  ")

        output_filename = f"{batch_name}_{annotator}_annotations.xml"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as f:
            f.write(xml_str)

    print("XML Generation complete.\n")