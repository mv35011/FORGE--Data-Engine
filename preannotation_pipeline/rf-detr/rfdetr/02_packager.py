import os
import argparse
from datetime import datetime

# Import modules from the tools folder
# Ensure you have an empty __init__.py in the tools/ folder
from tools.assignment_generator import generate_assignments
from tools.proposal_to_cvat import convert_proposals_to_xml

# --- CONFIGURATION ---
DEFAULT_ANNOTATORS = ["annotator1", "annotator2", "annotator3"]
FRAMES_INPUT_DIR = "outputs/mining_results/frames"
PROPOSALS_INPUT_FILE = "outputs/mining_results/proposals.json"
WORKPACKAGE_ROOT = "outputs/workpackages"


# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Create CVAT workpackages from mined data.")
    parser.add_argument("--batch_name", type=str, default=None, help="Name of the batch (default: batch_YYYYMMDD)")
    parser.add_argument("--annotators", nargs='+', default=DEFAULT_ANNOTATORS, help="List of annotator names")
    args = parser.parse_args()

    # 1. Setup Naming
    if args.batch_name:
        batch_name = args.batch_name
    else:
        # Auto-generate batch name if not provided
        batch_name = f"batch_{datetime.now().strftime('%Y%m%d')}"

    print(f"=== Starting Pipeline for {batch_name} ===")

    # Define output specific to this batch
    output_dir = os.path.join(WORKPACKAGE_ROOT, batch_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    # 2. Run Assignment Generator (Frames -> Zips + Map)
    map_path = generate_assignments(
        frames_dir=FRAMES_INPUT_DIR,
        output_dir=output_dir,
        batch_name=batch_name,
        annotators=args.annotators
    )

    if not map_path:
        print("Pipeline aborted: Failed to generate assignment map.")
        return

    # 3. Run Proposal Converter (Proposals + Map -> XMLs)
    convert_proposals_to_xml(
        proposals_file=PROPOSALS_INPUT_FILE,
        assignment_map_path=map_path,
        output_dir=output_dir,
        batch_name=batch_name
    )

    print(f"=== Workpackage '{batch_name}' Created Successfully! ===")
    print(f"Location: {os.path.abspath(output_dir)}")
    print("Contents:")
    for f in os.listdir(output_dir):
        print(f" - {f}")


if __name__ == "__main__":
    main()