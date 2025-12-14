import os
import cv2
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pre_process_image(img):
    """Convert image to grayscale and blur for robust comparison."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    return gray


def extract_keyframes(video_path: str, video_output_dir: str, basename: str, similarity_threshold: float,
                      min_frames_between_saves: int):
    """
    Extracts keyframes from a single video based on scene changes.

    Args:
        video_path: Path to the input .mp4 file.
        video_output_dir: Folder to save the extracted frames.
        basename: The name of the video file (e.g., "cctv_01").
        similarity_threshold: (0.0 to 1.0) How similar frames must be to be "skipped".
        min_frames_between_saves: Number of frames to skip *after* saving, to prevent micro-changes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open video file: {video_path}")
        return 0

    last_saved_frame = None
    last_frame_number = -9999
    saved_frame_count = 0
    frame_number = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Processing {basename}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            pbar.update(1)

            # --- Smart Sampling Logic ---
            # 1. Check if we're in the cooldown period
            if frame_number < (last_frame_number + min_frames_between_saves):
                continue

            processed_frame = pre_process_image(frame)

            # 2. Check if this is the first frame to be saved
            if last_saved_frame is None:
                is_new_scene = True
            else:
                # 3. Compare current frame to the last *saved* frame
                score = ssim(last_saved_frame, processed_frame,
                             data_range=processed_frame.max() - processed_frame.min())
                if score < similarity_threshold:
                    is_new_scene = True
                else:
                    is_new_scene = False

            # --- Save the Frame ---
            if is_new_scene:
                frame_filename = f"{basename}_frame_{frame_number:06d}.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)

                # Resize for manageable file size and faster inference
                frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.imwrite(frame_path, frame_resized)

                last_saved_frame = processed_frame
                last_frame_number = frame_number
                saved_frame_count += 1

    cap.release()
    logger.info(f"✅ Extracted {saved_frame_count} keyframes from {basename}")
    return saved_frame_count


def main(input_dir: str, output_dir: str, threshold: float, cooldown: int):
    """
    Main function to find all videos and process them.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not video_files:
        logger.warning(f"⚠️ No video files found in '{input_dir}'.")
        return

    logger.info(f"Found {len(video_files)} videos. Starting keyframe extraction...")

    total_saved = 0
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        basename = Path(video_file).stem
        video_output_dir = os.path.join(output_dir, basename)
        os.makedirs(video_output_dir, exist_ok=True)

        saved_count = extract_keyframes(video_path, video_output_dir, basename, threshold, cooldown)
        total_saved += saved_count

    logger.info(f"🎉🎉 Frame extraction complete. Total keyframes saved: {total_saved} 🎉🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Frame Extractor for Project FORGE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="Path to folder with raw video files.")
    parser.add_argument("--output_dir", required=True, help="Path to save extracted keyframe folders.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Similarity threshold (0.0-1.0). Lower = more frames. 0.92 is a good start."
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=20,
        help="Min frames to skip after saving a keyframe. (e.g., 20 frames = ~1 sec cooldown)"
    )
    args = parser.parse_args()

    # Create the output directory one level up, so we have `frames_sampled/`
    # and not `frames_sampled/cctv_01/` etc.
    # We will put all frames in one big folder.

    # --- This is a flaw in the logic. Let's fix it. ---
    # We want ONE output folder, not one per video.

    # Let's re-write the main function to be simpler.

    pass  # Pass on the old main, let's use a better one.


def main_v2(input_dir: str, output_dir: str, threshold: float, cooldown: int):
    """
    Main function to find all videos and process them, saving all frames
    to a SINGLE output directory for easier processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not video_files:
        logger.warning(f"⚠️ No video files found in '{input_dir}'.")
        return

    logger.info(f"Found {len(video_files)} videos. Starting keyframe extraction...")

    total_saved = 0
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        basename = Path(video_file).stem

        # We pass the main output_dir, not a sub-directory
        saved_count = extract_keyframes(video_path, output_dir, basename, threshold, cooldown)
        total_saved += saved_count

    logger.info(f"🎉🎉 Frame extraction complete. Total keyframes saved: {total_saved} 🎉🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Frame Extractor for Project FORGE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, help="Path to folder with raw video files. (e.g., ./videos_raw)")
    parser.add_argument("--output_dir", required=True,
                        help="Path to save all extracted keyframes. (e.g., ./frames_sampled)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Similarity threshold (0.0-1.0). Lower = more frames. 0.92 is a good start."
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=20,
        help="Min frames to skip after saving a keyframe. (e.g., 20 frames = ~1 sec cooldown)"
    )
    args = parser.parse_args()

    # Run the new main function
    main_v2(args.input_dir, args.output_dir, args.threshold, args.cooldown)