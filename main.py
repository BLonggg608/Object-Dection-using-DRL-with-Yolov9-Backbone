"""Quick script to run the offline DRL video annotation pipeline."""
import sys
from pathlib import Path
from config import DEVICE

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from demo.video_pipeline import annotate_video_with_drl, snapshot_frames_with_drl
from demo.model_loader import load_drl_components


def main():
    print(f"Using device: {DEVICE}")
    feature_model, deep_q, classification_model = load_drl_components()

    input_video = Path("input_vid/carousel_final_long_640.avi")
    output_video = Path("output_vid/carousel_final_long_640_drl.avi")
    output_dir = Path("output_img")

    # output_path = annotate_video_with_drl(
    #     feature_model=feature_model,
    #     deep_q=deep_q,
    #     classification_model=classification_model,
    #     video_path=input_video,
    #     output_path=output_video,
    #     frame_stride=30,
    #     progress_interval=100,
    # )

    # print(f"Saved annotated video to {output_path.resolve()}")

    count = snapshot_frames_with_drl(
        feature_model=feature_model,
        deep_q=deep_q,
        classification_model=classification_model,
        video_path=input_video,
        output_dir=output_dir,
        frame_stride=30,
        progress_interval=100,
    )

    print(f"save {count} images.")


if __name__ == "__main__":
    main()
