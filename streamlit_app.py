"""Streamlit UI for DRL video annotation and snapshot workflows."""

import io
import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Ensure package imports work whether run via `streamlit run` or as module.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.model_loader import load_drl_components
from demo.video_pipeline import annotate_video_with_drl, snapshot_frames_with_drl  # noqa: E402

OUTPUT_VIDEO_DIR = PACKAGE_ROOT / "output_vid"
OUTPUT_IMAGE_DIR = PACKAGE_ROOT / "output_img"
UPLOADED_DIR = PACKAGE_ROOT / "input_vid"


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[object, object, object]:
    """Load and cache the DRL components for reuse across interactions."""

    feature_model, deep_q, classification_model = load_drl_components()
    return feature_model, deep_q, classification_model


def save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Persist uploaded video to the output directory and return the saved path."""

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_VIDEO_DIR / uploaded_file.name

    # If a file with the same name exists, append a numeric suffix.
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while True:
            candidate = OUTPUT_VIDEO_DIR / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                file_path = candidate
                break
            counter += 1

    with file_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return file_path


def show_video(path: Path) -> None:
    """Display a local video inside the Streamlit app."""

    with path.open("rb") as video_file:
        st.video(video_file.read())


def show_snapshots(directory: Path) -> None:
    """Display snapshot images sorted by filename."""

    images = sorted(directory.glob("*.png"))
    if not images:
        st.info("No snapshots generated.")
        return

    cols = st.columns(3)
    for idx, image_path in enumerate(images):
        with image_path.open("rb") as img_file:
            image_bytes = img_file.read()
        cols[idx % 3].image(io.BytesIO(image_bytes), caption=image_path.name)


st.set_page_config(page_title="DRL Video Inspector", layout="wide")
st.title("🔍 DRL Baggage Inspection")
st.write("Tải video kiểm tra hành lý và chạy mô hình DRL để annotate hoặc trích ảnh đáng chú ý.")

uploaded_video = st.file_uploader(
    "Tải video X-ray", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    saved_video_path = save_uploaded_video(uploaded_video)
    # st.success(f"Đã lưu video tại: {saved_video_path.relative_to(PROJECT_ROOT)}")
    show_video(saved_video_path)

    with st.spinner("Đang tải mô hình DRL..."):
        feature_model, deep_q, classification_model = load_models()

    action = st.radio(
        "Chọn tác vụ",
        # ("Annotate Video", "Snapshot Frames"),
        ("Snapshot Frames"),
        captions=[
            "Chạy mô hình và lưu video annotate vào output_vid",
            "Lưu các frame có phát hiện vào output_img",
        ],
    )
    stride = st.number_input("frame_stride", min_value=1, value=30, step=1)

    if st.button("Thực thi", type="primary"):
        if action == "Annotate Video":
            OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            annotated_path = OUTPUT_VIDEO_DIR / \
                f"{saved_video_path.stem}_annotated.mp4"
            with st.spinner("Đang chạy annotate video..."):
                result_path = annotate_video_with_drl(
                    feature_model=feature_model,
                    deep_q=deep_q,
                    classification_model=classification_model,
                    video_path=saved_video_path,
                    output_path=annotated_path,
                    frame_stride=int(stride),
                )
            st.success(
                f"Đã lưu video annotate: {result_path.relative_to(PROJECT_ROOT)}")
            show_video(result_path)
        else:
            OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            gallery_placeholder = st.empty()
            displayed_snapshots: List[Path] = []

            def _update_snapshot_gallery(new_path: Path) -> None:
                if new_path not in displayed_snapshots:
                    displayed_snapshots.append(new_path)

                valid_paths = [
                    path for path in displayed_snapshots if path.exists()]
                if not valid_paths:
                    gallery_placeholder.empty()
                    return

                container = gallery_placeholder.empty()
                with container.container():
                    cols = st.columns(3)
                    for idx, path in enumerate(valid_paths):
                        try:
                            image_bytes = path.read_bytes()
                        except OSError:
                            continue
                        cols[idx % 3].image(io.BytesIO(
                            image_bytes), caption=path.name)

            with st.spinner("Đang trích snapshot..."):
                snapshot_count = snapshot_frames_with_drl(
                    feature_model=feature_model,
                    deep_q=deep_q,
                    classification_model=classification_model,
                    video_path=saved_video_path,
                    output_dir=OUTPUT_IMAGE_DIR,
                    frame_stride=int(stride),
                    on_snapshot=_update_snapshot_gallery,
                )
            st.success(f"Đã lưu {snapshot_count} ảnh vào output_img")
            # show_snapshots(OUTPUT_IMAGE_DIR)
else:
    st.info("Vui lòng tải video để bắt đầu.")
