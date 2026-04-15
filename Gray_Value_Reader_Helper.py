import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageEnhance

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from streamlit_drawable_canvas import st_canvas

    HAS_DRAWABLE_CANVAS = True
except ImportError:
    HAS_DRAWABLE_CANVAS = False


def patch_drawable_canvas_streamlit_image_api() -> bool:
    """Patch streamlit-drawable-canvas for newer Streamlit image internals."""
    try:
        import streamlit_drawable_canvas as drawable_canvas
        import streamlit.elements.image as st_image
        from streamlit.elements.lib.image_utils import image_to_url as current_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig
    except Exception:
        return False

    def legacy_image_to_url(image, width, clamp, channels, output_format, image_id):
        layout_config = LayoutConfig(width=width)
        return current_image_to_url(image, layout_config, clamp, channels, output_format, image_id)

    st_image.image_to_url = legacy_image_to_url
    drawable_canvas.st_image.image_to_url = legacy_image_to_url
    return True


CAN_USE_DRAWABLE_CANVAS = HAS_DRAWABLE_CANVAS and patch_drawable_canvas_streamlit_image_api()


@dataclass
class VideoInfo:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_s: float


@dataclass
class TemperatureCandidate:
    value: float
    raw_text: str
    method: str
    confidence: float
    processed_image: Optional[np.ndarray] = None


def save_uploaded_video(uploaded_file) -> str:
    """Save an uploaded video to a temporary file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def video_mime_type(file_name: str, uploaded_type: str = "") -> str:
    """Return a browser-friendly MIME type for Streamlit's video player."""
    if uploaded_type:
        return uploaded_type

    suffix = os.path.splitext(file_name.lower())[1]
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".avi":
        return "video/x-msvideo"
    if suffix == ".mkv":
        return "video/x-matroska"
    return "video/mp4"


def open_video(video_path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
    """Open a video with OpenCV and collect basic metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open this video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 1.0
    duration_s = frame_count / fps if frame_count > 0 else 0.0

    return cap, VideoInfo(video_path, frame_count, fps, width, height, duration_s)


def read_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Read a specific BGR frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def clamp_roi(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    """Clamp an ROI so it stays inside the image."""
    x = int(np.clip(x, 0, max(0, image_w - 1)))
    y = int(np.clip(y, 0, max(0, image_h - 1)))
    w = int(np.clip(w, 1, max(1, image_w - x)))
    h = int(np.clip(h, 1, max(1, image_h - y)))
    return x, y, w, h


def crop_temperature_roi(frame_bgr: np.ndarray, temp_roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the fixed software-overlay temperature ROI from a frame."""
    tx, ty, tw, th = temp_roi
    return frame_bgr[ty : ty + th, tx : tx + tw]


def set_roi_widget_state(key_prefix: str, roi: Tuple[int, int, int, int]) -> None:
    """Push ROI values into Streamlit widget state before numeric widgets render."""
    x, y, w, h = roi
    st.session_state[f"{key_prefix}_x"] = int(x)
    st.session_state[f"{key_prefix}_y"] = int(y)
    st.session_state[f"{key_prefix}_w"] = int(w)
    st.session_state[f"{key_prefix}_h"] = int(h)
    st.session_state[f"{key_prefix}_h_display"] = int(h)


def get_canvas_size(image_w: int, image_h: int, max_width: int = 850) -> Tuple[int, int, float]:
    """Return display size and scale factor for drawing on a large frame."""
    if image_w <= max_width:
        scale = 1.0
    else:
        scale = max_width / image_w
    canvas_w = max(1, int(round(image_w * scale)))
    canvas_h = max(1, int(round(image_h * scale)))
    return canvas_w, canvas_h, scale


def canvas_object_to_roi(
    canvas_object: dict,
    display_scale: float,
    image_w: int,
    image_h: int,
    force_square: bool,
) -> Optional[Tuple[int, int, int, int]]:
    """Convert a drawn canvas rectangle into original-frame ROI coordinates."""
    if canvas_object.get("type") != "rect":
        return None

    left = float(canvas_object.get("left", 0))
    top = float(canvas_object.get("top", 0))
    width = float(canvas_object.get("width", 0)) * float(canvas_object.get("scaleX", 1))
    height = float(canvas_object.get("height", 0)) * float(canvas_object.get("scaleY", 1))

    if abs(width) < 2 or abs(height) < 2:
        return None

    x1 = min(left, left + width) / display_scale
    y1 = min(top, top + height) / display_scale
    w = abs(width) / display_scale
    h = abs(height) / display_scale

    if force_square:
        side = max(1, int(round(min(w, h))))
        return clamp_roi(round(x1), round(y1), side, side, image_w, image_h)

    return clamp_roi(round(x1), round(y1), round(w), round(h), image_w, image_h)


def canvas_image_to_freeform_mask(
    canvas_image_data: Optional[np.ndarray],
    background_image: Image.Image,
    image_w: int,
    image_h: int,
) -> Optional[np.ndarray]:
    """Convert a drawn canvas overlay into a full-resolution boolean mask."""
    if canvas_image_data is None:
        return None

    canvas_rgb = np.asarray(canvas_image_data[:, :, :3], dtype=np.int16)
    background_rgb = np.asarray(background_image.convert("RGB"), dtype=np.int16)
    if canvas_rgb.shape[:2] != background_rgb.shape[:2]:
        background_rgb = cv2.resize(
            background_rgb.astype(np.uint8),
            (canvas_rgb.shape[1], canvas_rgb.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.int16)

    diff = np.max(np.abs(canvas_rgb - background_rgb), axis=2)
    mask_small = diff > 12
    if int(mask_small.sum()) < 10:
        return None

    mask_small = mask_small.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
    mask = cv2.resize(mask_small, (image_w, image_h), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def create_auto_pattern_mask(
    frame_bgr: np.ndarray,
    temp_roi: Tuple[int, int, int, int],
    search_roi: Optional[Tuple[int, int, int, int]],
    polarity: str,
    threshold_delta: float,
    min_area: int,
    max_components: int,
) -> np.ndarray:
    """Detect sample pattern pixels that differ from the local bright background."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

    tx, ty, tw, th = temp_roi
    valid_mask = np.ones(gray.shape, dtype=bool)
    valid_mask[ty : ty + th, tx : tx + tw] = False
    if search_roi is not None:
        sx, sy, sw, sh = search_roi
        search_mask = np.zeros(gray.shape, dtype=bool)
        search_mask[sy : sy + sh, sx : sx + sw] = True
        valid_mask &= search_mask

    background_level = float(np.median(smoothed[valid_mask])) if np.any(valid_mask) else float(np.median(smoothed))
    if polarity == "Darker than background":
        mask = smoothed < (background_level - threshold_delta)
    else:
        mask = smoothed > (background_level + threshold_delta)

    mask[~valid_mask] = False
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    kept = np.zeros_like(mask_uint8, dtype=np.uint8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    for _, label in components[:max_components]:
        kept[labels == label] = 255

    return kept > 0


def create_auto_edge_pattern_mask(
    frame_bgr: np.ndarray,
    temp_roi: Tuple[int, int, int, int],
    search_roi: Optional[Tuple[int, int, int, int]],
    edge_threshold: float,
    min_area: int,
    max_components: int,
    dilation: int,
) -> np.ndarray:
    """Detect dark ring/line edges inside the microscope bright field."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Estimate the bright microscope field so the black outside vignette is ignored.
    illumination = cv2.GaussianBlur(gray, (61, 61), 0)
    _, field_uint8 = cv2.threshold(illumination, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    field_mask = field_uint8 > 0
    if np.any(field_mask) and np.any(~field_mask) and np.mean(gray[field_mask]) < np.mean(gray[~field_mask]):
        field_mask = ~field_mask

    tx, ty, tw, th = temp_roi
    field_mask[ty : ty + th, tx : tx + tw] = False
    if search_roi is not None:
        sx, sy, sw, sh = search_roi
        search_mask = np.zeros(gray.shape, dtype=bool)
        search_mask[sy : sy + sh, sx : sx + sw] = True
        field_mask &= search_mask

    local_background = cv2.GaussianBlur(gray, (31, 31), 0)
    dark_response = cv2.subtract(local_background, gray)
    dark_line_mask = (dark_response > edge_threshold) & field_mask

    edges = cv2.Canny(gray, 35, 110) > 0
    mask = dark_line_mask & edges

    mask_uint8 = mask.astype(np.uint8) * 255
    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    kept = np.zeros_like(mask_uint8, dtype=np.uint8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    for _, label in components[:max_components]:
        kept[labels == label] = 255

    return kept > 0


def remove_small_mask_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny connected components from a binary mask."""
    if min_area <= 1 or not np.any(mask):
        return mask

    mask_uint8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            kept[labels == label] = True
    return kept


def edge_contour_length(mask: np.ndarray) -> float:
    """Measure edge masks by contour length, which is stable against line thickness."""
    if not np.any(mask):
        return 0.0
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def effective_pattern_area(mask: np.ndarray, optical_roi_mode: str) -> float:
    """
    Return a stable area-like signal for the selected mask.

    Edge masks are measured as contour length so small changes in detected edge
    thickness or scattered noise do not become large curve changes.
    """
    if mask is None or not np.any(mask):
        return 0.0

    cleaned = remove_small_mask_components(mask, min_area=8)
    if optical_roi_mode == "Auto dark-edge ROI":
        return edge_contour_length(cleaned)

    return float(cleaned.sum())


def canvas_roi_selector(
    frame_bgr: np.ndarray,
    image_w: int,
    image_h: int,
) -> None:
    """Optional drag-to-select ROI helper powered by streamlit-drawable-canvas."""
    st.subheader("Drag-select ROI")

    if not HAS_DRAWABLE_CANVAS:
        st.info(
            "Drag selection is optional. Install it with: "
            "`python -m pip install streamlit-drawable-canvas`"
        )
        return
    if not CAN_USE_DRAWABLE_CANVAS:
        st.warning(
            "The drag-selection component is installed, but it is not compatible with this Streamlit version. "
            "Use the numeric ROI controls on the right."
        )
        return

    target = st.radio(
        "ROI to draw",
        ["Auto-detection search ROI", "Free-drawn optical ROI", "Grayscale ROI", "Temperature OCR ROI"],
        horizontal=True,
        key="canvas_target_roi",
    )
    st.caption("Draw a yellow search box first, then auto-detect dark edges only inside that box.")

    canvas_w, canvas_h, scale = get_canvas_size(image_w, image_h, max_width=620)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas_image = Image.fromarray(frame_rgb).resize((canvas_w, canvas_h))

    if target == "Auto-detection search ROI":
        stroke_color = "#ffd000"
    elif target == "Temperature OCR ROI":
        stroke_color = "#ff5000"
    else:
        stroke_color = "#00ff00"
    fill_color = "rgba(0, 255, 0, 0.25)" if target == "Free-drawn optical ROI" else "rgba(0, 0, 0, 0)"
    drawing_mode = "polygon" if target == "Free-drawn optical ROI" else "rect"
    try:
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=2,
            stroke_color=stroke_color,
            background_image=canvas_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            key=f"roi_canvas_{target}_{st.session_state.get('preview_frame_index', 0)}",
        )
    except Exception as exc:
        st.warning(f"Drag selection could not start: {exc}")
        return

    if st.button("Apply drawn ROI", use_container_width=True):
        objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
        if target == "Free-drawn optical ROI":
            mask = canvas_image_to_freeform_mask(canvas_result.image_data, canvas_image, image_w, image_h)
            if mask is None:
                st.warning("No free-drawn region was detected. Draw a closed polygon around the sample first.")
                return
            st.session_state.freeform_optical_mask = mask
            st.session_state.optical_roi_mode = "Free-drawn ROI"
            st.success(f"Applied free-drawn optical ROI with {int(mask.sum()):,} pixels.")
            st.rerun()
            return

        rectangles = [obj for obj in objects if obj.get("type") == "rect"]
        if not rectangles:
            st.warning("No rectangle was drawn. Drag on the image first.")
            return

        selected = rectangles[-1]
        if target == "Auto-detection search ROI":
            roi = canvas_object_to_roi(selected, scale, image_w, image_h, force_square=False)
            if roi is None:
                st.warning("The drawn search ROI was too small.")
                return
            set_roi_widget_state("search_roi", roi)
            st.success(f"Applied search ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        elif target == "Grayscale ROI":
            roi = canvas_object_to_roi(selected, scale, image_w, image_h, force_square=True)
            if roi is None:
                st.warning("The drawn grayscale ROI was too small.")
                return
            set_roi_widget_state("gray_roi", roi)
            st.success(f"Applied grayscale ROI: x={roi[0]}, y={roi[1]}, size={roi[2]} px")
        else:
            roi = canvas_object_to_roi(selected, scale, image_w, image_h, force_square=False)
            if roi is None:
                st.warning("The drawn temperature ROI was too small.")
                return
            set_roi_widget_state("temp_roi", roi)
            st.success(f"Applied temperature ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        st.rerun()


def simplified_canvas_roi_selector(
    frame_bgr: np.ndarray,
    image_w: int,
    image_h: int,
    display_image_rgb: Optional[np.ndarray] = None,
    max_preview_width: int = 560,
) -> None:
    """Drag-select only the two ROIs used by the simplified workflow."""
    if not HAS_DRAWABLE_CANVAS:
        st.info("Install `streamlit-drawable-canvas` to draw ROI directly on the preview image.")
        return
    if not CAN_USE_DRAWABLE_CANVAS:
        st.warning("Drag selection is not compatible with this Streamlit version. Use the numeric ROI controls.")
        return

    target = st.radio(
        "Draw ROI on preview",
        ["Pattern search ROI", "Temperature OCR ROI"],
        horizontal=True,
        key="simple_canvas_target_roi",
    )
    if target == "Temperature OCR ROI":
        st.info("Box the full temperature label, including `Temp`, the number, and `°C`; avoid nearby unrelated text.")
    if st.session_state.get("simple_canvas_last_target") != target:
        st.session_state.simple_canvas_last_target = target
        st.session_state.simple_canvas_clear_counter = st.session_state.get("simple_canvas_clear_counter", 0) + 1
        st.rerun()

    canvas_w, canvas_h, scale = get_canvas_size(image_w, image_h, max_width=max_preview_width)
    frame_rgb = display_image_rgb.copy() if display_image_rgb is not None else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    canvas_image = Image.fromarray(np.ascontiguousarray(frame_rgb), mode="RGB").resize(
        (canvas_w, canvas_h),
        resample=Image.Resampling.LANCZOS,
    )
    stroke_color = "#ffd000" if target == "Pattern search ROI" else "#ff5000"
    canvas_refresh_col, canvas_help_col = st.columns([0.34, 0.66])
    with canvas_refresh_col:
        if st.button("Refresh preview canvas", use_container_width=True, key="simple_refresh_canvas"):
            st.session_state.simple_canvas_clear_counter = st.session_state.get("simple_canvas_clear_counter", 0) + 1
            st.rerun()
    with canvas_help_col:
        st.caption("Use refresh if the drawing area appears blank.")
    canvas_key = (
        f"simple_roi_canvas_{target}_"
        f"{st.session_state.get('preview_frame_index', 0)}_"
        f"{st.session_state.get('simple_canvas_clear_counter', 0)}"
    )
    try:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color=stroke_color,
            background_image=canvas_image,
            initial_drawing={"version": "4.4.0", "objects": []},
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=canvas_key,
        )
    except Exception as exc:
        st.warning(f"Drag selection could not start: {exc}")
        return

    def apply_latest_rectangle(rectangles: List[dict], rerun_after_apply: bool) -> None:
        roi = canvas_object_to_roi(rectangles[-1], scale, image_w, image_h, force_square=False)
        if roi is None:
            st.warning("The drawn ROI was too small.")
            return
        if target == "Pattern search ROI":
            st.session_state.pending_simple_search_roi = roi
        else:
            st.session_state.pending_simple_temp_roi = roi
            st.session_state.pop("simple_temperature_parse_mode", None)
            st.session_state.pop("simple_preview_temperature", None)
        st.session_state.simple_canvas_clear_counter = st.session_state.get("simple_canvas_clear_counter", 0) + 1
        if rerun_after_apply:
            st.rerun()

    objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
    rectangles = [obj for obj in objects if obj.get("type") == "rect"]
    if len(rectangles) > 1:
        apply_latest_rectangle(rectangles, rerun_after_apply=True)
        return

    if st.button("Apply drawn ROI", use_container_width=True, key="simple_apply_drawn_roi"):
        if not rectangles:
            st.warning("No rectangle was drawn. Drag on the image first.")
            return
        apply_latest_rectangle(rectangles, rerun_after_apply=False)
        st.success("Applied the latest ROI. The drawing canvas will clear on refresh.")
        st.rerun()


def plotly_box_to_roi(selection: Any, scale: float, image_w: int, image_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Convert a Plotly box-selection event into original-frame ROI coordinates."""
    if not selection:
        return None
    if hasattr(selection, "to_dict"):
        selection = selection.to_dict()

    box_data = None
    if isinstance(selection, dict):
        selection = selection.get("selection", selection)
        box_data = selection.get("box") or selection.get("range") if isinstance(selection, dict) else None
    if isinstance(box_data, list):
        box_data = box_data[-1] if box_data else None
    if not isinstance(box_data, dict):
        return None

    if "x" in box_data and "y" in box_data:
        x_values = box_data["x"]
        y_values = box_data["y"]
    elif "x0" in box_data and "x1" in box_data and "y0" in box_data and "y1" in box_data:
        x_values = [box_data["x0"], box_data["x1"]]
        y_values = [box_data["y0"], box_data["y1"]]
    else:
        return None

    if not isinstance(x_values, (list, tuple)) or not isinstance(y_values, (list, tuple)):
        return None
    if len(x_values) < 2 or len(y_values) < 2:
        return None

    try:
        x1, x2 = float(x_values[0]), float(x_values[1])
        y1, y2 = float(y_values[0]), float(y_values[1])
    except (TypeError, ValueError):
        return None

    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width < 2 or height < 2:
        return None

    return clamp_roi(
        int(round(left / scale)),
        int(round(top / scale)),
        int(round(width / scale)),
        int(round(height / scale)),
        image_w,
        image_h,
    )


def plotly_roi_selector(
    frame_bgr: np.ndarray,
    image_w: int,
    image_h: int,
    display_image_rgb: Optional[np.ndarray] = None,
    max_preview_width: int = 560,
) -> None:
    """Select simplified workflow ROIs with a native Plotly box-selection preview."""
    if not HAS_PLOTLY:
        st.warning("Plotly is not installed, so the clickable preview selector is unavailable.")
        simplified_canvas_roi_selector(frame_bgr, image_w, image_h, display_image_rgb, max_preview_width)
        return

    target = st.radio(
        "Draw ROI on preview",
        ["Pattern search ROI", "Temperature OCR ROI"],
        horizontal=True,
        key="simple_plotly_target_roi",
    )
    if target == "Temperature OCR ROI":
        st.info("Box the full temperature label, including `Temp`, the number, and `°C`; avoid nearby unrelated text.")

    canvas_w, canvas_h, scale = get_canvas_size(image_w, image_h, max_width=max_preview_width)
    frame_rgb = display_image_rgb.copy() if display_image_rgb is not None else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    preview_rgb = np.asarray(
        Image.fromarray(np.ascontiguousarray(frame_rgb), mode="RGB").resize(
            (canvas_w, canvas_h),
            resample=Image.Resampling.LANCZOS,
        )
    )

    grid_step = max(8, min(canvas_w, canvas_h) // 42)
    xs, ys = np.meshgrid(np.arange(0, canvas_w, grid_step), np.arange(0, canvas_h, grid_step))
    fig = go.Figure()
    fig.add_trace(go.Image(z=preview_rgb, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=xs.ravel(),
            y=ys.ravel(),
            mode="markers",
            marker={"size": 4, "opacity": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        dragmode="select",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=canvas_w,
        height=canvas_h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        selectdirection="any",
    )
    fig.update_xaxes(visible=False, range=[0, canvas_w], fixedrange=False, constrain="domain")
    fig.update_yaxes(visible=False, range=[canvas_h, 0], fixedrange=False, scaleanchor="x", scaleratio=1)
    config = {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
        "displaylogo": False,
    }
    selection = st.plotly_chart(
        fig,
        use_container_width=False,
        key=f"simple_plotly_roi_{target}_{st.session_state.get('preview_frame_index', 0)}",
        on_select="rerun",
        selection_mode=("box",),
        config=config,
    )

    st.caption("Drag a rectangle on the image. If the toolbar is active, choose the box-select icon.")
    selected_roi = plotly_box_to_roi(selection, scale, image_w, image_h)
    if selected_roi is not None:
        st.session_state.simple_plotly_selected_roi = selected_roi
        st.success(f"Selected ROI: x={selected_roi[0]}, y={selected_roi[1]}, w={selected_roi[2]}, h={selected_roi[3]}")
    else:
        st.session_state.pop("simple_plotly_selected_roi", None)

    if st.button("Apply selected ROI", use_container_width=True, key="simple_apply_plotly_roi"):
        roi = st.session_state.get("simple_plotly_selected_roi")
        if roi is None:
            st.warning("Drag a rectangle on the preview image first.")
            return
        if target == "Pattern search ROI":
            st.session_state.pending_simple_search_roi = roi
        else:
            st.session_state.pending_simple_temp_roi = roi
            st.session_state.pop("simple_temperature_parse_mode", None)
            st.session_state.pop("simple_preview_temperature", None)
        st.session_state.pop("simple_plotly_selected_roi", None)
        st.rerun()


def draw_rois(
    frame_bgr: np.ndarray,
    gray_roi: Tuple[int, int, int, int],
    temp_roi: Tuple[int, int, int, int],
    optical_roi_mode: str = "Square ROI",
    ring_roi: Optional[Tuple[int, int, int, int]] = None,
    freeform_mask: Optional[np.ndarray] = None,
    auto_pattern_mask: Optional[np.ndarray] = None,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
    show_temp_roi: bool = True,
    show_gray_roi: bool = True,
) -> np.ndarray:
    """Draw grayscale and temperature ROIs on a preview image."""
    preview = frame_bgr.copy()
    gx, gy, gw, gh = gray_roi
    tx, ty, tw, th = temp_roi
    if search_roi is not None:
        sx, sy, sw, sh = search_roi
        cv2.rectangle(preview, (sx, sy), (sx + sw, sy + sh), (0, 210, 255), 2)
        cv2.putText(
            preview,
            "Search ROI",
            (sx, max(14, sy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )

    active_mask = None
    active_label = ""
    if optical_roi_mode == "Free-drawn ROI" and freeform_mask is not None and np.any(freeform_mask):
        active_mask = freeform_mask
        active_label = "Free ROI"
    elif optical_roi_mode in {"Auto-detected pattern ROI", "Auto dark-edge ROI"} and auto_pattern_mask is not None and np.any(auto_pattern_mask):
        active_mask = auto_pattern_mask
        active_label = "Auto edge ROI" if optical_roi_mode == "Auto dark-edge ROI" else "Auto pattern ROI"

    if active_mask is not None:
        mask_uint8 = active_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        overlay = preview.copy()
        overlay[active_mask] = (0.65 * overlay[active_mask] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        preview = overlay
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(
                preview,
                active_label,
                (x, max(14, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    elif optical_roi_mode == "Dual-ring ROI" and ring_roi is not None:
        cx, cy, inner_r, outer_r = ring_roi
        cv2.circle(preview, (cx, cy), outer_r, (0, 255, 0), 2)
        cv2.circle(preview, (cx, cy), inner_r, (0, 180, 0), 2)
        cv2.circle(preview, (cx, cy), 2, (0, 255, 0), -1)
        cv2.putText(
            preview,
            "Dual-ring ROI",
            (max(0, cx - outer_r), max(14, cy - outer_r - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    elif show_gray_roi:
        cv2.rectangle(preview, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        cv2.putText(
            preview,
            "Gray ROI",
            (gx, max(14, gy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if show_temp_roi:
        cv2.rectangle(preview, (tx, ty), (tx + tw, ty + th), (255, 80, 0), 2)
        cv2.putText(
            preview,
            "Temp OCR ROI",
            (tx, max(14, ty - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def clamp_ring_roi(
    cx: int,
    cy: int,
    inner_r: int,
    outer_r: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    """Clamp a dual-ring ROI to valid image-centered circle parameters."""
    cx = int(np.clip(cx, 0, max(0, image_w - 1)))
    cy = int(np.clip(cy, 0, max(0, image_h - 1)))
    max_radius = max(1, int(min(cx, cy, image_w - 1 - cx, image_h - 1 - cy)))
    outer_r = int(np.clip(outer_r, 2, max_radius))
    inner_r = int(np.clip(inner_r, 1, max(1, outer_r - 1)))
    return cx, cy, inner_r, outer_r


def calculate_optical_signal(
    gray_frame: np.ndarray,
    optical_roi_mode: str,
    gray_roi: Tuple[int, int, int, int],
    ring_roi: Tuple[int, int, int, int],
    freeform_mask: Optional[np.ndarray] = None,
    auto_pattern_mask: Optional[np.ndarray] = None,
    temp_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """Calculate raw grayscale and dissolution metrics from square or dual-ring ROI."""
    image_area = int(gray_frame.shape[0] * gray_frame.shape[1])
    if optical_roi_mode in {"Auto-detected pattern ROI", "Auto dark-edge ROI"}:
        if auto_pattern_mask is None:
            return {
                "gray_value": np.nan,
                "sample_gray": np.nan,
                "background_gray": np.nan,
                "darkness_value": np.nan,
                "dissolution_signal": np.nan,
                "background_corrected_darkness": np.nan,
                "pattern_area_px": np.nan,
                "pattern_area_fraction": np.nan,
            }
        sample_mask = auto_pattern_mask
        pattern_area_px = effective_pattern_area(sample_mask, optical_roi_mode)
        if pattern_area_px == 0:
            return {
                "gray_value": np.nan,
                "sample_gray": np.nan,
                "background_gray": np.nan,
                "darkness_value": np.nan,
                "dissolution_signal": 0.0,
                "background_corrected_darkness": 0.0,
                "pattern_area_px": 0.0,
                "pattern_area_fraction": 0.0,
            }
        background_mask = ~sample_mask.copy()
        if temp_roi is not None:
            tx, ty, tw, th = temp_roi
            background_mask[ty : ty + th, tx : tx + tw] = False
        sample_gray = float(np.mean(gray_frame[sample_mask]))
        background_gray = float(np.median(gray_frame[background_mask])) if np.any(background_mask) else np.nan
        corrected_darkness = background_gray - sample_gray if not np.isnan(background_gray) else np.nan
        if np.isnan(corrected_darkness):
            corrected_darkness = 255.0 - sample_gray
        return {
            "gray_value": sample_gray,
            "sample_gray": sample_gray,
            "background_gray": background_gray,
            "darkness_value": 255.0 - sample_gray,
            "dissolution_signal": corrected_darkness,
            "background_corrected_darkness": corrected_darkness,
            "pattern_area_px": pattern_area_px,
            "pattern_area_fraction": float(pattern_area_px / image_area) if image_area else np.nan,
        }

    if optical_roi_mode == "Free-drawn ROI":
        sample_mask = freeform_mask if optical_roi_mode == "Free-drawn ROI" else auto_pattern_mask
        if sample_mask is not None and np.any(sample_mask):
            pattern_area_px = int(sample_mask.sum())
            background_mask = ~sample_mask.copy()
            if temp_roi is not None:
                tx, ty, tw, th = temp_roi
                background_mask[ty : ty + th, tx : tx + tw] = False
            sample_gray = float(np.mean(gray_frame[sample_mask]))
            background_gray = float(np.median(gray_frame[background_mask])) if np.any(background_mask) else np.nan
            corrected_darkness = background_gray - sample_gray if not np.isnan(background_gray) else np.nan
            if np.isnan(corrected_darkness):
                corrected_darkness = 255.0 - sample_gray
            return {
                "gray_value": sample_gray,
                "sample_gray": sample_gray,
                "background_gray": background_gray,
                "darkness_value": 255.0 - sample_gray,
                "dissolution_signal": corrected_darkness,
                "background_corrected_darkness": corrected_darkness,
                "pattern_area_px": float(pattern_area_px),
                "pattern_area_fraction": float(pattern_area_px / image_area) if image_area else np.nan,
            }

    if optical_roi_mode == "Free-drawn ROI" and freeform_mask is not None and np.any(freeform_mask):
        sample_gray = float(np.mean(gray_frame[freeform_mask]))
        darkness_value = 255.0 - sample_gray
        pattern_area_px = int(freeform_mask.sum())
        return {
            "gray_value": sample_gray,
            "sample_gray": sample_gray,
            "background_gray": np.nan,
            "darkness_value": darkness_value,
            "dissolution_signal": darkness_value,
            "background_corrected_darkness": np.nan,
            "pattern_area_px": float(pattern_area_px),
            "pattern_area_fraction": float(pattern_area_px / image_area) if image_area else np.nan,
        }

    if optical_roi_mode == "Dual-ring ROI":
        cx, cy, inner_r, outer_r = ring_roi
        yy, xx = np.ogrid[: gray_frame.shape[0], : gray_frame.shape[1]]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inner_mask = dist2 <= inner_r**2
        ring_mask = (dist2 > inner_r**2) & (dist2 <= outer_r**2)
        pattern_area_px = int(inner_mask.sum())

        sample_gray = float(np.mean(gray_frame[inner_mask])) if np.any(inner_mask) else np.nan
        background_gray = float(np.mean(gray_frame[ring_mask])) if np.any(ring_mask) else np.nan
        corrected_darkness = background_gray - sample_gray if not np.isnan(sample_gray) and not np.isnan(background_gray) else np.nan
        return {
            "gray_value": sample_gray,
            "sample_gray": sample_gray,
            "background_gray": background_gray,
            "darkness_value": 255.0 - sample_gray if not np.isnan(sample_gray) else np.nan,
            "dissolution_signal": corrected_darkness,
            "background_corrected_darkness": corrected_darkness,
            "pattern_area_px": float(pattern_area_px),
            "pattern_area_fraction": float(pattern_area_px / image_area) if image_area else np.nan,
        }

    gx, gy, gw, gh = gray_roi
    gray_crop = gray_frame[gy : gy + gh, gx : gx + gw]
    gray_value = float(np.mean(gray_crop)) if gray_crop.size else np.nan
    darkness_value = 255.0 - gray_value if not np.isnan(gray_value) else np.nan
    pattern_area_px = int(gw * gh)
    return {
        "gray_value": gray_value,
        "sample_gray": gray_value,
        "background_gray": np.nan,
        "darkness_value": darkness_value,
        "dissolution_signal": darkness_value,
        "background_corrected_darkness": np.nan,
        "pattern_area_px": float(pattern_area_px),
        "pattern_area_fraction": float(pattern_area_px / image_area) if image_area else np.nan,
    }


def preprocess_temperature_roi(
    roi_bgr: np.ndarray,
    scale: int = 3,
    blur: bool = False,
    threshold_method: str = "None",
    invert: bool = False,
    contrast_factor: float = 2.0,
) -> np.ndarray:
    """Prepare a temperature text ROI for OCR."""
    if threshold_method == "Contrast only":
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        if scale > 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(rgb)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        return np.array(image)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if threshold_method == "None":
        image = Image.fromarray(gray)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        gray = np.array(image)
    else:
        gray = cv2.equalizeHist(gray)

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold_method == "Adaptive":
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
    elif threshold_method == "Otsu":
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        processed = gray

    if invert:
        processed = cv2.bitwise_not(processed)

    return processed


def normalize_template_roi(roi_bgr: np.ndarray, output_size: Tuple[int, int] = (320, 80)) -> np.ndarray:
    """Normalize a software overlay ROI for template similarity matching."""
    processed = preprocess_temperature_roi(
        roi_bgr,
        scale=3,
        blur=False,
        threshold_method="Contrast only",
        invert=False,
        contrast_factor=2.0,
    )
    if processed.ndim == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    processed = cv2.resize(processed, output_size, interpolation=cv2.INTER_AREA)
    processed = processed.astype(np.float32)
    mean = float(processed.mean())
    std = float(processed.std())
    if std < 1e-6:
        return processed * 0
    return (processed - mean) / std


def template_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized similarity between two same-sized template images."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(a * b) / denom)


def normalize_temperature_text(text: str) -> str:
    """Normalize OCR temperature text before regex parsing."""
    cleaned = (text or "").replace(",", ".")
    cleaned = cleaned.replace("℃", "°C").replace("º", "°")
    for minus_like in ("−", "–", "—", "﹣", "－"):
        cleaned = cleaned.replace(minus_like, "-")
    cleaned = re.sub(r"([+-])\s+(\d)", r"\1\2", cleaned)
    return cleaned


def get_tesseract_status(tesseract_cmd: str = "") -> Tuple[bool, str]:
    """Check whether the Tesseract executable is available."""
    if tesseract_cmd.strip():
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd.strip().strip('"')

    executable = pytesseract.pytesseract.tesseract_cmd
    if executable == "tesseract":
        executable = shutil.which("tesseract") or "tesseract"

    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract ready: {version}"
    except Exception as exc:
        return False, f"Tesseract is not available: {exc}"


def default_tesseract_path() -> str:
    """Return a likely Tesseract path for local Windows or hosted Linux."""
    path_from_system = shutil.which("tesseract")
    if path_from_system:
        return path_from_system

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def parse_temperature_token(token: str, assume_one_decimal: bool) -> float:
    """Parse an OCR number token, optionally restoring one missing decimal place."""
    token = normalize_temperature_text(token).strip().replace(" ", "")
    if not token:
        return np.nan

    if assume_one_decimal and "." not in token:
        sign = ""
        digits = token
        if digits[0] in "+-":
            sign = digits[0]
            digits = digits[1:]
        if digits.isdigit() and len(digits) >= 2:
            token = f"{sign}{digits[:-1]}.{digits[-1]}"

    try:
        return float(token)
    except ValueError:
        return np.nan


def extract_temp_between_label_and_unit(text: str, assume_one_decimal: bool) -> float:
    """Strictly extract the number between a Temp label and a Celsius unit."""
    cleaned = normalize_temperature_text(text)

    # Tesseract may add or remove spaces, and sometimes reads "e" as "3".
    temp_label = r"T\s*[e3]\s*m\s*p"
    number = r"([-+]?\s*(?:\d+\.\d+|\d+|\.\d+))"

    patterns = [
        rf"{temp_label}[\s:=]*{number}\s*°\s*C",
        rf"{temp_label}[\s:=]*{number}\s*C",
        rf"{temp_label}[\s:=]*{number}\s*(?:deg|degree)\s*C",
        rf"{temp_label}[\s:=]*{number}\s*°",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = parse_temperature_token(match.group(1), assume_one_decimal)
            if not np.isnan(value):
                return value

    return np.nan


def extract_temperature(
    text: str,
    parse_mode: str = "Between Temp and C",
    assume_one_decimal: bool = False,
) -> float:
    """Extract the sample temperature from OCR text."""
    cleaned = normalize_temperature_text(text)
    number = r"([-+]?\s*(?:\d+\.\d+|\d+|\.\d+))"

    if parse_mode == "Between Temp and C":
        return extract_temp_between_label_and_unit(cleaned, assume_one_decimal)

    if parse_mode == "Sample Temp after label":
        between_match = re.search(
            rf"T\s*[e3]\s*m\s*p[^\d+\-.]{{0,20}}{number}\s*(?:deg|degree|o|0)?\s*C",
            cleaned,
            flags=re.IGNORECASE,
        )
        if between_match:
            value = parse_temperature_token(between_match.group(1), assume_one_decimal)
            if not np.isnan(value):
                return value

    if parse_mode == "Sample Temp after label":
        temp_match = re.search(
            rf"\bT\s*e?\s*m\s*p\b[^\d+\-.]{{0,12}}{number}",
            cleaned,
            flags=re.IGNORECASE,
        )
        if temp_match:
            value = parse_temperature_token(temp_match.group(1), assume_one_decimal)
            if not np.isnan(value):
                return value

    if parse_mode in {"First value followed by C", "Sample Temp after label"}:
        c_match = re.search(
            rf"{number}\s*(?:°|o|0)?\s*C\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if c_match:
            value = parse_temperature_token(c_match.group(1), assume_one_decimal)
            if not np.isnan(value):
                return value

    match = re.search(r"[-+]?\s*(?:\d+\.\d+|\d+|\.\d+)", cleaned)
    if not match:
        return np.nan
    return parse_temperature_token(match.group(0), assume_one_decimal)


def ocr_temperature(
    processed_roi: np.ndarray,
    psm: int = 7,
    parse_mode: str = "Between Temp and C",
    assume_one_decimal: bool = False,
) -> Tuple[str, float]:
    """Run pytesseract OCR and parse a temperature value."""
    if parse_mode == "Between Temp and C":
        whitelist = "0123456789.-+−–—﹣－TempCtempco°"
    elif parse_mode == "Only sample number in ROI":
        whitelist = "0123456789.-+−–—﹣－"
    elif parse_mode == "Sample Temp after label":
        whitelist = "0123456789.-+−–—﹣－TempCtempco"
    elif parse_mode == "First value followed by C":
        whitelist = "0123456789.-+−–—﹣－Cco"
    else:
        whitelist = "0123456789.-+−–—﹣－"

    config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
    try:
        text = pytesseract.image_to_string(processed_roi, config=config)
    except Exception as exc:
        return f"OCR_ERROR: {exc}", np.nan
    return text.strip(), extract_temperature(text, parse_mode=parse_mode, assume_one_decimal=assume_one_decimal)


def build_ocr_candidates(
    roi_bgr: np.ndarray,
    scale: int,
    blur: bool,
    threshold_method: str,
    invert: bool,
    contrast_factor: float,
) -> list[Tuple[str, np.ndarray]]:
    """Create several OCR-ready images so difficult video overlays have fallbacks."""
    candidates = [
        (
            f"selected {threshold_method}",
            preprocess_temperature_roi(
                roi_bgr,
                scale=scale,
                blur=blur,
                threshold_method=threshold_method,
                invert=invert,
                contrast_factor=contrast_factor,
            ),
        )
    ]

    fallback_settings = [
        ("rgb contrast only", False, "Contrast only", False),
        ("gray contrast only", False, "None", False),
        ("otsu", True, "Otsu", False),
        ("otsu inverted", True, "Otsu", True),
        ("adaptive", True, "Adaptive", False),
        ("adaptive inverted", True, "Adaptive", True),
    ]
    for label, fallback_blur, fallback_threshold, fallback_invert in fallback_settings:
        candidates.append(
            (
                label,
                preprocess_temperature_roi(
                    roi_bgr,
                    scale=scale,
                    blur=fallback_blur,
                    threshold_method=fallback_threshold,
                    invert=fallback_invert,
                    contrast_factor=contrast_factor,
                ),
            )
        )

    return candidates


def run_ocr_candidates(
    roi_bgr: np.ndarray,
    scale: int,
    blur: bool,
    threshold_method: str,
    invert: bool,
    contrast_factor: float,
    psm: int,
    parse_mode: str,
    assume_one_decimal: bool,
    use_all_preprocessing: bool,
) -> List[TemperatureCandidate]:
    """Run OCR over one or many overlay-specific preprocessing candidates."""
    if use_all_preprocessing:
        image_candidates = build_ocr_candidates(roi_bgr, scale, blur, threshold_method, invert, contrast_factor)
        psm_candidates = list(dict.fromkeys([psm, 7, 8, 6]))
    else:
        image_candidates = [
            (
                f"simple {threshold_method}",
                preprocess_temperature_roi(
                    roi_bgr,
                    scale=scale,
                    blur=blur,
                    threshold_method=threshold_method,
                    invert=invert,
                    contrast_factor=contrast_factor,
                ),
            )
        ]
        psm_candidates = [psm]

    candidates: List[TemperatureCandidate] = []
    for method, processed in image_candidates:
        for candidate_psm in psm_candidates:
            raw_text, value = ocr_temperature(
                processed,
                psm=candidate_psm,
                parse_mode=parse_mode,
                assume_one_decimal=assume_one_decimal,
            )
            has_value = not np.isnan(value)
            has_temp = bool(re.search(r"T\s*[e3]\s*m\s*p", raw_text or "", flags=re.IGNORECASE))
            has_c = bool(re.search(r"(?:°\s*)?C|℃", raw_text or "", flags=re.IGNORECASE))
            confidence = 0.15
            if has_value:
                confidence += 0.45
            if parse_mode == "Between Temp and C" and has_temp and has_c:
                confidence += 0.30
            elif has_temp or has_c:
                confidence += 0.10
            if raw_text and not raw_text.startswith("OCR_ERROR"):
                confidence += min(0.10, len(raw_text.strip()) / 80)
            candidates.append(
                TemperatureCandidate(
                    value=value,
                    raw_text=raw_text,
                    method=f"OCR {method}, psm {candidate_psm}",
                    confidence=float(np.clip(confidence, 0, 1)),
                    processed_image=processed,
                )
            )
    return candidates


def run_template_matching(
    roi_bgr: np.ndarray,
    templates: List[Dict[str, Any]],
    threshold: float,
) -> List[TemperatureCandidate]:
    """Match the current ROI against manually labeled software-overlay templates."""
    if not templates:
        return []

    normalized = normalize_template_roi(roi_bgr)
    candidates: List[TemperatureCandidate] = []
    for template in templates:
        template_image = template.get("image")
        value = template.get("value", np.nan)
        if template_image is None or pd.isna(value):
            continue
        score = template_similarity(normalized, template_image)
        if score >= threshold:
            candidates.append(
                TemperatureCandidate(
                    value=float(value),
                    raw_text=f"manual template {value:g} C",
                    method="Template match",
                    confidence=float(np.clip(score, 0, 1)),
                    processed_image=(np.clip(normalized * 45 + 128, 0, 255)).astype(np.uint8),
                )
            )
    return candidates


def score_temperature_candidates(
    candidates: List[TemperatureCandidate],
    previous_value: float,
    min_temp: float,
    max_temp: float,
    max_jump: float,
    expected_trend: str,
) -> TemperatureCandidate:
    """Choose the candidate that best fits format, bounds, and recent temporal behavior."""
    if not candidates:
        return TemperatureCandidate(np.nan, "", "No candidate", 0.0, None)

    scored: List[TemperatureCandidate] = []
    values = [candidate.value for candidate in candidates if not np.isnan(candidate.value)]
    for candidate in candidates:
        if np.isnan(candidate.value):
            scored.append(candidate)
            continue

        score = candidate.confidence
        if min_temp <= candidate.value <= max_temp:
            score += 0.20
        else:
            score -= 0.50

        if not pd.isna(previous_value):
            jump = candidate.value - previous_value
            if max_jump > 0 and abs(jump) <= max_jump:
                score += 0.15
            elif max_jump > 0:
                score -= min(0.60, abs(jump) / max_jump * 0.20)

            if expected_trend == "Monotonic cooling" and jump <= 0.05:
                score += 0.10
            elif expected_trend == "Monotonic cooling" and jump > max(0.1, max_jump * 0.25):
                score -= 0.25
            elif expected_trend == "Monotonic warming" and jump >= -0.05:
                score += 0.10
            elif expected_trend == "Monotonic warming" and jump < -max(0.1, max_jump * 0.25):
                score -= 0.25

        agreement_count = sum(1 for value in values if abs(value - candidate.value) <= 0.05)
        if agreement_count > 1:
            score += min(0.20, 0.05 * agreement_count)

        scored.append(
            TemperatureCandidate(
                value=candidate.value,
                raw_text=candidate.raw_text,
                method=candidate.method,
                confidence=float(np.clip(score, 0, 1)),
                processed_image=candidate.processed_image,
            )
        )

    return max(scored, key=lambda item: item.confidence)


def robust_ocr_temperature(
    roi_bgr: np.ndarray,
    scale: int,
    blur: bool,
    threshold_method: str,
    invert: bool,
    contrast_factor: float,
    psm: int,
    parse_mode: str,
    assume_one_decimal: bool,
    templates: Optional[List[Dict[str, Any]]] = None,
    template_threshold: float = 0.92,
    previous_value: float = np.nan,
    min_temp: float = -100.0,
    max_temp: float = 200.0,
    max_jump: float = 5.0,
    expected_trend: str = "Mostly smooth",
    use_all_preprocessing: bool = False,
) -> Tuple[str, float, str, float, np.ndarray, List[TemperatureCandidate]]:
    """Hybrid recognition using structured OCR plus optional template matching."""
    ocr_candidates = run_ocr_candidates(
        roi_bgr=roi_bgr,
        scale=scale,
        blur=blur,
        threshold_method=threshold_method,
        invert=invert,
        contrast_factor=contrast_factor,
        psm=psm,
        parse_mode=parse_mode,
        assume_one_decimal=assume_one_decimal,
        use_all_preprocessing=use_all_preprocessing,
    )
    template_candidates = run_template_matching(roi_bgr, templates or [], template_threshold)
    best = score_temperature_candidates(
        candidates=ocr_candidates + template_candidates,
        previous_value=previous_value,
        min_temp=min_temp,
        max_temp=max_temp,
        max_jump=max_jump,
        expected_trend=expected_trend,
    )

    processed_image = best.processed_image
    if processed_image is None:
        processed_image = preprocess_temperature_roi(
            roi_bgr,
            scale=scale,
            blur=blur,
            threshold_method=threshold_method,
            invert=invert,
            contrast_factor=contrast_factor,
        )
    return best.raw_text, best.value, best.method, best.confidence, processed_image, ocr_candidates + template_candidates


def robust_auto_ocr_temperature(
    roi_bgr: np.ndarray,
    scale: int,
    blur: bool,
    threshold_method: str,
    invert: bool,
    contrast_factor: float,
    psm: int,
    assume_one_decimal: bool,
    templates: Optional[List[Dict[str, Any]]] = None,
    template_threshold: float = 0.92,
    previous_value: float = np.nan,
    min_temp: float = -100.0,
    max_temp: float = 200.0,
    max_jump: float = 0.0,
    expected_trend: str = "Mostly smooth",
    use_all_preprocessing: bool = False,
) -> Tuple[str, float, str, float, np.ndarray, List[TemperatureCandidate]]:
    """Try several temperature parsing modes and keep the best candidate."""
    parse_modes = [
        "Between Temp and C",
        "First value followed by C",
        "Sample Temp after label",
        "Only sample number in ROI",
        "First number in ROI",
    ]
    image_candidates = build_ocr_candidates(roi_bgr, scale, blur, threshold_method, invert, contrast_factor)
    if not use_all_preprocessing:
        image_candidates = image_candidates[:1]
    psm_candidates = [psm] if not use_all_preprocessing else list(dict.fromkeys([psm, 7, 8, 6]))
    all_candidates: List[TemperatureCandidate] = []
    for image_label, processed in image_candidates:
        for candidate_psm in psm_candidates:
            raw_text, _ = ocr_temperature(
                processed,
                psm=candidate_psm,
                parse_mode="First number in ROI",
                assume_one_decimal=assume_one_decimal,
            )
            for parse_mode in parse_modes:
                value = extract_temperature(raw_text, parse_mode=parse_mode, assume_one_decimal=assume_one_decimal)
                has_value = not np.isnan(value)
                has_temp = bool(re.search(r"T\s*[e3]\s*m\s*p", raw_text or "", flags=re.IGNORECASE))
                has_c = bool(re.search(r"(?:°\s*)?C|℃", raw_text or "", flags=re.IGNORECASE))
                confidence = 0.12
                if has_value:
                    confidence += 0.42
                if parse_mode == "Between Temp and C" and has_temp and has_c:
                    confidence += 0.30
                elif parse_mode in {"First value followed by C", "Sample Temp after label"} and (has_temp or has_c):
                    confidence += 0.18
                elif has_temp or has_c:
                    confidence += 0.08
                if raw_text and not raw_text.startswith("OCR_ERROR"):
                    confidence += min(0.10, len(raw_text.strip()) / 80)
                all_candidates.append(
                    TemperatureCandidate(
                        value=value,
                        raw_text=raw_text,
                        method=f"OCR {image_label}, psm {candidate_psm}; {parse_mode}",
                        confidence=float(np.clip(confidence, 0, 1)),
                        processed_image=processed,
                    )
                )

    all_candidates.extend(run_template_matching(roi_bgr, templates or [], template_threshold))
    best = score_temperature_candidates(
        candidates=all_candidates,
        previous_value=previous_value,
        min_temp=min_temp,
        max_temp=max_temp,
        max_jump=max_jump,
        expected_trend=expected_trend,
    )
    raw_text = best.raw_text
    value = best.value
    method = best.method
    confidence = best.confidence
    processed_image = best.processed_image
    if processed_image is None:
        processed_image = preprocess_temperature_roi(
            roi_bgr,
            scale=scale,
            blur=blur,
            threshold_method=threshold_method,
            invert=invert,
            contrast_factor=contrast_factor,
        )
    return raw_text, value, method, confidence, processed_image, all_candidates


def parse_mode_from_ocr_method(method: str) -> str:
    """Extract the parse mode suffix stored by robust_auto_ocr_temperature."""
    known_modes = [
        "Between Temp and C",
        "First value followed by C",
        "Sample Temp after label",
        "Only sample number in ROI",
        "First number in ROI",
    ]
    for mode in known_modes:
        if mode in method:
            return mode
    return "Auto"


def interpolate_temperature(series: pd.Series) -> pd.Series:
    """Interpolate missing temperature values when enough valid readings exist."""
    valid_count = int(series.notna().sum())
    if valid_count < 2:
        return series
    return series.interpolate(method="linear", limit_direction="both")


def clean_temperature_series(
    series: pd.Series,
    confidence: pd.Series,
    enabled: bool,
    min_temp: float,
    max_temp: float,
    max_jump: float,
    min_confidence: float,
    expected_trend: str,
) -> pd.Series:
    """Remove obvious OCR temperature errors using simple physical bounds."""
    cleaned = series.copy()
    if not enabled or cleaned.empty:
        return cleaned

    cleaned = cleaned.where(cleaned.between(min_temp, max_temp))
    cleaned = cleaned.where(confidence >= min_confidence)

    if max_jump > 0:
        last_valid = np.nan
        for idx, value in cleaned.items():
            if pd.isna(value):
                continue
            if not pd.isna(last_valid) and abs(float(value) - float(last_valid)) > max_jump:
                cleaned.loc[idx] = np.nan
            else:
                last_valid = value

    if expected_trend == "Monotonic cooling":
        last_valid = np.nan
        for idx, value in cleaned.items():
            if pd.isna(value):
                continue
            if not pd.isna(last_valid) and value > last_valid + max(0.05, max_jump * 0.25):
                cleaned.loc[idx] = np.nan
            else:
                last_valid = value
    elif expected_trend == "Monotonic warming":
        last_valid = np.nan
        for idx, value in cleaned.items():
            if pd.isna(value):
                continue
            if not pd.isna(last_valid) and value < last_valid - max(0.05, max_jump * 0.25):
                cleaned.loc[idx] = np.nan
            else:
                last_valid = value

    return cleaned


def enforce_temporal_consistency(
    results: pd.DataFrame,
    enabled: bool,
    min_temperature: float,
    max_temperature: float,
    max_temperature_jump: float,
    min_confidence: float,
    expected_trend: str,
) -> pd.DataFrame:
    """Clean, flag, and interpolate temperature readings using temporal continuity."""
    if results.empty:
        return results

    results = results.copy()
    raw = results["temperature_C_raw"]
    confidence = results.get("confidence", pd.Series(1.0, index=results.index)).fillna(0.0)
    clean = clean_temperature_series(
        raw,
        confidence=confidence,
        enabled=enabled,
        min_temp=min_temperature,
        max_temp=max_temperature,
        max_jump=max_temperature_jump,
        min_confidence=min_confidence,
        expected_trend=expected_trend,
    )

    results["temperature_C_clean"] = clean
    results["temperature_repaired"] = raw.notna() & clean.isna()
    results["low_confidence"] = confidence < min_confidence
    results["temperature_C"] = interpolate_temperature(clean)

    if expected_trend == "Monotonic cooling" and results["temperature_C"].notna().any():
        results["temperature_C"] = results["temperature_C"].cummin()
    elif expected_trend == "Monotonic warming" and results["temperature_C"].notna().any():
        results["temperature_C"] = results["temperature_C"].cummax()

    return results


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Apply centered moving-average smoothing."""
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def robust_mad(values: pd.Series) -> float:
    """Return a robust median absolute deviation estimate."""
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return 0.0
    median = float(valid.median())
    return float(np.median(np.abs(valid.to_numpy(dtype=float) - median)))


def signal_time_order_column(results: pd.DataFrame) -> Optional[str]:
    """Choose the most reliable time-order column available in the result table."""
    for column in ("time_s", "frame", "image_index"):
        if column in results.columns:
            return column
    return None


def stabilize_pre_lcst_signal(results: pd.DataFrame, y_column: str, smoothing_window: int) -> pd.Series:
    """
    Flatten the initial pre-LCST plateau until the signal makes a sustained move.

    The raw measurements are left intact. This correction is only for plotting and
    LCST estimation, where small illumination/segmentation drift before the real
    transition would otherwise be exaggerated by normalization.
    """
    if y_column not in results.columns or results.empty:
        return pd.Series(dtype=float, index=results.index)

    order_column = signal_time_order_column(results)
    ordered = results.copy()
    ordered["_original_index"] = ordered.index
    if order_column is not None:
        ordered = ordered.sort_values(order_column)

    signal = pd.to_numeric(ordered[y_column], errors="coerce")
    valid_mask = signal.notna()
    valid_count = int(valid_mask.sum())
    stabilized = signal.copy()
    if valid_count < 6:
        return stabilized.reindex(results.index)

    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    baseline_count = max(3, min(valid_count, int(np.ceil(valid_count * 0.20))))
    initial_positions = valid_positions[:baseline_count]
    final_positions = valid_positions[-baseline_count:]
    initial_values = signal.iloc[initial_positions]
    final_values = signal.iloc[final_positions]

    initial_level = float(initial_values.median())
    final_level = float(final_values.median())
    total_change = final_level - initial_level
    finite_signal = signal.dropna()
    observed_span = float(finite_signal.quantile(0.90) - finite_signal.quantile(0.10))
    if abs(total_change) < max(1e-9, 0.05 * max(observed_span, 1e-9)):
        return stabilized.reindex(results.index)

    noise = robust_mad(initial_values)
    threshold = max(3.0 * noise, 0.12 * abs(total_change), 0.03 * max(observed_span, abs(total_change)), 1e-9)

    detection_window = max(3, min(11, int(smoothing_window) if int(smoothing_window) > 1 else 3))
    if detection_window % 2 == 0:
        detection_window += 1
    smoothed = signal.rolling(window=detection_window, center=True, min_periods=1).median().ffill().bfill()

    direction = 1.0 if total_change > 0 else -1.0
    signed_deviation = (smoothed - initial_level) * direction
    departure = signed_deviation > threshold

    onset_position = None
    departure_array = departure.to_numpy(dtype=bool)
    for pos in valid_positions[baseline_count:]:
        local = departure_array[pos : min(len(departure_array), pos + 3)]
        if int(np.count_nonzero(local)) >= min(2, len(local)):
            onset_position = int(pos)
            break

    if onset_position is None or onset_position <= 0:
        return stabilized.reindex(results.index)

    pre_positions = valid_positions[valid_positions < onset_position]
    stabilized.iloc[pre_positions] = initial_level
    stabilized.index = ordered["_original_index"]
    return stabilized.reindex(results.index)


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a signal to 0-1."""
    valid = series.dropna()
    if valid.empty:
        return series
    min_value = float(valid.min())
    max_value = float(valid.max())
    if abs(max_value - min_value) < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def normalize_to_first(series: pd.Series) -> pd.Series:
    """Normalize a signal to the first valid analyzed frame, matching the LCST method."""
    valid = series.dropna()
    if valid.empty:
        return series
    first_value = float(valid.iloc[0])
    if abs(first_value) < 1e-12:
        return series
    return series / first_value


def normalize_to_max(series: pd.Series) -> pd.Series:
    """Normalize a signal by its maximum valid value."""
    valid = series.dropna()
    if valid.empty:
        return series
    max_value = float(valid.max())
    if abs(max_value) < 1e-12:
        return series
    return series / max_value


def aggregate_signal_by_temperature(
    results: pd.DataFrame,
    y_column: str,
    aggregation: str = "Mean",
) -> pd.DataFrame:
    """Collapse repeated temperature values so each x-value has one plotted signal."""
    plot_df = results.dropna(subset=["temperature_C", y_column]).copy()
    if plot_df.empty:
        return plot_df

    plot_df = plot_df.sort_values("temperature_C")
    if aggregation == "Raw points":
        return plot_df[["temperature_C", y_column]].reset_index(drop=True)
    if aggregation == "Median":
        return plot_df.groupby("temperature_C", as_index=False)[y_column].median()
    if aggregation == "First":
        return plot_df.groupby("temperature_C", as_index=False)[y_column].first()
    return plot_df.groupby("temperature_C", as_index=False)[y_column].mean()


def estimate_lcst_inflection(
    results: pd.DataFrame,
    y_column: str,
    smoothing_window: int,
    normalization_mode: str,
    aggregation: str = "Mean",
) -> Dict[str, float]:
    """Estimate LCST as the inflection point of signal vs temperature."""
    stable_column = f"{y_column}__pre_lcst_stable"
    stable_results = results.copy()
    stable_results[stable_column] = stabilize_pre_lcst_signal(results, y_column, smoothing_window)
    grouped = aggregate_signal_by_temperature(stable_results, stable_column, aggregation)
    if len(grouped) < 5:
        return {"lcst_C": np.nan, "signal": np.nan, "slope": np.nan, "second_derivative": np.nan}

    grouped = grouped.sort_values("temperature_C")
    if len(grouped["temperature_C"].unique()) < 5:
        return {"lcst_C": np.nan, "signal": np.nan, "slope": np.nan, "second_derivative": np.nan}

    x = grouped["temperature_C"].to_numpy(dtype=float)
    y = smooth_series(grouped[stable_column], smoothing_window).to_numpy(dtype=float)
    if normalization_mode == "Min-max 0-1":
        y = normalize_series(pd.Series(y)).to_numpy(dtype=float)
    elif normalization_mode == "First frame = 1":
        y = normalize_to_first(pd.Series(y)).to_numpy(dtype=float)
    elif normalization_mode == "Max area = 1":
        y = normalize_to_max(pd.Series(y)).to_numpy(dtype=float)

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or len(np.unique(x)) < 5:
        return {"lcst_C": np.nan, "signal": np.nan, "slope": np.nan, "second_derivative": np.nan}

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    sign_changes = np.where(np.signbit(second_derivative[:-1]) != np.signbit(second_derivative[1:]))[0]

    if len(sign_changes):
        best_idx = int(sign_changes[np.argmax(np.abs(first_derivative[sign_changes]))])
    else:
        best_idx = int(np.argmax(np.abs(first_derivative)))

    return {
        "lcst_C": float(x[best_idx]),
        "signal": float(y[best_idx]),
        "slope": float(first_derivative[best_idx]),
        "second_derivative": float(second_derivative[best_idx]),
    }


def estimate_lcst_half_change(
    results: pd.DataFrame,
    y_column: str,
    smoothing_window: int,
    aggregation: str = "Mean",
) -> Dict[str, float]:
    """Estimate LCST where the signal has crossed 50% of its observed change."""
    stable_column = f"{y_column}__pre_lcst_stable"
    stable_results = results.copy()
    stable_results[stable_column] = stabilize_pre_lcst_signal(results, y_column, smoothing_window)
    grouped = aggregate_signal_by_temperature(stable_results, stable_column, aggregation)
    if len(grouped) < 2:
        return {"lcst_C": np.nan, "signal": np.nan, "target_signal": np.nan, "slope": np.nan}

    grouped = grouped.sort_values("temperature_C")
    if len(grouped["temperature_C"].unique()) < 2:
        return {"lcst_C": np.nan, "signal": np.nan, "target_signal": np.nan, "slope": np.nan}

    x = grouped["temperature_C"].to_numpy(dtype=float)
    y = smooth_series(grouped[stable_column], smoothing_window).to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or abs(float(np.nanmax(y) - np.nanmin(y))) < 1e-12:
        return {"lcst_C": np.nan, "signal": np.nan, "target_signal": np.nan, "slope": np.nan}

    order_column = signal_time_order_column(stable_results)
    time_ordered_results = stable_results.sort_values(order_column) if order_column is not None else stable_results
    time_ordered_signal = pd.to_numeric(time_ordered_results[stable_column], errors="coerce").dropna()
    if len(time_ordered_signal) >= 2:
        edge_count = max(1, min(len(time_ordered_signal), int(np.ceil(len(time_ordered_signal) * 0.20))))
        initial_level = float(time_ordered_signal.iloc[:edge_count].median())
        final_level = float(time_ordered_signal.iloc[-edge_count:].median())
        target = float(initial_level + 0.5 * (final_level - initial_level))
    else:
        target = float(np.nanmin(y) + 0.5 * (np.nanmax(y) - np.nanmin(y)))
    crossing_candidates = []
    for idx in range(len(x) - 1):
        y1 = float(y[idx])
        y2 = float(y[idx + 1])
        if (y1 - target) == 0:
            crossing_candidates.append((abs(y2 - y1), float(x[idx]), y1, idx))
        elif (y1 - target) * (y2 - target) <= 0 and y1 != y2:
            fraction = (target - y1) / (y2 - y1)
            temperature = float(x[idx] + fraction * (x[idx + 1] - x[idx]))
            crossing_candidates.append((abs(y2 - y1), temperature, target, idx))

    if crossing_candidates:
        _, temperature, signal, idx = max(crossing_candidates, key=lambda item: item[0])
        slope = float((y[idx + 1] - y[idx]) / (x[idx + 1] - x[idx])) if x[idx + 1] != x[idx] else np.nan
        return {"lcst_C": temperature, "signal": signal, "target_signal": target, "slope": slope}

    best_idx = int(np.argmin(np.abs(y - target)))
    slope_values = np.gradient(y, x) if len(np.unique(x)) >= 2 else np.full_like(y, np.nan)
    return {
        "lcst_C": float(x[best_idx]),
        "signal": float(y[best_idx]),
        "target_signal": target,
        "slope": float(slope_values[best_idx]),
    }


def estimate_lcst(
    results: pd.DataFrame,
    y_column: str,
    smoothing_window: int,
    normalization_mode: str,
    lcst_method: str,
    aggregation: str = "Mean",
) -> Dict[str, float]:
    """Estimate LCST with the selected method."""
    if lcst_method == "50% pattern disappearance":
        return estimate_lcst_half_change(results, y_column, smoothing_window, aggregation)
    return estimate_lcst_inflection(results, y_column, smoothing_window, normalization_mode, aggregation)


def process_video(
    video_path: str,
    info: VideoInfo,
    gray_roi: Tuple[int, int, int, int],
    optical_roi_mode: str,
    ring_roi: Tuple[int, int, int, int],
    freeform_mask: Optional[np.ndarray],
    auto_pattern_mask: Optional[np.ndarray],
    temp_roi: Tuple[int, int, int, int],
    frame_step: int,
    start_frame: int,
    end_frame: int,
    max_frames: Optional[int],
    preprocess_scale: int,
    blur_ocr: bool,
    threshold_method: str,
    invert_ocr: bool,
    contrast_factor: float,
    psm: int,
    temperature_parse_mode: str,
    assume_one_decimal: bool,
    expected_trend: str,
    use_all_preprocessing: bool,
    template_threshold: float,
    manual_templates: Optional[List[Dict[str, Any]]],
    clean_temperature: bool,
    min_temperature: float,
    max_temperature: float,
    max_temperature_jump: float,
    min_confidence: float,
    auto_detection_settings: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Process video frames and return gray value and temperature results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not reopen the video for analysis.")

    tx, ty, tw, th = temp_roi

    start_frame = int(np.clip(start_frame, 0, max(0, info.frame_count - 1)))
    end_frame = int(np.clip(end_frame, start_frame, max(0, info.frame_count - 1)))
    candidate_frames = list(range(start_frame, end_frame + 1, max(1, frame_step)))
    if max_frames is not None and max_frames > 0:
        candidate_frames = candidate_frames[:max_frames]

    progress = st.progress(0, text="Processing frames...")
    rows = []
    previous_temperature = np.nan

    for idx, frame_number in enumerate(candidate_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_auto_pattern_mask = auto_pattern_mask
        if optical_roi_mode == "Auto dark-edge ROI" and auto_detection_settings:
            frame_auto_pattern_mask = create_auto_edge_pattern_mask(
                frame,
                temp_roi=temp_roi,
                search_roi=auto_detection_settings.get("search_roi"),
                edge_threshold=float(auto_detection_settings.get("edge_threshold", 12.0)),
                min_area=int(auto_detection_settings.get("min_area", 25)),
                max_components=int(auto_detection_settings.get("max_components", 80)),
                dilation=int(auto_detection_settings.get("dilation", 1)),
            )
        elif optical_roi_mode == "Auto-detected pattern ROI" and auto_detection_settings:
            frame_auto_pattern_mask = create_auto_pattern_mask(
                frame,
                temp_roi=temp_roi,
                search_roi=auto_detection_settings.get("search_roi"),
                polarity=str(auto_detection_settings.get("polarity", "Darker than background")),
                threshold_delta=float(auto_detection_settings.get("threshold_delta", 18.0)),
                min_area=int(auto_detection_settings.get("min_area", 60)),
                max_components=int(auto_detection_settings.get("max_components", 20)),
            )
        optical_values = calculate_optical_signal(
            gray_frame,
            optical_roi_mode,
            gray_roi,
            ring_roi,
            freeform_mask=freeform_mask,
            auto_pattern_mask=frame_auto_pattern_mask,
            temp_roi=temp_roi,
        )

        temp_crop = crop_temperature_roi(frame, temp_roi)
        if temp_crop.size:
            if temperature_parse_mode == "Auto":
                ocr_text, temperature, ocr_method, confidence, _, _ = robust_auto_ocr_temperature(
                    temp_crop,
                    scale=preprocess_scale,
                    blur=blur_ocr,
                    threshold_method=threshold_method,
                    invert=invert_ocr,
                    contrast_factor=contrast_factor,
                    psm=psm,
                    assume_one_decimal=assume_one_decimal,
                    templates=manual_templates,
                    template_threshold=template_threshold,
                    previous_value=previous_temperature,
                    min_temp=min_temperature,
                    max_temp=max_temperature,
                    max_jump=max_temperature_jump,
                    expected_trend=expected_trend,
                    use_all_preprocessing=use_all_preprocessing,
                )
            else:
                ocr_text, temperature, ocr_method, confidence, _, _ = robust_ocr_temperature(
                    temp_crop,
                    scale=preprocess_scale,
                    blur=blur_ocr,
                    threshold_method=threshold_method,
                    invert=invert_ocr,
                    contrast_factor=contrast_factor,
                    psm=psm,
                    parse_mode=temperature_parse_mode,
                    assume_one_decimal=assume_one_decimal,
                    templates=manual_templates,
                    template_threshold=template_threshold,
                    previous_value=previous_temperature,
                    min_temp=min_temperature,
                    max_temp=max_temperature,
                    max_jump=max_temperature_jump,
                    expected_trend=expected_trend,
                    use_all_preprocessing=use_all_preprocessing,
                )
        else:
            ocr_text, temperature, ocr_method, confidence = "", np.nan, "", 0.0

        if not np.isnan(temperature):
            previous_temperature = temperature

        rows.append(
            {
                "frame": int(frame_number),
                "time_s": frame_number / info.fps,
                "temperature_C": temperature,
                "gray_value": optical_values["gray_value"],
                "sample_gray": optical_values["sample_gray"],
                "background_gray": optical_values["background_gray"],
                "darkness_value": optical_values["darkness_value"],
                "dissolution_signal": optical_values["dissolution_signal"],
                "background_corrected_darkness": optical_values["background_corrected_darkness"],
                "pattern_area_px": optical_values["pattern_area_px"],
                "pattern_area_fraction": optical_values["pattern_area_fraction"],
                "raw_text": ocr_text,
                "ocr_text": ocr_text,
                "ocr_method": ocr_method,
                "method": ocr_method,
                "confidence": confidence,
            }
        )

        progress.progress(
            (idx + 1) / max(1, len(candidate_frames)),
            text=f"Processing frame {idx + 1} of {len(candidate_frames)}",
        )

    cap.release()
    progress.empty()

    results = pd.DataFrame(rows)
    if not results.empty:
        results["gray_norm_first"] = normalize_to_first(results["gray_value"])
        results["dissolution_norm_first"] = normalize_to_first(results["dissolution_signal"])
        results["pattern_area_norm_first"] = normalize_to_first(results["pattern_area_px"])
        results["temperature_C_raw"] = results["temperature_C"]
        results = enforce_temporal_consistency(
            results,
            enabled=clean_temperature,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            max_temperature_jump=max_temperature_jump,
            min_confidence=min_confidence,
            expected_trend=expected_trend,
        )

    return results


def make_plot(
    results: pd.DataFrame,
    invert_x_axis: bool,
    smoothing_window: int,
    y_column: str,
    y_label: str,
    normalization_mode: str,
    publication_style: bool,
    show_lcst: bool,
    lcst_method: str = "Inflection point",
    temperature_aggregation: str = "Mean",
) -> plt.Figure:
    """Create a temperature plot for the selected optical signal."""
    stable_column = f"{y_column}__pre_lcst_stable"
    stable_results = results.copy()
    stable_results[stable_column] = stabilize_pre_lcst_signal(results, y_column, smoothing_window)
    plot_df = aggregate_signal_by_temperature(stable_results, stable_column, temperature_aggregation)
    signal = smooth_series(plot_df[stable_column], smoothing_window)
    if normalization_mode == "Min-max 0-1":
        signal = normalize_series(signal)
        y_label = f"Normalized {y_label}"
    elif normalization_mode == "First frame = 1":
        signal = normalize_to_first(signal)
        y_label = f"Normalized {y_label}"
    elif normalization_mode == "Max area = 1":
        signal = normalize_to_max(signal)
        y_label = f"{y_label} / max"
    plot_df["signal_plot"] = signal
    lcst = estimate_lcst(results, y_column, smoothing_window, normalization_mode, lcst_method, temperature_aggregation)

    fig, ax = plt.subplots(figsize=(3.2, 3.9) if publication_style else (4.2, 3.0), dpi=160)
    if publication_style:
        ax.plot(
            plot_df["temperature_C"],
            plot_df["signal_plot"],
            linestyle="-",
            linewidth=1.0,
            color="red",
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="red",
            markeredgewidth=1.0,
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="both", which="major", direction="out", length=5, width=1.0, labelsize=12)
        ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
        ax.minorticks_on()
        if normalization_mode in {"Min-max 0-1", "First frame = 1"}:
            ax.set_ylim(-0.05, 1.1)
        ax.set_title("")
    else:
        ax.plot(
            plot_df["temperature_C"],
            plot_df["signal_plot"],
            marker="o",
            linewidth=1.4,
            markersize=3,
            color="#1f77b4",
        )
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{y_label} vs temperature")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(y_label)
    if show_lcst and not np.isnan(lcst["lcst_C"]):
        ax.axvline(lcst["lcst_C"], color="black", linestyle="--", linewidth=1.0)
        ax.text(
            lcst["lcst_C"],
            ax.get_ylim()[0],
            f" LCST {lcst['lcst_C']:.2f}°C",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
        )
    if invert_x_axis:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def number_input_roi(
    label: str,
    image_w: int,
    image_h: int,
    default_x: int,
    default_y: int,
    default_w: int,
    default_h: int,
    square: bool,
    key_prefix: str,
) -> Tuple[int, int, int, int]:
    """Render Streamlit numeric inputs for a rectangular ROI."""
    st.subheader(label)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x = st.number_input(
            "X",
            min_value=0,
            max_value=max(0, image_w - 1),
            value=int(np.clip(default_x, 0, max(0, image_w - 1))),
            step=1,
            key=f"{key_prefix}_x",
        )
    with col2:
        y = st.number_input(
            "Y",
            min_value=0,
            max_value=max(0, image_h - 1),
            value=int(np.clip(default_y, 0, max(0, image_h - 1))),
            step=1,
            key=f"{key_prefix}_y",
        )
    with col3:
        max_width = max(1, image_w - int(x))
        if square:
            max_width = min(max_width, max(1, image_h - int(y)))
        w = st.number_input(
            "Width",
            min_value=1,
            max_value=max_width,
            value=int(np.clip(default_w, 1, max_width)),
            step=1,
            key=f"{key_prefix}_w",
        )
    with col4:
        if square:
            h = int(w)
            st.number_input(
                "Height",
                min_value=1,
                max_value=max(1, image_h - int(y)),
                value=int(np.clip(h, 1, max(1, image_h - int(y)))),
                step=1,
                key=f"{key_prefix}_h_display",
                disabled=True,
            )
        else:
            h = st.number_input(
                "Height",
                min_value=1,
                max_value=max(1, image_h - int(y)),
                value=int(np.clip(default_h, 1, max(1, image_h - int(y)))),
                step=1,
                key=f"{key_prefix}_h",
            )

    return clamp_roi(int(x), int(y), int(w), int(h), image_w, image_h)


def number_input_ring_roi(
    image_w: int,
    image_h: int,
    default_cx: int,
    default_cy: int,
    default_inner_r: int,
    default_outer_r: int,
) -> Tuple[int, int, int, int]:
    """Render numeric controls for a movable concentric dual-ring ROI."""
    st.subheader("Dual-ring optical ROI")
    col1, col2 = st.columns(2)
    with col1:
        cx = st.number_input(
            "Center X",
            min_value=0,
            max_value=max(0, image_w - 1),
            value=int(np.clip(default_cx, 0, max(0, image_w - 1))),
            step=1,
            key="ring_roi_cx",
        )
        inner_r = st.number_input(
            "Inner radius",
            min_value=1,
            max_value=max(1, min(image_w, image_h) // 2),
            value=int(np.clip(default_inner_r, 1, max(1, min(image_w, image_h) // 2))),
            step=1,
            key="ring_roi_inner_r",
        )
    with col2:
        cy = st.number_input(
            "Center Y",
            min_value=0,
            max_value=max(0, image_h - 1),
            value=int(np.clip(default_cy, 0, max(0, image_h - 1))),
            step=1,
            key="ring_roi_cy",
        )
        outer_r = st.number_input(
            "Outer radius",
            min_value=2,
            max_value=max(2, min(image_w, image_h) // 2),
            value=int(np.clip(default_outer_r, 2, max(2, min(image_w, image_h) // 2))),
            step=1,
            key="ring_roi_outer_r",
        )

    move_col1, move_col2, move_col3 = st.columns(3)
    with move_col1:
        dx = st.number_input("Move X", value=0, step=1, key="ring_move_dx")
    with move_col2:
        dy = st.number_input("Move Y", value=0, step=1, key="ring_move_dy")
    with move_col3:
        if st.button("Apply move", use_container_width=True):
            st.session_state.ring_roi_cx = int(cx + dx)
            st.session_state.ring_roi_cy = int(cy + dy)
            st.session_state.ring_move_dx = 0
            st.session_state.ring_move_dy = 0
            st.rerun()

    return clamp_ring_roi(int(cx), int(cy), int(inner_r), int(outer_r), image_w, image_h)


def show_video_info(info: VideoInfo) -> None:
    """Display video metadata in compact metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Frames", f"{info.frame_count:,}")
    col2.metric("FPS", f"{info.fps:.3g}")
    col3.metric("Duration", f"{info.duration_s:.2f} s")
    col4.metric("Width", f"{info.width}px")
    col5.metric("Height", f"{info.height}px")


def read_uploaded_image_bgr(uploaded_file) -> np.ndarray:
    """Read an uploaded image into OpenCV BGR format."""
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def parse_temperature_list(text: str) -> List[float]:
    """Parse newline/comma/space-separated temperatures."""
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")
    return [float(token) for token in tokens]


def process_image_series(
    images: List[Tuple[str, np.ndarray]],
    temperatures: List[float],
    detection_mode: str,
    temp_roi: Tuple[int, int, int, int],
    search_roi: Tuple[int, int, int, int],
    detection_settings: Dict[str, Any],
) -> pd.DataFrame:
    """Measure auto-detected pattern area for a temperature-labeled image series."""
    rows = []
    for idx, ((name, frame), temperature) in enumerate(zip(images, temperatures)):
        if detection_mode == "Auto dark-edge ROI":
            mask = create_auto_edge_pattern_mask(
                frame,
                temp_roi=temp_roi,
                search_roi=search_roi,
                edge_threshold=float(detection_settings.get("edge_threshold", 12.0)),
                min_area=int(detection_settings.get("min_area", 25)),
                max_components=int(detection_settings.get("max_components", 80)),
                dilation=int(detection_settings.get("dilation", 1)),
            )
        else:
            mask = create_auto_pattern_mask(
                frame,
                temp_roi=temp_roi,
                search_roi=search_roi,
                polarity=str(detection_settings.get("polarity", "Darker than background")),
                threshold_delta=float(detection_settings.get("threshold_delta", 18.0)),
                min_area=int(detection_settings.get("min_area", 60)),
                max_components=int(detection_settings.get("max_components", 20)),
            )

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        optical_values = calculate_optical_signal(
            gray_frame,
            detection_mode,
            gray_roi=(0, 0, 1, 1),
            ring_roi=(0, 0, 1, 2),
            auto_pattern_mask=mask,
            temp_roi=temp_roi,
        )
        rows.append(
            {
                "image_index": idx + 1,
                "file_name": name,
                "temperature_C": float(temperature),
                "pattern_area_px": optical_values["pattern_area_px"],
                "pattern_area_fraction": optical_values["pattern_area_fraction"],
                "sample_gray": optical_values["sample_gray"],
                "background_gray": optical_values["background_gray"],
                "background_corrected_darkness": optical_values["background_corrected_darkness"],
                "gray_value": optical_values["gray_value"],
                "darkness_value": optical_values["darkness_value"],
                "dissolution_signal": optical_values["dissolution_signal"],
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty:
        results["pattern_area_norm_first"] = normalize_to_first(results["pattern_area_px"])
        results["temperature_C_raw"] = results["temperature_C"]
    return results


def render_image_series_mode() -> None:
    """Run pattern-area analysis on manually temperature-labeled images."""
    st.caption("Upload images from high to low temperature, or in any order you like, then enter matching temperatures in the same order.")
    uploaded_images = st.file_uploader(
        "Upload image series",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
    )
    if not uploaded_images:
        st.info("Upload two or more images to calculate pattern area vs temperature.")
        return

    images: List[Tuple[str, np.ndarray]] = []
    for uploaded_image in uploaded_images:
        try:
            images.append((uploaded_image.name, read_uploaded_image_bgr(uploaded_image)))
        except Exception as exc:
            st.error(f"Could not read {uploaded_image.name}: {exc}")
            return

    first_name, preview_frame = images[0]
    image_h, image_w = preview_frame.shape[:2]
    if any(frame.shape[:2] != (image_h, image_w) for _, frame in images):
        st.error("All images must have the same pixel size so one search ROI can be reused.")
        return

    st.write("Image order:", ", ".join(name for name, _ in images))
    temperature_text = st.text_area(
        "Temperatures (C), one per image",
        placeholder="Example:\n30\n25\n20",
        help="The first value is paired with the first uploaded image, the second value with the second image, and so on.",
    )
    temperatures = parse_temperature_list(temperature_text)
    if temperatures and len(temperatures) != len(images):
        st.warning(f"Found {len(temperatures)} temperatures for {len(images)} images. Enter exactly one temperature per image.")

    default_search_roi = (
        max(0, int(image_w * 0.10)),
        max(0, int(image_h * 0.08)),
        max(20, int(image_w * 0.72)),
        max(20, int(image_h * 0.70)),
    )
    if "image_search_roi_x" not in st.session_state:
        set_roi_widget_state("image_search_roi", clamp_roi(*default_search_roi, image_w, image_h))

    left_col, right_col = st.columns([1.15, 1])
    with right_col:
        st.header("Area detection")
        detection_mode = st.selectbox(
            "Pattern detection mode",
            ["Auto dark-edge ROI", "Auto-detected pattern ROI"],
            index=0,
            help="Use dark-edge for ring/line outlines; use pattern ROI for filled dark or bright regions.",
        )
        search_roi = number_input_roi(
            "Auto-detection search ROI",
            image_w,
            image_h,
            default_x=st.session_state.image_search_roi_x,
            default_y=st.session_state.image_search_roi_y,
            default_w=st.session_state.image_search_roi_w,
            default_h=st.session_state.image_search_roi_h,
            square=False,
            key_prefix="image_search_roi",
        )
        temp_roi = (0, 0, 1, 1)
        detection_settings: Dict[str, Any] = {}
        if detection_mode == "Auto dark-edge ROI":
            edge_col1, edge_col2 = st.columns(2)
            with edge_col1:
                detection_settings["edge_threshold"] = st.slider("Dark edge strength", 2, 80, 12, 1, key="image_edge_threshold")
                detection_settings["dilation"] = st.slider("Edge thickness", 0, 5, 1, 1, key="image_edge_dilation")
            with edge_col2:
                detection_settings["min_area"] = st.number_input("Min edge component area", min_value=5, value=25, step=5, key="image_edge_min_area")
                detection_settings["max_components"] = st.number_input("Max edge components", min_value=1, value=80, step=1, key="image_edge_max_components")
            preview_mask = create_auto_edge_pattern_mask(
                preview_frame,
                temp_roi=temp_roi,
                search_roi=search_roi,
                edge_threshold=float(detection_settings["edge_threshold"]),
                min_area=int(detection_settings["min_area"]),
                max_components=int(detection_settings["max_components"]),
                dilation=int(detection_settings["dilation"]),
            )
        else:
            pattern_col1, pattern_col2 = st.columns(2)
            with pattern_col1:
                detection_settings["polarity"] = st.selectbox(
                    "Pattern type",
                    ["Darker than background", "Brighter than background"],
                    index=0,
                    key="image_pattern_polarity",
                )
                detection_settings["threshold_delta"] = st.slider("Background difference threshold", 2, 80, 18, 1, key="image_pattern_delta")
            with pattern_col2:
                detection_settings["min_area"] = st.number_input("Min component area", min_value=5, value=60, step=5, key="image_pattern_min_area")
                detection_settings["max_components"] = st.number_input("Max components", min_value=1, value=20, step=1, key="image_pattern_max_components")
            preview_mask = create_auto_pattern_mask(
                preview_frame,
                temp_roi=temp_roi,
                search_roi=search_roi,
                polarity=str(detection_settings["polarity"]),
                threshold_delta=float(detection_settings["threshold_delta"]),
                min_area=int(detection_settings["min_area"]),
                max_components=int(detection_settings["max_components"]),
            )

        preview_effective_area = effective_pattern_area(preview_mask, detection_mode)
        st.metric("Preview effective area", f"{preview_effective_area:,.0f} px")
        run_images = st.button("Run image area analysis", type="primary", use_container_width=True)

    with left_col:
        st.header("Preview")
        overlay = draw_rois(
            preview_frame,
            gray_roi=(0, 0, 1, 1),
            temp_roi=temp_roi,
            optical_roi_mode=detection_mode,
            auto_pattern_mask=preview_mask,
            search_roi=search_roi,
        )
        st.image(overlay, caption=f"Preview: {first_name}", use_container_width=True)

    if run_images:
        if len(temperatures) != len(images):
            st.error("Enter exactly one temperature for each image before running analysis.")
            return
        results = process_image_series(
            images=images,
            temperatures=temperatures,
            detection_mode=detection_mode,
            temp_roi=temp_roi,
            search_roi=search_roi,
            detection_settings=detection_settings,
        )
        st.session_state.image_results = results

    results = st.session_state.get("image_results")
    if results is None or results.empty:
        return

    st.header("Results")
    st.dataframe(results, use_container_width=True, height=320)
    plot_col, export_col = st.columns([1, 2])
    with export_col:
        st.header("Plot controls")
        normalization_mode = st.selectbox("Normalization", ["None", "First frame = 1", "Min-max 0-1", "Max area = 1"], index=0, key="image_norm")
        smoothing_window = st.slider("Moving-average smoothing window", 1, 51, 1, 2, key="image_smoothing")
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        temperature_aggregation = st.selectbox(
            "Repeated temperatures",
            ["Mean", "Median", "First", "Raw points"],
            index=0,
            key="image_temperature_aggregation",
            help="Mean collapses repeated temperatures into one point, so one x-value has only one y-value.",
        )
        invert_x_axis = st.checkbox("Invert x-axis for cooling", value=True, key="image_invert_x")
        show_lcst = st.checkbox("Estimate LCST", value=True, key="image_show_lcst")
        lcst_method = st.selectbox(
            "LCST method",
            ["50% pattern disappearance", "Inflection point"],
            index=0,
            key="image_lcst_method",
        )
        lcst = estimate_lcst(
            results,
            "pattern_area_px",
            smoothing_window,
            normalization_mode,
            lcst_method,
            temperature_aggregation,
        )
        if show_lcst and not np.isnan(lcst["lcst_C"]):
            st.metric("Estimated LCST", f"{lcst['lcst_C']:.2f} °C")
            if lcst_method == "50% pattern disappearance":
                st.caption(
                    "50% method: LCST is where pattern area reaches the midpoint between the observed maximum and minimum areas."
                )
            else:
                st.caption("Inflection method: LCST is where the smoothed area-temperature curve turns most sharply.")
        elif show_lcst:
            st.caption("Not enough valid points to estimate LCST.")
        area_csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=area_csv,
            file_name="pattern_area_temperature_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with plot_col:
        fig = make_plot(
            results,
            invert_x_axis=invert_x_axis,
            smoothing_window=smoothing_window,
            y_column="pattern_area_px",
            y_label="Pattern area (px)",
            normalization_mode=normalization_mode,
            publication_style=True,
            show_lcst=show_lcst,
            lcst_method=lcst_method,
            temperature_aggregation=temperature_aggregation,
        )
        st.pyplot(fig, clear_figure=True, use_container_width=False)


def add_manual_temperature_template(
    frame_bgr: np.ndarray,
    temp_roi: Tuple[int, int, int, int],
    value: float,
) -> None:
    """Store a manually labeled ROI image as a template for later matching."""
    crop = crop_temperature_roi(frame_bgr, temp_roi)
    if crop.size == 0:
        return
    template = {
        "value": float(value),
        "image": normalize_template_roi(crop),
        "display": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
    }
    st.session_state.setdefault("temperature_templates", [])
    st.session_state.temperature_templates.append(template)


def debug_temperature_panel(
    video_path: str,
    info: VideoInfo,
    temp_roi: Tuple[int, int, int, int],
    preview_frame: np.ndarray,
    preprocess_scale: int,
    blur_ocr: bool,
    threshold_method: str,
    invert_ocr: bool,
    contrast_factor: float,
    psm: int,
    temperature_parse_mode: str,
    assume_one_decimal: bool,
    template_threshold: float,
    min_temperature: float,
    max_temperature: float,
    max_temperature_jump: float,
    expected_trend: str,
    use_all_preprocessing: bool,
) -> None:
    """Show OCR/template candidates and sampled-frame checks for temperature debugging."""
    with st.expander("Temperature debug", expanded=False):
        crop = crop_temperature_roi(preview_frame, temp_roi)
        if crop.size == 0:
            st.warning("Temperature ROI is empty.")
            return

        st.subheader("Current-frame candidates")
        _, _, _, _, _, candidates = robust_ocr_temperature(
            crop,
            scale=preprocess_scale,
            blur=blur_ocr,
            threshold_method=threshold_method,
            invert=invert_ocr,
            contrast_factor=contrast_factor,
            psm=psm,
            parse_mode=temperature_parse_mode,
            assume_one_decimal=assume_one_decimal,
            templates=st.session_state.get("temperature_templates", []),
            template_threshold=template_threshold,
            previous_value=np.nan,
            min_temp=min_temperature,
            max_temp=max_temperature,
            max_jump=max_temperature_jump,
            expected_trend=expected_trend,
            use_all_preprocessing=True,
        )
        candidate_rows = [
            {
                "value": candidate.value,
                "confidence": candidate.confidence,
                "method": candidate.method,
                "raw_text": candidate.raw_text,
            }
            for candidate in candidates
        ]
        st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, height=220)

        st.subheader("Preprocessing previews")
        image_candidates = build_ocr_candidates(
            crop,
            preprocess_scale,
            blur_ocr,
            threshold_method,
            invert_ocr,
            contrast_factor,
        )
        preview_cols = st.columns(min(3, len(image_candidates)))
        for idx, (label, image) in enumerate(image_candidates[:6]):
            with preview_cols[idx % len(preview_cols)]:
                st.image(Image.fromarray(image), caption=label, use_container_width=True)

        st.subheader("Sampled frames")
        sample_count = st.slider("Number of frames to inspect", 3, 12, 5, 1, key="debug_sample_count")
        sample_frames = np.linspace(0, max(0, info.frame_count - 1), sample_count, dtype=int)
        sample_rows = []
        previous_value = np.nan
        for frame_number in sample_frames:
            frame = read_frame(video_path, int(frame_number))
            if frame is None:
                continue
            sample_crop = crop_temperature_roi(frame, temp_roi)
            raw_text, value, method, confidence, _, _ = robust_ocr_temperature(
                sample_crop,
                scale=preprocess_scale,
                blur=blur_ocr,
                threshold_method=threshold_method,
                invert=invert_ocr,
                contrast_factor=contrast_factor,
                psm=psm,
                parse_mode=temperature_parse_mode,
                assume_one_decimal=assume_one_decimal,
                templates=st.session_state.get("temperature_templates", []),
                template_threshold=template_threshold,
                previous_value=previous_value,
                min_temp=min_temperature,
                max_temp=max_temperature,
                max_jump=max_temperature_jump,
                expected_trend=expected_trend,
                use_all_preprocessing=use_all_preprocessing,
            )
            if not np.isnan(value):
                previous_value = value
            sample_rows.append(
                {
                    "frame": int(frame_number),
                    "time_s": int(frame_number) / info.fps,
                    "temperature": value,
                    "confidence": confidence,
                    "method": method,
                    "raw_text": raw_text,
                }
            )
        st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, height=240)


def render_simplified_video_mode() -> None:
    """Simplified primary workflow for pattern-area LCST analysis."""
    st.title("Gray Value Reader Helper")
    st.caption("Select the pattern area, preview detection, then measure normalized pattern area vs temperature.")

    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Temperature text should stay in the same frame location throughout the video.",
    )
    if uploaded_file is None:
        st.info("Upload a video to begin.")
        return

    if "video_path" not in st.session_state or st.session_state.get("uploaded_name") != uploaded_file.name:
        st.session_state.video_path = save_uploaded_video(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.video_bytes = uploaded_file.getvalue()
        st.session_state.video_mime = video_mime_type(uploaded_file.name, uploaded_file.type)
        st.session_state.pop("auto_pattern_mask", None)
        st.session_state.pop("results", None)
        st.session_state.pop("simple_search_roi_applied", None)
        st.session_state.pop("simple_temp_roi_applied", None)
        st.session_state.pop("simple_search_roi", None)
        st.session_state.pop("simple_temp_roi", None)
        st.session_state.pop("pending_simple_search_roi", None)
        st.session_state.pop("pending_simple_temp_roi", None)
        st.session_state.pop("simple_temperature_parse_mode", None)
        st.session_state.pop("simple_preview_temperature", None)
    elif "video_bytes" not in st.session_state or "video_mime" not in st.session_state:
        st.session_state.video_bytes = uploaded_file.getvalue()
        st.session_state.video_mime = video_mime_type(uploaded_file.name, uploaded_file.type)

    try:
        cap, info = open_video(st.session_state.video_path)
        cap.release()
    except Exception as exc:
        st.error(f"Could not read video: {exc}")
        return

    show_video_info(info)
    if info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
        st.error("The video metadata is incomplete. Try converting the video to MP4 and uploading again.")
        return

    pending_search_roi = st.session_state.pop("pending_simple_search_roi", None)
    if pending_search_roi is not None:
        st.session_state.simple_search_roi = clamp_roi(*pending_search_roi, info.width, info.height)
        st.session_state.simple_search_roi_applied = True
        st.session_state.pop("auto_pattern_mask", None)
    pending_temp_roi = st.session_state.pop("pending_simple_temp_roi", None)
    if pending_temp_roi is not None:
        st.session_state.simple_temp_roi = clamp_roi(*pending_temp_roi, info.width, info.height)
        st.session_state.simple_temp_roi_applied = True
        st.session_state.pop("auto_pattern_mask", None)
    st.session_state.setdefault("temperature_templates", [])

    st.header("Step 1: Draw ROIs and Preview Detection")
    if "preview_frame_index" not in st.session_state:
        st.session_state.preview_frame_index = 0
    st.session_state.preview_frame_index = int(
        np.clip(st.session_state.preview_frame_index, 0, max(0, info.frame_count - 1))
    )
    st.session_state.setdefault("preview_playing", False)

    def set_preview_frame(frame_index: int, reset_canvas: bool = True) -> None:
        st.session_state.preview_frame_index = int(np.clip(frame_index, 0, max(0, info.frame_count - 1)))
        if reset_canvas:
            st.session_state.simple_canvas_clear_counter = st.session_state.get("simple_canvas_clear_counter", 0) + 1

    player_col1, player_col2, player_col3, player_col4, player_col5 = st.columns([0.8, 0.8, 0.8, 0.8, 1.2])
    with player_col1:
        if st.button("Play", use_container_width=True):
            st.session_state.preview_playing = True
            st.session_state.preview_play_started_at = time.monotonic()
            st.session_state.preview_play_start_frame = int(st.session_state.preview_frame_index)
    with player_col2:
        if st.button("Pause", use_container_width=True):
            st.session_state.preview_playing = False
            st.session_state.simple_canvas_clear_counter = st.session_state.get("simple_canvas_clear_counter", 0) + 1
            st.rerun()
    with player_col3:
        if st.button("-1 s", use_container_width=True):
            set_preview_frame(st.session_state.preview_frame_index - int(round(info.fps)))
            st.session_state.preview_playing = False
            st.rerun()
    with player_col4:
        if st.button("+1 s", use_container_width=True):
            set_preview_frame(st.session_state.preview_frame_index + int(round(info.fps)))
            st.session_state.preview_playing = False
            st.rerun()
    with player_col5:
        playback_speed = st.slider(
            "Speed",
            min_value=0.25,
            max_value=2.0,
            value=1.0,
            step=0.25,
            help="Preview playback speed.",
        )

    if st.session_state.preview_playing:
        started_at = float(st.session_state.get("preview_play_started_at", time.monotonic()))
        start_frame = int(st.session_state.get("preview_play_start_frame", st.session_state.preview_frame_index))
        elapsed = max(0.0, time.monotonic() - started_at)
        next_frame = start_frame + int(round(elapsed * info.fps * float(playback_speed)))
        if next_frame <= st.session_state.preview_frame_index:
            next_frame = st.session_state.preview_frame_index + 1
        if next_frame >= info.frame_count - 1:
            set_preview_frame(info.frame_count - 1, reset_canvas=True)
            st.session_state.preview_playing = False
        else:
            set_preview_frame(next_frame, reset_canvas=False)

    preview_frame_index = st.slider(
        "Preview frame",
        min_value=0,
        max_value=max(0, info.frame_count - 1),
        value=st.session_state.preview_frame_index,
        step=1,
        key="preview_frame_index",
    )
    preview_frame = read_frame(info.path, preview_frame_index)
    if preview_frame is None:
        st.error("Could not read the selected preview frame.")
        return

    if st.session_state.get("simple_search_roi_applied", False) and "simple_search_roi" not in st.session_state:
        st.session_state.simple_search_roi_applied = False
    if st.session_state.get("simple_temp_roi_applied", False) and "simple_temp_roi" not in st.session_state:
        st.session_state.simple_temp_roi_applied = False

    current_search_roi = st.session_state.get("simple_search_roi")
    if current_search_roi is not None:
        current_search_roi = clamp_roi(*current_search_roi, info.width, info.height)
        st.session_state.simple_search_roi = current_search_roi
    current_temp_roi = st.session_state.get("simple_temp_roi")
    if current_temp_roi is not None:
        current_temp_roi = clamp_roi(*current_temp_roi, info.width, info.height)
        st.session_state.simple_temp_roi = current_temp_roi

    left_col, right_col = st.columns([1.6, 0.9])
    search_roi = current_search_roi
    temp_roi = current_temp_roi
    with right_col:
        st.subheader("ROI status")
        st.caption("Draw on the preview image, choose the ROI type above the image, then apply it.")
        if st.session_state.get("simple_search_roi_applied", False) and search_roi is not None:
            st.success(f"Pattern search ROI: {search_roi[2]} x {search_roi[3]} px")
        else:
            st.warning("Draw and apply a Pattern search ROI on the preview image.")
        if st.session_state.get("simple_temp_roi_applied", False) and temp_roi is not None:
            st.success(f"Temperature OCR ROI: {temp_roi[2]} x {temp_roi[3]} px")
        else:
            st.warning("Draw and apply a Temperature OCR ROI on the preview image.")

    with right_col:
        st.subheader("Detection preview")
        edge_threshold = 12
        edge_dilation = 1
        edge_min_area = 80
        edge_max_components = 80
        with st.expander("Detection tuning", expanded=False):
            edge_threshold = st.slider("Dark edge strength", 2, 80, edge_threshold, 1)
            edge_dilation = st.slider("Edge thickness", 0, 5, edge_dilation, 1)
            edge_min_area = st.number_input("Min edge component area", min_value=5, value=edge_min_area, step=5)
            edge_max_components = st.number_input("Max edge components", min_value=1, value=edge_max_components, step=1)

        auto_detection_settings = {
            "search_roi": search_roi,
            "edge_threshold": float(edge_threshold),
            "min_area": int(edge_min_area),
            "max_components": int(edge_max_components),
            "dilation": int(edge_dilation),
        }
        tesseract_cmd = st.session_state.get("tesseract_cmd", default_tesseract_path())
        tesseract_ok, tesseract_message = get_tesseract_status(tesseract_cmd)
        if st.button("Preview detection", type="primary", use_container_width=True):
            if not st.session_state.get("simple_search_roi_applied", False) or search_roi is None:
                st.error("Draw and apply a Pattern search ROI first.")
                return
            if not st.session_state.get("simple_temp_roi_applied", False) or temp_roi is None:
                st.error("Draw and apply a Temperature OCR ROI first.")
                return
            auto_pattern_mask = create_auto_edge_pattern_mask(
                preview_frame,
                temp_roi=temp_roi,
                search_roi=search_roi,
                edge_threshold=float(edge_threshold),
                min_area=int(edge_min_area),
                max_components=int(edge_max_components),
                dilation=int(edge_dilation),
            )
            st.session_state.auto_pattern_mask = auto_pattern_mask
            effective_area = effective_pattern_area(auto_pattern_mask, "Auto dark-edge ROI")
            st.success(f"Detected effective edge area: {effective_area:,.0f} px.")
            st.rerun()

        if st.session_state.get("simple_temp_roi_applied", False) and temp_roi is not None:
            tx, ty, tw, th = temp_roi
            temp_crop = preview_frame[ty : ty + th, tx : tx + tw]
            if not tesseract_ok:
                st.warning("Temperature preview needs Tesseract setup.")
            elif temp_crop.size == 0:
                st.warning("Temperature ROI is empty.")
            else:
                ocr_text, parsed_temp, preview_method, preview_confidence, _, _ = robust_auto_ocr_temperature(
                    temp_crop,
                    scale=3,
                    blur=False,
                    threshold_method="Contrast only",
                    invert=False,
                    contrast_factor=2.0,
                    psm=7,
                    assume_one_decimal=True,
                    templates=st.session_state.get("temperature_templates", []),
                    template_threshold=0.92,
                    previous_value=np.nan,
                    min_temp=-100.0,
                    max_temp=150.0,
                    max_jump=0.0,
                    expected_trend="Monotonic cooling",
                    use_all_preprocessing=False,
                )
                if np.isnan(parsed_temp):
                    st.warning("Preview temperature: not detected")
                    if ocr_text:
                        st.caption(f"OCR text: `{ocr_text}`")
                    st.session_state.pop("simple_preview_temperature", None)
                else:
                    st.session_state.simple_temperature_parse_mode = parse_mode_from_ocr_method(preview_method)
                    st.session_state.simple_preview_temperature = float(parsed_temp)
                    st.metric("Preview temperature", f"{parsed_temp:g} °C")
                    st.caption(
                        f"OCR confidence: {preview_confidence:.2f}. "
                        f"Run mode: {st.session_state.simple_temperature_parse_mode}"
                    )

        if not tesseract_ok:
            with st.expander("Tesseract setup", expanded=True):
                tesseract_cmd = st.text_input(
                    "Tesseract executable path",
                    value=tesseract_cmd,
                    placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe or /usr/bin/tesseract",
                )
                st.session_state.tesseract_cmd = tesseract_cmd
                tesseract_ok, tesseract_message = get_tesseract_status(tesseract_cmd)
                if not tesseract_ok:
                    st.error(tesseract_message)

    auto_pattern_mask = st.session_state.get("auto_pattern_mask")
    if isinstance(auto_pattern_mask, np.ndarray) and auto_pattern_mask.shape != (info.height, info.width):
        auto_pattern_mask = None
        st.session_state.pop("auto_pattern_mask", None)

    with left_col:
        visible_search_roi = search_roi if st.session_state.get("simple_search_roi_applied", False) and search_roi is not None else None
        visible_temp_roi = bool(st.session_state.get("simple_temp_roi_applied", False) and temp_roi is not None)
        overlay = draw_rois(
            preview_frame,
            gray_roi=(0, 0, 1, 1),
            temp_roi=temp_roi or (0, 0, 1, 1),
            optical_roi_mode="Auto dark-edge ROI",
            auto_pattern_mask=auto_pattern_mask,
            search_roi=visible_search_roi,
            show_temp_roi=visible_temp_roi,
            show_gray_roi=False,
        )
        st.caption("Drag a box directly on the preview image, then click Apply selected ROI.")
        if st.session_state.preview_playing:
            st.image(overlay, use_container_width=True)
            st.info("Playing preview. Press Pause to draw or adjust ROIs on this frame.")
        else:
            plotly_roi_selector(
                preview_frame,
                info.width,
                info.height,
                display_image_rgb=overlay,
                max_preview_width=560,
            )

    if st.session_state.preview_playing:
        time.sleep(0.04)
        st.rerun()

    st.header("Step 2: Analysis Settings")
    settings_col, run_col = st.columns([1.4, 0.8])
    with settings_col:
        analyze_partial = st.checkbox("Analyze only part of the video", value=False)
        if analyze_partial:
            start_time_s, end_time_s = st.slider(
                "Time range to analyze (seconds)",
                min_value=0.0,
                max_value=max(0.0, float(info.duration_s)),
                value=(0.0, max(0.0, float(info.duration_s))),
                step=max(0.001, 1.0 / info.fps),
            )
        else:
            start_time_s, end_time_s = 0.0, info.duration_s

        start_frame = int(np.clip(round(start_time_s * info.fps), 0, max(0, info.frame_count - 1)))
        end_frame = int(np.clip(round(end_time_s * info.fps), start_frame, max(0, info.frame_count - 1)))
        st.caption(f"Frame range: {start_frame} to {end_frame}")
        frame_step = st.number_input(
            "Analyze every Nth frame",
            min_value=1,
            max_value=max(1, info.frame_count),
            value=max(1, min(10, int(round(info.fps)) if info.fps > 0 else 1)),
            step=1,
        )
        st.caption("Larger values are faster. Use 1 only when every frame is needed.")
    with run_col:
        st.write("")
        st.write("")
        run_analysis = st.button("Run analysis", type="primary", use_container_width=True)
        st.caption("Output uses pre-LCST plateau stabilization, area/max-area normalization, median aggregation, and light smoothing.")

    if run_analysis:
        if not st.session_state.get("simple_search_roi_applied", False) or search_roi is None:
            st.error("Draw and apply a Pattern search ROI first.")
            return
        if not st.session_state.get("simple_temp_roi_applied", False) or temp_roi is None:
            st.error("Draw and apply a Temperature OCR ROI first.")
            return
        if not tesseract_ok:
            st.error("Cannot run temperature OCR because Tesseract is not available.")
            return
        tx, ty, tw, th = temp_roi
        temp_crop = preview_frame[ty : ty + th, tx : tx + tw]
        if temp_crop.size == 0:
            st.error("Temperature ROI is empty. Draw and apply the Temperature OCR ROI again.")
            return
        preview_check_text, preview_check_temp, _, _, _, _ = robust_ocr_temperature(
            temp_crop,
            scale=3,
            blur=False,
            threshold_method="Contrast only",
            invert=False,
            contrast_factor=2.0,
            psm=7,
            parse_mode=st.session_state.get("simple_temperature_parse_mode", "First number in ROI"),
            assume_one_decimal=True,
            templates=st.session_state.get("temperature_templates", []),
            template_threshold=0.92,
            previous_value=np.nan,
            min_temp=-100.0,
            max_temp=150.0,
            max_jump=0.0,
            expected_trend="Monotonic cooling",
            use_all_preprocessing=False,
        )
        if np.isnan(preview_check_temp):
            st.error(
                "The current Temperature OCR ROI does not read a temperature at run time. "
                "Draw and apply the Temperature OCR ROI again."
            )
            if preview_check_text:
                st.caption(f"OCR text: `{preview_check_text}`")
            return
        preview_temp = st.session_state.get("simple_preview_temperature")
        if preview_temp is not None and abs(float(preview_check_temp) - float(preview_temp)) > 0.2:
            st.warning(
                f"Run-time preview temperature is {preview_check_temp:g} °C, "
                f"while the displayed preview was {float(preview_temp):g} °C. "
                "The analysis will use the current applied Temperature OCR ROI."
            )
        try:
            with st.spinner("Analyzing video frames..."):
                results = process_video(
                    video_path=info.path,
                    info=info,
                    gray_roi=(0, 0, 1, 1),
                    optical_roi_mode="Auto dark-edge ROI",
                    ring_roi=(0, 0, 1, 2),
                    freeform_mask=None,
                    auto_pattern_mask=auto_pattern_mask,
                    temp_roi=temp_roi,
                    frame_step=int(frame_step),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    max_frames=None,
                    preprocess_scale=3,
                    blur_ocr=False,
                    threshold_method="Contrast only",
                    invert_ocr=False,
                    contrast_factor=2.0,
                    psm=7,
                    temperature_parse_mode=st.session_state.get("simple_temperature_parse_mode", "Auto"),
                    assume_one_decimal=True,
                    expected_trend="Mostly smooth",
                    use_all_preprocessing=False,
                    template_threshold=0.92,
                    manual_templates=st.session_state.get("temperature_templates", []),
                    clean_temperature=False,
                    min_temperature=-100.0,
                    max_temperature=150.0,
                    max_temperature_jump=0.0,
                    min_confidence=0.20,
                    auto_detection_settings=auto_detection_settings,
                )
            st.session_state.results = results
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    results = st.session_state.get("results")
    if results is None or results.empty:
        return

    st.header("Step 3: Get Your LCST")
    if "temperature_C_raw" not in results.columns:
        results["temperature_C_raw"] = results.get("temperature_C", np.nan)
    if "confidence" not in results.columns:
        results["confidence"] = np.nan

    display_cols = [
        "frame",
        "time_s",
        "temperature_C",
        "temperature_C_raw",
        "pattern_area_px",
        "pattern_area_fraction",
        "pattern_area_norm_first",
        "raw_text",
        "confidence",
    ]
    for column in display_cols:
        if column not in results.columns:
            results[column] = "" if column == "raw_text" else np.nan

    normalization_mode = "Max area = 1"
    y_column = "pattern_area_px"
    y_label = "Pattern area (px)"
    smoothing_window = st.session_state.get("simple_smoothing_window", 5)
    temperature_aggregation = st.session_state.get("simple_temperature_aggregation", "Median")
    lcst_method = "50% pattern disappearance"
    lcst = estimate_lcst(results, y_column, smoothing_window, normalization_mode, lcst_method, temperature_aggregation)
    total = len(results)
    raw_missing = int(results["temperature_C_raw"].isna().sum())
    valid_temperature_count = int(results["temperature_C"].notna().sum())
    unique_temperature_count = int(results["temperature_C"].dropna().round(3).nunique())
    unique_raw_temperature_count = int(results["temperature_C_raw"].dropna().round(3).nunique())
    temperature_min = float(results["temperature_C"].min()) if valid_temperature_count else np.nan
    temperature_max = float(results["temperature_C"].max()) if valid_temperature_count else np.nan

    plot_col, export_col = st.columns([1.15, 0.85])
    with plot_col:
        fig = make_plot(
            results,
            invert_x_axis=True,
            smoothing_window=smoothing_window,
            y_column=y_column,
            y_label=y_label,
            normalization_mode=normalization_mode,
            publication_style=True,
            show_lcst=True,
            lcst_method=lcst_method,
            temperature_aggregation=temperature_aggregation,
        )
        st.pyplot(fig, clear_figure=True, use_container_width=False)
    with export_col:
        st.metric("Estimated LCST", f"{lcst['lcst_C']:.2f} °C" if not np.isnan(lcst["lcst_C"]) else "N/A")
        st.metric("Processed frames", f"{total:,}")
        st.metric("Successful temp reads", f"{total - raw_missing:,}")
        st.metric("Unique temperatures", f"{unique_temperature_count:,}")
        if unique_raw_temperature_count != unique_temperature_count:
            st.caption(f"Raw unique temperatures: {unique_raw_temperature_count:,}")
        st.metric("OCR missing rate", f"{raw_missing / total:.1%}" if total else "0.0%")
        if unique_temperature_count < 2:
            st.warning(
                "Only one unique temperature was detected. The plot collapses to one point and LCST cannot be estimated."
            )
            if valid_temperature_count:
                st.caption(f"Detected temperature range: {temperature_min:g} to {temperature_max:g} °C")
            st.caption("Check whether the temperature overlay changes during the selected time range, or reduce Analyze every Nth frame.")
        st.caption("Output uses pre-LCST plateau stabilization, area/max-area normalization, median aggregation, and light smoothing.")
        with st.expander("Curve smoothing", expanded=False):
            st.session_state.simple_temperature_aggregation = st.selectbox(
                "Repeated temperatures",
                ["Median", "Mean", "First", "Raw points"],
                index=["Median", "Mean", "First", "Raw points"].index(temperature_aggregation)
                if temperature_aggregation in {"Median", "Mean", "First", "Raw points"}
                else 0,
                help="Median is usually more stable when frame-by-frame area detection jitters.",
            )
            smoothing_value = st.slider(
                "Smoothing window",
                1,
                21,
                int(smoothing_window),
                2,
                help="Higher values reduce noisy area fluctuations but can soften sharp transitions.",
            )
            if smoothing_value % 2 == 0:
                smoothing_value += 1
            st.session_state.simple_smoothing_window = smoothing_value
            st.caption("Change these settings, then rerun analysis or refresh to update the plot.")
        csv = results[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="pattern_area_temperature_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with st.expander("Result table", expanded=False):
        st.dataframe(results[display_cols], use_container_width=True, height=320)


def legacy_main() -> None:
    st.title("Gray Value Reader Helper")
    st.caption("Extract grayscale or auto-detected pattern area and pair it with temperature.")

    input_mode = st.radio(
        "Input type",
        ["Image series with manual temperatures", "Video with OCR temperature"],
        horizontal=True,
    )
    if input_mode == "Image series with manual temperatures":
        render_image_series_mode()
        return

    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Temperature text should remain in the same frame location throughout the video.",
    )

    if uploaded_file is None:
        st.info("Upload a video to begin.")
        return

    if "video_path" not in st.session_state or st.session_state.get("uploaded_name") != uploaded_file.name:
        st.session_state.video_path = save_uploaded_video(uploaded_file)
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.video_bytes = uploaded_file.getvalue()
        st.session_state.video_mime = video_mime_type(uploaded_file.name, uploaded_file.type)
    elif "video_bytes" not in st.session_state or "video_mime" not in st.session_state:
        st.session_state.video_bytes = uploaded_file.getvalue()
        st.session_state.video_mime = video_mime_type(uploaded_file.name, uploaded_file.type)

    try:
        cap, info = open_video(st.session_state.video_path)
        cap.release()
    except Exception as exc:
        st.error(f"Could not read video: {exc}")
        return

    show_video_info(info)

    if info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
        st.error("The video metadata is incomplete. Try converting the video to MP4 and uploading again.")
        return

    st.divider()

    default_gray_size = min(7, info.width, info.height)
    default_gray_roi = (
        max(0, info.width // 2 - default_gray_size // 2),
        max(0, info.height // 2 - default_gray_size // 2),
        default_gray_size,
        default_gray_size,
    )
    default_temp_roi = (
        max(0, int(info.width * 0.65)),
        max(0, int(info.height * 0.05)),
        max(20, int(info.width * 0.25)),
        max(12, int(info.height * 0.12)),
    )
    default_ring_roi = (
        info.width // 2,
        info.height // 2,
        max(4, min(info.width, info.height) // 40),
        max(8, min(info.width, info.height) // 20),
    )
    default_search_roi = (
        max(0, int(info.width * 0.10)),
        max(0, int(info.height * 0.08)),
        max(20, int(info.width * 0.72)),
        max(20, int(info.height * 0.70)),
    )

    if "gray_roi_x" not in st.session_state:
        set_roi_widget_state("gray_roi", clamp_roi(*default_gray_roi, info.width, info.height))
    if "temp_roi_x" not in st.session_state:
        set_roi_widget_state("temp_roi", clamp_roi(*default_temp_roi, info.width, info.height))
    if "search_roi_x" not in st.session_state:
        set_roi_widget_state("search_roi", clamp_roi(*default_search_roi, info.width, info.height))
    if "ring_roi_cx" not in st.session_state:
        cx, cy, inner_r, outer_r = clamp_ring_roi(*default_ring_roi, info.width, info.height)
        st.session_state.ring_roi_cx = cx
        st.session_state.ring_roi_cy = cy
        st.session_state.ring_roi_inner_r = inner_r
        st.session_state.ring_roi_outer_r = outer_r
    st.session_state.setdefault("temperature_templates", [])

    st.header("Video playback")
    st.video(
        st.session_state.video_bytes,
        format=st.session_state.video_mime,
    )
    if os.path.splitext(st.session_state.uploaded_name.lower())[1] in {".avi", ".mkv"}:
        st.caption(
            "If the browser cannot play this AVI/MKV file, the analysis can still run. "
            "For browser playback, MP4 with H.264 encoding is the most reliable."
        )

    st.header("Frame selection")
    preview_frame_index = st.slider(
        "Preview frame",
        min_value=0,
        max_value=max(0, info.frame_count - 1),
        value=0,
        step=1,
        key="preview_frame_index",
    )
    st.caption(f"Preview time: {preview_frame_index / info.fps:.3f} s")

    preview_frame = read_frame(info.path, preview_frame_index)
    if preview_frame is None:
        st.error("Could not read the selected preview frame.")
        return

    left_col, right_col = st.columns([1.15, 1])

    current_gray_roi = clamp_roi(
        st.session_state.gray_roi_x,
        st.session_state.gray_roi_y,
        st.session_state.gray_roi_w,
        st.session_state.gray_roi_w,
        info.width,
        info.height,
    )
    current_temp_roi = clamp_roi(
        st.session_state.temp_roi_x,
        st.session_state.temp_roi_y,
        st.session_state.temp_roi_w,
        st.session_state.temp_roi_h,
        info.width,
        info.height,
    )
    current_search_roi = clamp_roi(
        st.session_state.search_roi_x,
        st.session_state.search_roi_y,
        st.session_state.search_roi_w,
        st.session_state.search_roi_h,
        info.width,
        info.height,
    )
    current_ring_roi = clamp_ring_roi(
        st.session_state.ring_roi_cx,
        st.session_state.ring_roi_cy,
        st.session_state.ring_roi_inner_r,
        st.session_state.ring_roi_outer_r,
        info.width,
        info.height,
    )
    freeform_mask = st.session_state.get("freeform_optical_mask")
    if isinstance(freeform_mask, np.ndarray) and freeform_mask.shape != (info.height, info.width):
        freeform_mask = None
        st.session_state.pop("freeform_optical_mask", None)
    auto_pattern_mask = st.session_state.get("auto_pattern_mask")
    if isinstance(auto_pattern_mask, np.ndarray) and auto_pattern_mask.shape != (info.height, info.width):
        auto_pattern_mask = None
        st.session_state.pop("auto_pattern_mask", None)

    with left_col:
        st.header("Preview")
        optical_roi_mode = st.session_state.get("optical_roi_mode", "Dual-ring ROI")
        overlay = draw_rois(
            preview_frame,
            current_gray_roi,
            current_temp_roi,
            optical_roi_mode=optical_roi_mode,
            ring_roi=current_ring_roi,
            freeform_mask=freeform_mask,
            auto_pattern_mask=auto_pattern_mask,
            search_roi=current_search_roi,
        )
        st.image(
            overlay,
            caption="Green: optical ROI. Inner circle/sample, outer ring/background. Blue/orange: temperature OCR ROI.",
            use_container_width=True,
        )
        canvas_roi_selector(preview_frame, info.width, info.height)

    with right_col:
        st.header("ROI controls")
        auto_detection_settings = None
        optical_roi_mode = st.selectbox(
            "Optical analysis ROI mode",
            ["Auto dark-edge ROI", "Auto-detected pattern ROI", "Free-drawn ROI", "Dual-ring ROI", "Square ROI"],
            index=0,
            key="optical_roi_mode",
            help="Auto dark-edge ROI detects black ring/line edges inside the bright microscope field.",
        )
        if optical_roi_mode == "Auto dark-edge ROI":
            ring_roi = current_ring_roi
            search_roi = number_input_roi(
                "Auto-detection search ROI",
                info.width,
                info.height,
                default_x=current_search_roi[0],
                default_y=current_search_roi[1],
                default_w=current_search_roi[2],
                default_h=current_search_roi[3],
                square=False,
                key_prefix="search_roi",
            )
            st.caption("Auto detection only runs inside this yellow box. Use it to exclude the bottom text overlay and black vignette.")
            st.subheader("Auto dark-edge detection")
            edge_col1, edge_col2 = st.columns(2)
            with edge_col1:
                edge_threshold = st.slider("Dark edge strength", 2, 80, 12, 1)
                edge_dilation = st.slider("Edge thickness", 0, 5, 1, 1)
            with edge_col2:
                edge_min_area = st.number_input("Min edge component area", min_value=5, value=25, step=5)
                edge_max_components = st.number_input("Max edge components", min_value=1, value=80, step=1)
            auto_detection_settings = {
                "search_roi": search_roi,
                "edge_threshold": float(edge_threshold),
                "min_area": int(edge_min_area),
                "max_components": int(edge_max_components),
                "dilation": int(edge_dilation),
            }
            if st.button("Detect dark edges from preview frame", use_container_width=True):
                auto_pattern_mask = create_auto_edge_pattern_mask(
                    preview_frame,
                    temp_roi=current_temp_roi,
                    search_roi=search_roi,
                    edge_threshold=float(edge_threshold),
                    min_area=int(edge_min_area),
                    max_components=int(edge_max_components),
                    dilation=int(edge_dilation),
                )
                st.session_state.auto_pattern_mask = auto_pattern_mask
                effective_area = effective_pattern_area(auto_pattern_mask, "Auto dark-edge ROI")
                st.success(f"Detected dark-edge ROI with {effective_area:,.0f} effective px.")
                st.rerun()
            if auto_pattern_mask is None:
                st.warning("No dark-edge ROI has been generated yet. Click the detection button above.")
            else:
                effective_area = effective_pattern_area(auto_pattern_mask, "Auto dark-edge ROI")
                st.success(f"Auto dark-edge ROI active: {effective_area:,.0f} effective px.")
                if st.button("Clear auto dark-edge ROI", use_container_width=True):
                    st.session_state.pop("auto_pattern_mask", None)
                    st.rerun()
        elif optical_roi_mode == "Auto-detected pattern ROI":
            ring_roi = current_ring_roi
            search_roi = number_input_roi(
                "Auto-detection search ROI",
                info.width,
                info.height,
                default_x=current_search_roi[0],
                default_y=current_search_roi[1],
                default_w=current_search_roi[2],
                default_h=current_search_roi[3],
                square=False,
                key_prefix="search_roi",
            )
            st.caption("Auto detection only runs inside this yellow box.")
            st.subheader("Auto pattern detection")
            pattern_col1, pattern_col2 = st.columns(2)
            with pattern_col1:
                pattern_polarity = st.selectbox(
                    "Pattern type",
                    ["Darker than background", "Brighter than background"],
                    index=0,
                )
                pattern_delta = st.slider("Background difference threshold", 2, 80, 18, 1)
            with pattern_col2:
                pattern_min_area = st.number_input("Min component area", min_value=5, value=60, step=5)
                pattern_max_components = st.number_input("Max components", min_value=1, value=20, step=1)
            auto_detection_settings = {
                "search_roi": search_roi,
                "polarity": pattern_polarity,
                "threshold_delta": float(pattern_delta),
                "min_area": int(pattern_min_area),
                "max_components": int(pattern_max_components),
            }
            if st.button("Detect pattern ROI from preview frame", use_container_width=True):
                auto_pattern_mask = create_auto_pattern_mask(
                    preview_frame,
                    temp_roi=current_temp_roi,
                    search_roi=search_roi,
                    polarity=pattern_polarity,
                    threshold_delta=float(pattern_delta),
                    min_area=int(pattern_min_area),
                    max_components=int(pattern_max_components),
                )
                st.session_state.auto_pattern_mask = auto_pattern_mask
                effective_area = effective_pattern_area(auto_pattern_mask, "Auto-detected pattern ROI")
                st.success(f"Detected pattern ROI with {effective_area:,.0f} effective px.")
                st.rerun()
            if auto_pattern_mask is None:
                st.warning("No auto pattern ROI has been generated yet. Click the detection button above.")
            else:
                effective_area = effective_pattern_area(auto_pattern_mask, "Auto-detected pattern ROI")
                st.success(f"Auto pattern ROI active: {effective_area:,.0f} effective px.")
                if st.button("Clear auto pattern ROI", use_container_width=True):
                    st.session_state.pop("auto_pattern_mask", None)
                    st.rerun()
        elif optical_roi_mode == "Free-drawn ROI":
            ring_roi = current_ring_roi
            if freeform_mask is None:
                st.warning("No free-drawn ROI has been saved yet. Draw it in the left panel and click Apply drawn ROI.")
            else:
                st.success(f"Free-drawn ROI active: {int(freeform_mask.sum()):,} pixels.")
                if st.button("Clear free-drawn ROI", use_container_width=True):
                    st.session_state.pop("freeform_optical_mask", None)
                    st.rerun()
        elif optical_roi_mode == "Dual-ring ROI":
            ring_roi = number_input_ring_roi(
                info.width,
                info.height,
                default_cx=current_ring_roi[0],
                default_cy=current_ring_roi[1],
                default_inner_r=current_ring_roi[2],
                default_outer_r=current_ring_roi[3],
            )
            st.caption("Inner circle = sample; outer ring = nearby background. Move X/Y shifts the whole dual-ring ROI.")
        else:
            ring_roi = current_ring_roi
        gray_roi = number_input_roi(
            "Grayscale ROI",
            info.width,
            info.height,
            default_x=current_gray_roi[0],
            default_y=current_gray_roi[1],
            default_w=current_gray_roi[2],
            default_h=current_gray_roi[3],
            square=True,
            key_prefix="gray_roi",
        )
        st.caption("The app calculates the mean gray value across this square region, not a single pixel.")

        temp_roi = number_input_roi(
            "Temperature OCR ROI",
            info.width,
            info.height,
            default_x=current_temp_roi[0],
            default_y=current_temp_roi[1],
            default_w=current_temp_roi[2],
            default_h=current_temp_roi[3],
            square=False,
            key_prefix="temp_roi",
        )

    st.divider()

    ocr_col, settings_col = st.columns([1, 1])

    with settings_col:
        st.header("OCR preprocessing")
        tesseract_cmd = st.text_input(
            "Tesseract executable path",
            value=st.session_state.get("tesseract_cmd", default_tesseract_path()),
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe or /usr/bin/tesseract",
            help="Hosted Linux usually uses /usr/bin/tesseract. Local Windows usually uses C:\\Program Files\\Tesseract-OCR\\tesseract.exe.",
        )
        st.session_state.tesseract_cmd = tesseract_cmd
        tesseract_ok, tesseract_message = get_tesseract_status(tesseract_cmd)
        if tesseract_ok:
            st.success(tesseract_message)
        else:
            st.error(tesseract_message)
            st.caption("Install Tesseract OCR on the server/local machine, or paste the full executable path above.")

        preprocess_scale = st.slider("OCR resize scale", 2, 8, 3, 1)
        contrast_factor = st.slider("OCR contrast boost", 1.0, 4.0, 2.0, 0.1)
        blur_ocr = st.checkbox("Slight blur before thresholding", value=False)
        threshold_method = st.selectbox("OCR image mode", ["Contrast only", "None", "Otsu", "Adaptive"], index=0)
        invert_ocr = st.checkbox("Invert OCR image", value=False)
        psm = st.selectbox(
            "Tesseract page segmentation mode",
            [7, 6, 8, 13],
            index=0,
            help="7 is usually best for one line of temperature text.",
        )
        temperature_parse_mode = st.selectbox(
            "Temperature parsing mode",
            ["Between Temp and C", "Only sample number in ROI", "Sample Temp after label", "First value followed by C", "First number in ROI"],
            index=0,
            help="Strictly reads only values in text like 'Temp 4.1°C'. If Temp/C are not recognized, the frame is treated as missing.",
        )
        st.caption("Best practice now: draw the blue box tightly around `Temp 4.1°C`, from `Temp` through `°C`. Other numbers in the frame are ignored.")
        assume_one_decimal = st.checkbox(
            "Restore missing decimal point for one-decimal display",
            value=True,
            help="If OCR reads 4.0 as 40 or 15.9 as 159, insert the decimal before the last digit.",
        )
        use_all_preprocessing = st.checkbox(
            "Try multiple preprocessing methods when reading temperature",
            value=False,
            help="Slower, but useful for debugging difficult videos. Leave off for clean software overlays.",
        )

        st.header("Analysis settings")
        analyze_partial = st.checkbox("Analyze only part of the video", value=False)
        if analyze_partial:
            start_time_s, end_time_s = st.slider(
                "Time range to analyze (seconds)",
                min_value=0.0,
                max_value=max(0.0, float(info.duration_s)),
                value=(0.0, max(0.0, float(info.duration_s))),
                step=max(0.001, 1.0 / info.fps),
            )
        else:
            start_time_s, end_time_s = 0.0, info.duration_s

        start_frame = int(np.clip(round(start_time_s * info.fps), 0, max(0, info.frame_count - 1)))
        end_frame = int(np.clip(round(end_time_s * info.fps), start_frame, max(0, info.frame_count - 1)))
        st.caption(f"Frame range: {start_frame} to {end_frame}")

        frame_step = st.number_input(
            "Analyze every Nth frame",
            min_value=1,
            max_value=max(1, info.frame_count),
            value=1,
            step=1,
        )
        limit_frames = st.checkbox("Limit number of processed frames", value=False)
        max_frames = None
        if limit_frames:
            max_frames = st.number_input(
                "Maximum frames to process",
                min_value=1,
                max_value=max(1, info.frame_count),
                value=min(500, max(1, info.frame_count)),
                step=1,
            )

        st.header("Temperature cleanup")
        clean_temperature = st.checkbox("Remove obvious OCR temperature errors", value=True)
        expected_trend = st.selectbox(
            "Expected temperature trend",
            ["Mostly smooth", "Monotonic cooling", "Monotonic warming"],
            index=0,
        )
        cleanup_col1, cleanup_col2, cleanup_col3, cleanup_col4 = st.columns(4)
        with cleanup_col1:
            min_temperature = st.number_input("Min valid temp (C)", value=-50.0, step=1.0)
        with cleanup_col2:
            max_temperature = st.number_input("Max valid temp (C)", value=100.0, step=1.0)
        with cleanup_col3:
            max_temperature_jump = st.number_input(
                "Max jump (C)",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="Set 0 to disable jump filtering.",
            )
        with cleanup_col4:
            min_confidence = st.number_input(
                "Min confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
            )

        st.header("Template fallback")
        template_threshold = st.slider(
            "Template match threshold",
            0.50,
            0.99,
            0.92,
            0.01,
            help="Used only if you add manually labeled temperature templates.",
        )

    with ocr_col:
        st.header("OCR preview")
        tx, ty, tw, th = temp_roi
        temp_crop = preview_frame[ty : ty + th, tx : tx + tw]

        if temp_crop.size == 0:
            st.warning("Temperature ROI is empty. Adjust its position or size.")
        else:
            ocr_text, parsed_temp, ocr_method, preview_confidence, best_processed_temp, preview_candidates = robust_ocr_temperature(
                temp_crop,
                scale=preprocess_scale,
                blur=blur_ocr,
                threshold_method=threshold_method,
                invert=invert_ocr,
                contrast_factor=contrast_factor,
                psm=psm,
                parse_mode=temperature_parse_mode,
                assume_one_decimal=assume_one_decimal,
                templates=st.session_state.get("temperature_templates", []),
                template_threshold=template_threshold,
                previous_value=np.nan,
                min_temp=float(min_temperature),
                max_temp=float(max_temperature),
                max_jump=float(max_temperature_jump),
                expected_trend=expected_trend,
                use_all_preprocessing=use_all_preprocessing,
            )

            raw_rgb = cv2.cvtColor(temp_crop, cv2.COLOR_BGR2RGB)
            col_raw, col_processed = st.columns(2)
            col_raw.image(raw_rgb, caption="Raw temperature ROI", use_container_width=True)
            col_processed.image(
                Image.fromarray(best_processed_temp),
                caption=f"Processed for OCR ({ocr_method})",
                use_container_width=True,
            )

            st.write("OCR text:", f"`{ocr_text}`" if ocr_text else "`<empty>`")
            st.write("Confidence:", f"`{preview_confidence:.2f}`")
            if np.isnan(parsed_temp):
                st.warning("No valid temperature number was detected in this preview frame.")
                st.caption("Try drawing the blue OCR box tightly around `Temp 4.1°C`, then use `Between Temp and C`.")
            else:
                st.success(f"Parsed preview temperature: {parsed_temp:g} C")

            st.subheader("Manual template calibration")
            st.caption("Optional: label a few clear frames so the app can match the same software-rendered overlay without OCR.")
            calib_col1, calib_col2, calib_col3 = st.columns([1, 1, 1])
            with calib_col1:
                manual_temp_value = st.number_input(
                    "Correct temp for this preview frame",
                    value=float(parsed_temp) if not np.isnan(parsed_temp) else 0.0,
                    step=0.1,
                    format="%.3f",
                )
            with calib_col2:
                if st.button("Add current ROI as template", use_container_width=True):
                    add_manual_temperature_template(preview_frame, temp_roi, manual_temp_value)
                    st.success(f"Added template for {manual_temp_value:g} C")
            with calib_col3:
                if st.button("Clear templates", use_container_width=True):
                    st.session_state.temperature_templates = []
                    st.success("Cleared temperature templates.")
            st.caption(f"Stored templates: {len(st.session_state.temperature_templates)}")

            debug_temperature_panel(
                video_path=info.path,
                info=info,
                temp_roi=temp_roi,
                preview_frame=preview_frame,
                preprocess_scale=preprocess_scale,
                blur_ocr=blur_ocr,
                threshold_method=threshold_method,
                invert_ocr=invert_ocr,
                contrast_factor=contrast_factor,
                psm=psm,
                temperature_parse_mode=temperature_parse_mode,
                assume_one_decimal=assume_one_decimal,
                template_threshold=template_threshold,
                min_temperature=float(min_temperature),
                max_temperature=float(max_temperature),
                max_temperature_jump=float(max_temperature_jump),
                expected_trend=expected_trend,
                use_all_preprocessing=use_all_preprocessing,
            )

    st.divider()

    run_analysis = st.button("Run analysis", type="primary", use_container_width=True)

    if run_analysis:
        if not tesseract_ok:
            st.error("Cannot run temperature OCR because Tesseract is not available. Set the correct tesseract.exe path first.")
            return
        if optical_roi_mode == "Free-drawn ROI" and freeform_mask is None:
            st.error("Draw and apply a free-drawn optical ROI before running analysis, or switch to Dual-ring/Square ROI mode.")
            return
        if optical_roi_mode in {"Auto-detected pattern ROI", "Auto dark-edge ROI"} and auto_detection_settings is None:
            st.error("Choose auto-detection settings before running analysis, or switch to another ROI mode.")
            return
        try:
            with st.spinner("Analyzing video frames..."):
                results = process_video(
                    video_path=info.path,
                    info=info,
                    gray_roi=gray_roi,
                    optical_roi_mode=optical_roi_mode,
                    ring_roi=ring_roi,
                    freeform_mask=freeform_mask,
                    auto_pattern_mask=auto_pattern_mask,
                    temp_roi=temp_roi,
                    frame_step=int(frame_step),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    max_frames=int(max_frames) if max_frames is not None else None,
                    preprocess_scale=int(preprocess_scale),
                    blur_ocr=bool(blur_ocr),
                    threshold_method=threshold_method,
                    invert_ocr=bool(invert_ocr),
                    contrast_factor=float(contrast_factor),
                    psm=int(psm),
                    temperature_parse_mode=temperature_parse_mode,
                    assume_one_decimal=bool(assume_one_decimal),
                    expected_trend=expected_trend,
                    use_all_preprocessing=bool(use_all_preprocessing),
                    template_threshold=float(template_threshold),
                    manual_templates=st.session_state.get("temperature_templates", []),
                    clean_temperature=bool(clean_temperature),
                    min_temperature=float(min_temperature),
                    max_temperature=float(max_temperature),
                    max_temperature_jump=float(max_temperature_jump),
                    min_confidence=float(min_confidence),
                    auto_detection_settings=auto_detection_settings,
                )
            st.session_state.results = results
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    results = st.session_state.get("results")
    if results is None or results.empty:
        return

    st.header("Results")

    if "temperature_C_raw" not in results.columns:
        results["temperature_C_raw"] = results.get("temperature_C", np.nan)
    if "confidence" not in results.columns:
        results["confidence"] = np.nan

    raw_missing = int(results["temperature_C_raw"].isna().sum())
    total = len(results)
    fail_rate = raw_missing / total if total else 0.0
    repaired_count = int(results.get("temperature_repaired", pd.Series(False, index=results.index)).fillna(False).sum())
    low_confidence_count = int(results.get("low_confidence", pd.Series(False, index=results.index)).fillna(False).sum())
    low_confidence_rate = low_confidence_count / total if total else 0.0

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Processed frames", f"{total:,}")
    metric_col2.metric("Successful temp reads", f"{total - raw_missing:,}")
    metric_col3.metric("OCR missing rate", f"{fail_rate:.1%}")
    metric_col4.metric("Repaired frames", f"{repaired_count:,}")
    metric_col5.metric("Low confidence", f"{low_confidence_rate:.1%}")

    if raw_missing == total:
        st.error("OCR did not read any temperatures. Adjust the temperature ROI or preprocessing settings.")
        st.caption("For best results, draw the blue OCR box tightly around `Temp 4.1°C` so both `Temp` and `°C` are visible.")
    elif fail_rate > 0.35:
        st.warning("Many temperature readings failed. Check ROI placement, thresholding, inversion, and Tesseract setup.")
    elif fail_rate > 0.10:
        st.warning("Some temperature readings failed. Missing values were interpolated where possible.")
    if low_confidence_rate > 0.20:
        st.warning("Many frames have low recognition confidence. Inspect the temperature debug panel and ROI placement.")
    st.info(
        "`gray_value` uses the image convention 0=black and 255=white. "
        "`pattern_area_px` is a noise-resistant effective area; Auto dark-edge ROI uses contour length. "
        "In auto-detection modes, the mask is recalculated for every analyzed frame."
    )

    display_cols = [
        "frame",
        "time_s",
        "temperature_C",
        "temperature_C_clean",
        "temperature_C_raw",
        "pattern_area_px",
        "pattern_area_fraction",
        "pattern_area_norm_first",
        "sample_gray",
        "background_gray",
        "background_corrected_darkness",
        "gray_value",
        "darkness_value",
        "dissolution_signal",
        "gray_norm_first",
        "dissolution_norm_first",
        "raw_text",
        "method",
        "confidence",
        "temperature_repaired",
        "low_confidence",
    ]
    for column in display_cols:
        if column not in results.columns:
            results[column] = "" if column in {"raw_text", "method"} else np.nan
    st.dataframe(results[display_cols], use_container_width=True, height=360)

    suspicious_cols = ["frame", "time_s", "temperature_C_raw", "temperature_C", "raw_text", "method", "confidence"]
    suspicious_mask = (
        results.get("low_confidence", pd.Series(False, index=results.index)).fillna(False)
        | results.get("temperature_repaired", pd.Series(False, index=results.index)).fillna(False)
        | results["temperature_C_raw"].isna()
    )
    suspicious = results.loc[suspicious_mask, [col for col in suspicious_cols if col in results.columns]]
    if not suspicious.empty:
        with st.expander("Suspicious temperature frames", expanded=False):
            st.dataframe(suspicious, use_container_width=True, height=260)

    plot_col, export_col = st.columns([1, 2])
    with export_col:
        st.header("Plot controls")
        publication_style = st.checkbox("Publication-style plot", value=True)
        normalization_mode = st.selectbox(
            "Normalization",
            ["First frame = 1", "Min-max 0-1", "None"],
            index=0,
            help="The LCST paper normalizes grayscale values to the first image.",
        )
        y_metric = st.selectbox(
            "Y-axis signal",
            [
                "Pattern area",
                "Dissolution signal",
                "Mean gray value",
                "Darkness (255 - gray)",
            ],
            index=0,
            help="Pattern area is usually best for images that visibly disappear as temperature changes.",
        )
        if y_metric == "Pattern area":
            y_column = "pattern_area_px"
            y_label = "Pattern area (px)"
        elif y_metric == "Dissolution signal":
            y_column = "dissolution_signal"
            y_label = "Dissolution signal"
        elif y_metric.startswith("Darkness"):
            y_column = "darkness_value"
            y_label = "Darkness (255 - mean gray)"
        else:
            y_column = "gray_value"
            y_label = "Mean gray value"

        invert_x_axis = st.checkbox("Invert x-axis for cooling", value=True)
        smoothing_window = st.slider("Moving-average smoothing window", 1, 51, 1, 2)
        if smoothing_window % 2 == 0:
            smoothing_window += 1
            st.caption(f"Using odd smoothing window: {smoothing_window}")
        temperature_aggregation = st.selectbox(
            "Repeated temperatures",
            ["Mean", "Median", "First", "Raw points"],
            index=0,
            help="Mean collapses repeated temperatures into one point, so one x-value has only one y-value.",
        )
        show_lcst = st.checkbox("Estimate LCST", value=True)
        lcst_method = st.selectbox(
            "LCST method",
            ["50% pattern disappearance", "Inflection point"],
            index=0 if y_metric == "Pattern area" else 1,
            help="50% method uses the temperature where the stabilized signal reaches the midpoint between the initial plateau and final state.",
        )

        lcst = estimate_lcst(
            results,
            y_column,
            smoothing_window,
            normalization_mode,
            lcst_method,
            temperature_aggregation,
        )
        if show_lcst and not np.isnan(lcst["lcst_C"]):
            st.metric("Estimated LCST", f"{lcst['lcst_C']:.2f} °C")
            if lcst_method == "50% pattern disappearance":
                st.caption(
                    "50% method: LCST is where the stabilized signal reaches the midpoint between the initial plateau and final state."
                )
            else:
                st.caption("Inflection method: LCST is estimated from the inflection point of the smoothed signal-temperature curve.")
        elif show_lcst:
            st.caption("Not enough valid points to estimate LCST.")

        csv = results[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="gray_value_temperature_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with plot_col:
        valid_points = results.dropna(subset=["temperature_C", y_column])
        if valid_points.empty:
            st.warning("No valid points are available for plotting.")
        else:
            fig = make_plot(
                results,
                invert_x_axis=invert_x_axis,
                smoothing_window=smoothing_window,
                y_column=y_column,
                y_label=y_label,
                normalization_mode=normalization_mode,
                publication_style=publication_style,
                show_lcst=show_lcst,
                lcst_method=lcst_method,
                temperature_aggregation=temperature_aggregation,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=False)


def main() -> None:
    st.set_page_config(
        page_title="Gray Value Reader Helper",
        layout="wide",
    )
    render_simplified_video_mode()


if __name__ == "__main__":
    main()
