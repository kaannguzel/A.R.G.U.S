#!/usr/bin/env python3
"""People-focused motion detector for wide CCTV scenes."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect motion and save annotated outputs + CSV logs.")

    p.add_argument("--inputs", nargs="+", default=["*.mp4","*.avi"], help="Input files or glob patterns.")
    p.add_argument("--output-dir", default="motion_output", help="Directory for output files.")

    p.add_argument("--show", dest="show", action="store_true", help="Show live preview.")
    p.add_argument("--no-show", dest="show", action="store_false", help="Disable live preview.")
    p.set_defaults(show=True)
    p.add_argument("--preview-mode", choices=["fg", "delta", "both"], default="both", help="Preview mask.")
    p.add_argument("--debug-stages", action="store_true", help="Show DIFF/BLUR/THRESH windows.")
    p.add_argument("--preview-fps", type=float, default=0.0, help="Preview FPS override (0 = source FPS).")
    p.add_argument("--preview-speed", type=float, default=1.1, help="Preview speed multiplier.")

    p.add_argument("--save-mask", action="store_true", help="Save mask video.")
    p.add_argument("--warmup-sec", type=float, default=0.2, help="Warmup seconds.")
    p.add_argument("--process-scale", type=float, default=1.0, help="Processing scale (0.5-1.0 typical).")

    # Background modeling
    p.add_argument("--bg-method", choices=["knn", "mog2"], default="mog2", help="Background subtractor.")
    p.add_argument("--history", type=int, default=300, help="Background history.")
    p.add_argument("--threshold", type=int, default=80, help="Background threshold.")
    p.add_argument("--learning-rate", type=float, default=0.003, help="Background learning rate.")
    p.add_argument("--fg-threshold", type=int, default=205, help="Binary foreground threshold.")
    p.add_argument("--use-running-bg", action="store_true", help="Use running-average background for frame differencing.")
    p.add_argument("--running-bg-alpha", type=float, default=0.01, help="Running-average alpha.")

    # Noise / illumination
    p.add_argument("--blur-kernel", type=int, default=5, help="Gaussian blur kernel (odd).")
    p.add_argument("--illumination-comp", dest="illumination_comp", action="store_true", help="Use CLAHE.")
    p.add_argument("--no-illumination-comp", dest="illumination_comp", action="store_false", help="Disable CLAHE.")
    p.set_defaults(illumination_comp=True)
    p.add_argument("--suppress-sunlight", dest="suppress_sunlight", action="store_true", help="Mask bright low-sat glare.")
    p.add_argument("--no-suppress-sunlight", dest="suppress_sunlight", action="store_false", help="Disable glare mask.")
    p.set_defaults(suppress_sunlight=True)
    p.add_argument("--sun-value-thresh", type=int, default=205, help="HSV V threshold for glare mask.")
    p.add_argument("--sun-sat-max", type=int, default=70, help="HSV S max for glare mask.")
    p.add_argument("--bright-glare-thresh", type=int, default=235, help="Gray threshold for generic glare mask.")
    p.add_argument("--illum-change-thresh", type=float, default=12.0, help="Mean-brightness jump treated as lighting event.")

    # Perspective-aware contour thresholds
    p.add_argument("--min-area-ratio", type=float, default=0.00015, help="Near contour area ratio.")
    p.add_argument("--far-min-area-ratio", type=float, default=0.00010, help="Far contour area ratio.")
    p.add_argument("--min-contour-area", type=int, default=30, help="Near absolute contour area.")
    p.add_argument("--far-min-contour-area", type=int, default=12, help="Far absolute contour area.")
    p.add_argument("--min-width-ratio", type=float, default=0.018, help="Near min bbox width ratio.")
    p.add_argument("--min-height-ratio", type=float, default=0.040, help="Near min bbox height ratio.")
    p.add_argument("--min-fill-ratio", type=float, default=0.16, help="Near min fill ratio (area/(w*h)).")
    p.add_argument("--far-min-fill-ratio", type=float, default=0.05, help="Far min fill ratio.")
    p.add_argument("--min-solidity", type=float, default=0.32, help="Near min solidity.")
    p.add_argument("--far-min-solidity", type=float, default=0.11, help="Far min solidity.")

    # Delta + morphology + fusion
    p.add_argument("--delta-threshold", type=int, default=20, help="Delta threshold (20-25 helps distant people).")
    p.add_argument("--min-delta-ratio", type=float, default=0.0012, help="Min changed-pixel ratio in contour.")
    p.add_argument("--min-delta-pixels", type=int, default=10, help="Near min changed pixels.")
    p.add_argument("--far-min-delta-pixels", type=int, default=4, help="Far min changed pixels.")
    p.add_argument("--min-delta-area-ratio", type=float, default=0.000008, help="Near min changed-area ratio vs frame area.")
    p.add_argument("--far-min-delta-area-ratio", type=float, default=0.000003, help="Far min changed-area ratio vs frame area.")
    p.add_argument("--morph-kernel", type=int, default=3, help="Morphology kernel size.")
    p.add_argument("--near-open-iter", type=int, default=1, help="Near-zone OPEN iterations.")
    p.add_argument("--near-close-iter", type=int, default=1, help="Near-zone CLOSE iterations.")
    p.add_argument("--near-dilate-iter", type=int, default=0, help="Near-zone DILATE iterations.")
    p.add_argument("--far-open-iter", type=int, default=0, help="Far-zone OPEN iterations.")
    p.add_argument("--far-close-iter", type=int, default=1, help="Far-zone CLOSE iterations.")
    p.add_argument("--far-dilate-iter", type=int, default=1, help="Far-zone DILATE iterations.")
    p.add_argument("--temporal-fusion-frames", type=int, default=2, help="OR-fuse N recent masks.")
    p.add_argument("--far-y-ratio", type=float, default=0.55, help="Horizon split ratio; above this line is far zone.")

    # Global motion decision
    p.add_argument("--debounce-on", type=int, default=2, help="Frames to turn motion ON.")
    p.add_argument("--debounce-off", type=int, default=3, help="Frames to turn motion OFF.")
    p.add_argument("--total-area-ratio", type=float, default=0.0008, help="Total area ratio fallback threshold.")
    p.add_argument("--motion-by-box", dest="motion_by_box", action="store_true", help="Declare motion if at least one valid box exists.")
    p.add_argument("--no-motion-by-box", dest="motion_by_box", action="store_false", help="Disable box-based motion trigger.")
    p.set_defaults(motion_by_box=True)

    # Stability / tracking
    p.add_argument("--static-hold-frames", type=int, default=7, help="Suppress static ghost tracks after N frames.")
    p.add_argument("--static-max-shift", type=float, default=1.5, help="Max center shift considered static.")
    p.add_argument("--far-static-disable-weight", type=float, default=0.45, help="Disable static suppression for far objects.")
    p.add_argument("--track-hold-frames", type=int, default=3, help="Keep recent boxes alive for N frames.")
    p.add_argument("--far-confirm-frames", type=int, default=2, help="Require N consecutive far detections before boxing.")

    # ROI
    p.add_argument("--use-roi", dest="use_roi", action="store_true", help="Use default central ROI.")
    p.add_argument("--no-use-roi", dest="use_roi", action="store_false", help="Disable default central ROI.")
    p.set_defaults(use_roi=True)
    p.add_argument("--roi-rel", default="", help="Relative ROI x1,y1,x2,y2 in [0..1].")

    # Person detection preference
    p.add_argument("--people-only", dest="people_only", action="store_true", help="Keep mostly person-like detections.")
    p.add_argument("--no-people-only", dest="people_only", action="store_false", help="Disable person filtering.")
    p.set_defaults(people_only=False)
    p.add_argument("--person-detector", choices=["hog", "yolo", "none"], default="none", help="Person detector backend.")
    p.add_argument("--person-detect-every", type=int, default=3, help="Run person detector every N frames.")
    p.add_argument("--person-overlap-ratio", type=float, default=0.30, help="Min overlap with person box for near/mid boxes.")
    p.add_argument("--far-human-min-ratio", type=float, default=1.25, help="Far min h/w ratio.")
    p.add_argument("--far-human-max-ratio", type=float, default=5.2, help="Far max h/w ratio.")
    p.add_argument("--far-human-min-height", type=int, default=8, help="Far min pixel height.")
    p.add_argument("--yolo-cfg", default="", help="YOLO Darknet cfg path.")
    p.add_argument("--yolo-weights", default="", help="YOLO Darknet weights path.")
    p.add_argument("--yolo-input-size", type=int, default=416, help="YOLO input size.")
    p.add_argument("--yolo-confidence", type=float, default=0.35, help="YOLO confidence.")
    p.add_argument("--yolo-nms", type=float, default=0.45, help="YOLO NMS threshold.")

    return p.parse_args()


def resolve_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for token in patterns:
        p = Path(token)
        if p.is_file():
            files.append(p)
        else:
            files.extend(sorted(Path(".").glob(token)))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for f in files:
        k = f.resolve()
        if k in seen or not f.is_file():
            continue
        seen.add(k)
        deduped.append(f)
    return deduped


def build_sun_mask(hsv: np.ndarray, v_thresh: int, s_max: int) -> np.ndarray:
    m = cv2.inRange(
        hsv,
        np.array([0, 0, int(v_thresh)], dtype=np.uint8),
        np.array([180, int(s_max), 255], dtype=np.uint8),
    )
    return cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=1)


def apply_zone_morph(
    mask: np.ndarray,
    far_y: int,
    kernel: np.ndarray,
    near_open: int,
    near_close: int,
    near_dilate: int,
    far_open: int,
    far_close: int,
    far_dilate: int,
) -> np.ndarray:
    near = mask.copy()
    far = mask.copy()
    near[:far_y, :] = 0
    far[far_y:, :] = 0

    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, kernel, iterations=max(0, int(near_open)))
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, kernel, iterations=max(0, int(near_close)))
    if int(near_dilate) > 0:
        near = cv2.dilate(near, kernel, iterations=int(near_dilate))

    far = cv2.morphologyEx(far, cv2.MORPH_OPEN, kernel, iterations=max(0, int(far_open)))
    far = cv2.morphologyEx(far, cv2.MORPH_CLOSE, kernel, iterations=max(0, int(far_close)))
    if int(far_dilate) > 0:
        far = cv2.dilate(far, kernel, iterations=int(far_dilate))

    return cv2.bitwise_or(near, far)


def overlap_ratio(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    return inter / float(max(1, aw * ah))


def process_video(video_path: Path, output_dir: Path, args: argparse.Namespace) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = src_fps if 5.0 <= float(src_fps) <= 240.0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    warmup_frames = max(0, int(fps * args.warmup_sec))
    preview_fps = float(args.preview_fps) if float(args.preview_fps) > 0.0 else float(fps)
    frame_delay_ms = max(1, int(1000 / max(0.1, preview_fps * max(0.1, float(args.preview_speed)))))

    min_area_near = max(int(args.min_contour_area), int(frame_area * float(args.min_area_ratio)))
    min_area_far = max(int(args.far_min_contour_area), int(frame_area * float(args.far_min_area_ratio)))
    min_delta_near = max(1, int(args.min_delta_pixels))
    min_delta_far = max(1, int(args.far_min_delta_pixels))
    total_area_thresh = int(frame_area * float(args.total_area_ratio))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = output_dir / f"{video_path.stem}_motion.mp4"
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    mask_writer = None
    if args.save_mask:
        mask_video = output_dir / f"{video_path.stem}_mask.mp4"
        mask_writer = cv2.VideoWriter(str(mask_video), fourcc, fps, (width, height), False)

    csv_path = output_dir / f"{video_path.stem}_motion.csv"

    if args.bg_method == "mog2":
        subtractor = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=args.threshold, detectShadows=False)
    else:
        subtractor = cv2.createBackgroundSubtractorKNN(history=args.history, dist2Threshold=args.threshold, detectShadows=False)

    hog = None
    yolo_model = None
    if args.people_only and args.person_detector == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    elif args.people_only and args.person_detector == "yolo":
        if not args.yolo_cfg or not args.yolo_weights:
            raise RuntimeError("YOLO selected but --yolo-cfg/--yolo-weights were not provided.")
        net = cv2.dnn.readNetFromDarknet(args.yolo_cfg, args.yolo_weights)
        yolo_model = cv2.dnn_DetectionModel(net)
        yolo_model.setInputParams(scale=1.0 / 255.0, size=(int(args.yolo_input_size), int(args.yolo_input_size)), swapRB=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    roi_mask = None
    if args.use_roi:
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[int(height * 0.08):int(height * 0.99), int(width * 0.05):int(width * 0.95)] = 255
    if args.roi_rel:
        parts = [s.strip() for s in str(args.roi_rel).split(",")]
        if len(parts) == 4:
            try:
                x1f, y1f, x2f, y2f = map(float, parts)
                x1 = int(np.clip(x1f, 0.0, 1.0) * width)
                y1 = int(np.clip(y1f, 0.0, 1.0) * height)
                x2 = int(np.clip(x2f, 0.0, 1.0) * width)
                y2 = int(np.clip(y2f, 0.0, 1.0) * height)
                if x2 > x1 and y2 > y1:
                    roi_mask = np.zeros((height, width), dtype=np.uint8)
                    roi_mask[y1:y2, x1:x2] = 255
            except ValueError:
                pass

    mk = max(1, int(args.morph_kernel))
    if mk % 2 == 0:
        mk += 1
    morph_kernel = np.ones((mk, mk), np.uint8)

    prev_gray: np.ndarray | None = None
    running_gray: np.ndarray | None = None
    prev_mean_brightness: float | None = None
    person_boxes_cache: list[tuple[int, int, int, int]] = []
    static_tracks: list[tuple[float, float, int]] = []
    persist_tracks: list[tuple[int, int, int, int, int]] = []
    fusion_queue: deque[np.ndarray] = deque(maxlen=max(1, int(args.temporal_fusion_frames)))

    frame_idx = 0
    detected_frames = 0
    motion_state = False
    on_count = 0
    off_count = 0
    interrupted = False
    startup_frames = max(1, int(fps * 1.0))

    with csv_path.open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "time_sec", "motion", "regions", "max_region_area", "motion_pixels"])

        window_name = f"Motion Detection - {video_path.name}"

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            scale = float(args.process_scale)
            proc_frame = frame
            if 0.1 <= scale < 1.0:
                proc_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            bk = max(3, int(args.blur_kernel))
            if bk % 2 == 0:
                bk += 1
            blurred = cv2.GaussianBlur(proc_frame, (bk, bk), 0)

            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            if args.illumination_comp:
                gray = clahe.apply(gray)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            far_y = int(height * float(args.far_y_ratio))

            # Person detector cache
            if args.people_only and (frame_idx % max(1, int(args.person_detect_every)) == 0):
                if hog is not None:
                    det_rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    if 0.1 <= scale < 1.0:
                        inv = 1.0 / scale
                        person_boxes_cache = [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in det_rects]
                    else:
                        person_boxes_cache = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in det_rects]
                elif yolo_model is not None:
                    class_ids, _scores, boxes = yolo_model.detect(
                        blurred,
                        confThreshold=float(args.yolo_confidence),
                        nmsThreshold=float(args.yolo_nms),
                    )
                    dets: list[tuple[int, int, int, int]] = []
                    if len(class_ids) > 0:
                        for cls_id, box in zip(class_ids.flatten(), boxes):
                            if int(cls_id) != 0:
                                continue
                            x, y, w, h = box
                            if 0.1 <= scale < 1.0:
                                inv = 1.0 / scale
                                dets.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))
                            else:
                                dets.append((int(x), int(y), int(w), int(h)))
                    person_boxes_cache = dets

            sun_mask = build_sun_mask(hsv, args.sun_value_thresh, args.sun_sat_max) if args.suppress_sunlight else None
            glare_mask = cv2.threshold(gray, int(args.bright_glare_thresh), 255, cv2.THRESH_BINARY)[1]
            glare_mask = cv2.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=1)

            if frame_idx < startup_frames:
                startup_boost = 1.0 - (frame_idx / float(startup_frames))
                adaptive_lr = min(0.05, float(args.learning_rate) + (0.02 * startup_boost))
            else:
                adaptive_lr = float(args.learning_rate)
            fg = subtractor.apply(blurred, learningRate=adaptive_lr)
            _, fg = cv2.threshold(fg, int(args.fg_threshold), 255, cv2.THRESH_BINARY)
            fg = apply_zone_morph(
                fg,
                far_y=far_y,
                kernel=morph_kernel,
                near_open=int(args.near_open_iter),
                near_close=int(args.near_close_iter),
                near_dilate=int(args.near_dilate_iter),
                far_open=int(args.far_open_iter),
                far_close=int(args.far_close_iter),
                far_dilate=int(args.far_dilate_iter),
            )

            if prev_gray is None:
                frame_delta = np.zeros_like(gray)
                delta_mask = np.zeros_like(gray)
                if args.use_running_bg and running_gray is None:
                    running_gray = gray.astype(np.float32)
            else:
                if args.use_running_bg:
                    if running_gray is None:
                        running_gray = prev_gray.astype(np.float32)
                    cv2.accumulateWeighted(gray, running_gray, float(args.running_bg_alpha))
                    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(running_gray))
                else:
                    frame_delta = cv2.absdiff(gray, prev_gray)
                _, delta_mask = cv2.threshold(frame_delta, int(args.delta_threshold), 255, cv2.THRESH_BINARY)
                delta_mask = apply_zone_morph(
                    delta_mask,
                    far_y=far_y,
                    kernel=morph_kernel,
                    near_open=int(args.near_open_iter),
                    near_close=int(args.near_close_iter),
                    near_dilate=int(args.near_dilate_iter),
                    far_open=int(args.far_open_iter),
                    far_close=int(args.far_close_iter),
                    far_dilate=int(args.far_dilate_iter),
                )

            if 0.1 <= scale < 1.0:
                fg = cv2.resize(fg, (width, height), interpolation=cv2.INTER_NEAREST)
                delta_mask = cv2.resize(delta_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                frame_delta = cv2.resize(frame_delta, (width, height), interpolation=cv2.INTER_NEAREST)
                if sun_mask is not None:
                    sun_mask = cv2.resize(sun_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            if roi_mask is not None:
                fg = cv2.bitwise_and(fg, fg, mask=roi_mask)
                delta_mask = cv2.bitwise_and(delta_mask, delta_mask, mask=roi_mask)
                frame_delta = cv2.bitwise_and(frame_delta, frame_delta, mask=roi_mask)
            if sun_mask is not None:
                inv_sun = cv2.bitwise_not(sun_mask)
                fg = cv2.bitwise_and(fg, inv_sun)
                delta_mask = cv2.bitwise_and(delta_mask, inv_sun)
            fg = cv2.bitwise_and(fg, cv2.bitwise_not(glare_mask))
            delta_mask = cv2.bitwise_and(delta_mask, cv2.bitwise_not(glare_mask))

            curr_mean = float(np.mean(gray))
            illum_event = (
                prev_mean_brightness is not None
                and abs(curr_mean - prev_mean_brightness) > float(args.illum_change_thresh)
            )
            if illum_event:
                delta_mask = np.zeros_like(delta_mask)
            prev_mean_brightness = curr_mean

            motion_mask = cv2.bitwise_or(fg, delta_mask)
            near_mask = motion_mask.copy()
            near_mask[:far_y, :] = 0
            far_mask = motion_mask.copy()
            far_mask[far_y:, :] = 0
            fusion_queue.append(far_mask)
            fused_far = far_mask.copy()
            if len(fusion_queue) > 1:
                for pm in list(fusion_queue)[:-1]:
                    fused_far = cv2.bitwise_or(fused_far, pm)
            fused_mask = cv2.bitwise_or(near_mask, fused_far)

            motion = False
            boxes: list[tuple[int, int, int, int, float]] = []
            max_area = 0.0

            if frame_idx >= warmup_frames:
                contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates: list[tuple[int, int, int, int, float, float, float, float]] = []

                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 5:
                        continue

                    x, y, w, h = cv2.boundingRect(c)
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    far_weight = float(np.clip(1.0 - (cy / max(1.0, float(height - 1))), 0.0, 1.0))

                    area_thresh = int(min_area_near - (min_area_near - min_area_far) * far_weight)
                    if area < area_thresh:
                        continue

                    min_w = max(2, int((width * args.min_width_ratio) - ((width * args.min_width_ratio * 0.65) * far_weight)))
                    min_h = max(4, int((height * args.min_height_ratio) - ((height * args.min_height_ratio * 0.70) * far_weight)))
                    if w < min_w or h < min_h:
                        continue

                    bbox_pixels = max(1, w * h)
                    fill_ratio = area / float(bbox_pixels)
                    fill_thresh = float(args.min_fill_ratio) - (float(args.min_fill_ratio) - float(args.far_min_fill_ratio)) * far_weight
                    if fill_ratio < fill_thresh:
                        continue

                    hull = cv2.convexHull(c)
                    hull_area = max(1.0, cv2.contourArea(hull))
                    solidity = area / hull_area
                    solidity_thresh = float(args.min_solidity) - (float(args.min_solidity) - float(args.far_min_solidity)) * far_weight
                    if solidity < solidity_thresh:
                        continue

                    shape_ratio = h / float(w + 1e-6)
                    if shape_ratio < 0.8 or shape_ratio > 6.0:
                        continue

                    local = c.copy()
                    local[:, 0, 0] -= x
                    local[:, 0, 1] -= y
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [local], -1, 255, thickness=cv2.FILLED)

                    delta_roi = fused_mask[y:y + h, x:x + w]
                    moving_inside = cv2.bitwise_and(delta_roi, contour_mask)
                    delta_pixels = int(np.count_nonzero(moving_inside))
                    contour_pixels = max(1, int(np.count_nonzero(contour_mask)))
                    delta_ratio = delta_pixels / contour_pixels

                    delta_thresh_px = int(min_delta_near - (min_delta_near - min_delta_far) * far_weight)
                    delta_area_ratio = float(args.min_delta_area_ratio) - (
                        (float(args.min_delta_area_ratio) - float(args.far_min_delta_area_ratio)) * far_weight
                    )
                    delta_thresh_area = int(frame_area * max(0.0, delta_area_ratio))
                    delta_thresh = max(delta_thresh_px, delta_thresh_area)
                    if delta_pixels < delta_thresh or delta_ratio < float(args.min_delta_ratio):
                        continue

                    candidates.append((x, y, w, h, area, cx, cy, far_weight))
                    max_area = max(max_area, area)

                # Static-ghost suppression track
                kept: list[tuple[int, int, int, int, float]] = []
                new_static_tracks: list[tuple[float, float, int]] = []
                used_static: set[int] = set()
                max_shift_sq = float(args.static_max_shift) ** 2

                for x, y, w, h, area, cx, cy, far_weight in candidates:
                    if args.people_only:
                        if far_weight < 0.55:
                            hit = any(overlap_ratio((x, y, w, h), pb) >= float(args.person_overlap_ratio) for pb in person_boxes_cache)
                            if not hit:
                                continue
                        else:
                            human_ratio = h / float(w + 1e-6)
                            if human_ratio < float(args.far_human_min_ratio) or human_ratio > float(args.far_human_max_ratio) or h < int(args.far_human_min_height):
                                continue

                    best_i = -1
                    best_d2 = float("inf")
                    for i, (sx, sy, _) in enumerate(static_tracks):
                        if i in used_static:
                            continue
                        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
                        if d2 <= max_shift_sq and d2 < best_d2:
                            best_d2 = d2
                            best_i = i

                    if best_i >= 0:
                        used_static.add(best_i)
                        _, _, prev_streak = static_tracks[best_i]
                        streak = prev_streak + 1
                    else:
                        streak = 0

                    # Keep track history even if we don't draw the box this frame.
                    new_static_tracks.append((cx, cy, streak))

                    if streak >= int(args.static_hold_frames) and far_weight <= float(args.far_static_disable_weight):
                        continue

                    if far_weight >= float(args.far_y_ratio) and streak < max(0, int(args.far_confirm_frames) - 1):
                        continue

                    kept.append((x, y, w, h, area))

                static_tracks = new_static_tracks

                # Persistence tracker to reduce flicker.
                updated_persist: list[tuple[int, int, int, int, int]] = []
                used_persist: set[int] = set()
                for x, y, w, h, area in kept:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    best_i = -1
                    best_d2 = float("inf")
                    for i, (tx, ty, tw, th, ttl) in enumerate(persist_tracks):
                        if i in used_persist:
                            continue
                        tcx = tx + tw / 2.0
                        tcy = ty + th / 2.0
                        d2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
                        gate = float(max(w, h, tw, th)) * 0.8
                        if d2 <= gate * gate and d2 < best_d2:
                            best_d2 = d2
                            best_i = i
                    if best_i >= 0:
                        used_persist.add(best_i)
                    updated_persist.append((x, y, w, h, int(args.track_hold_frames)))

                for i, (tx, ty, tw, th, ttl) in enumerate(persist_tracks):
                    if i in used_persist:
                        continue
                    if ttl - 1 > 0:
                        updated_persist.append((tx, ty, tw, th, ttl - 1))

                persist_tracks = updated_persist
                boxes = [(x, y, w, h, float(w * h)) for x, y, w, h, _ in persist_tracks]

                total_area = int(sum(a for *_, a in boxes))
                raw_motion = (len(boxes) > 0 and args.motion_by_box) or (total_area >= total_area_thresh)

                if raw_motion:
                    on_count += 1
                    off_count = 0
                else:
                    off_count += 1
                    on_count = 0

                if not motion_state and on_count >= max(1, int(args.debounce_on)):
                    motion_state = True
                elif motion_state and off_count >= max(1, int(args.debounce_off)):
                    motion_state = False

                motion = motion_state
                if not motion:
                    boxes = []
            else:
                static_tracks = []
                persist_tracks = []

            if motion:
                detected_frames += 1
                for x, y, w, h, _ in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            status = "MOTION" if motion else "NO MOTION"
            cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if motion else (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            motion_pixels = int(np.count_nonzero(fused_mask))
            wcsv.writerow([frame_idx, round(frame_idx / fps, 3), int(motion), len(boxes), round(max_area, 2), motion_pixels])

            writer.write(frame)
            if mask_writer is not None:
                if args.preview_mode == "fg":
                    writer_mask = fg
                elif args.preview_mode == "delta":
                    writer_mask = delta_mask
                else:
                    writer_mask = fused_mask
                mask_writer.write(writer_mask)

            if args.show:
                if args.preview_mode == "fg":
                    preview_mask = fg
                elif args.preview_mode == "delta":
                    preview_mask = delta_mask
                else:
                    preview_mask = fused_mask
                preview = np.hstack((frame, cv2.cvtColor(preview_mask, cv2.COLOR_GRAY2BGR)))
                cv2.imshow(window_name, preview)
                if args.debug_stages:
                    cv2.imshow("DIFF", frame_delta)
                    cv2.imshow("BLUR", gray)
                    cv2.imshow("THRESH", fused_mask)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key in (27, ord("q")):
                    interrupted = True
                    break

            prev_gray = gray
            frame_idx += 1

    cap.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()
    if args.show:
        cv2.destroyWindow(f"Motion Detection - {video_path.name}")

    ratio = (detected_frames / frame_idx * 100.0) if frame_idx else 0.0
    print(
        f"{video_path.name}: frames={frame_idx}, motion_frames={detected_frames} "
        f"({ratio:.1f}%), video={out_video.name}, csv={csv_path.name}"
    )
    return interrupted


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    videos = resolve_inputs(args.inputs)
    if not videos:
        print("No input videos matched. Use --inputs with files or glob patterns.")
        return 1

    print(f"Found {len(videos)} input video(s). Output: {out}")
    for video in videos:
        interrupted = process_video(video, out, args)
        if interrupted:
            print("Stopped by user.")
            break

    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
