import argparse
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.image_segmenter import ImageSegmenterResult

from gesture_detector import BambooJutsuDetector, DetectionStatus
from jutsu_effects import BambooJutsuEffect, FaceExpression, extract_face_expression, get_person_box

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
GREEN_SCREEN_PATH = ASSETS_DIR / "green_screen.png"
CUSTOM_BG_PATH = ASSETS_DIR / "custom_bg.png"
CHROMA_GREEN_BGR = (64, 177, 0)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "selfie_segmenter_landscape.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def ensure_models() -> dict[str, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in MODEL_URLS.items():
        target = MODELS_DIR / filename
        if not target.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, target)
        paths[filename] = target
    return paths


def create_green_screen_image(width: int, height: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cy, cx = height * 0.45, width * 0.5
    dist = np.sqrt(((x - cx) / (width * 0.65)) ** 2 + ((y - cy) / (height * 0.65)) ** 2)
    shade = np.clip(1.08 - 0.22 * (dist ** 2), 0.78, 1.08)[..., None]
    base = np.array(CHROMA_GREEN_BGR, dtype=np.float32)
    return np.clip(base * shade, 0, 255).astype(np.uint8)


def load_green_screen(width: int, height: int) -> np.ndarray:
    if GREEN_SCREEN_PATH.exists():
        image = cv2.imread(str(GREEN_SCREEN_PATH))
        if image is not None:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    image = create_green_screen_image(width, height)
    cv2.imwrite(str(GREEN_SCREEN_PATH), image)
    return image


def load_overscan_background(path: Path, width: int, height: int, overscan: float = 1.18) -> np.ndarray:
    target_w = int(round(width * overscan))
    target_h = int(round(height * overscan))
    if path.exists():
        image = cv2.imread(str(path))
        if image is not None:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return load_green_screen(target_w, target_h)


def warp_background_perspective(
    bg_overscan: np.ndarray,
    width: int,
    height: int,
    norm_x: float,
    norm_y: float,
) -> np.ndarray:
    bh, bw = bg_overscan.shape[:2]
    margin_x = max(1.0, (bw - width) * 0.5)
    margin_y = max(1.0, (bh - height) * 0.5)

    ox = max(-0.4, min(0.4, norm_x - 0.5))
    oy = max(-0.4, min(0.4, norm_y - 0.5))

    x0 = margin_x - ox * margin_x * 1.4
    y0 = margin_y - oy * margin_y * 1.4
    tilt_y = ox * margin_y * 0.65
    tilt_x = oy * margin_x * 0.45

    x1 = max(abs(tilt_x) + 1.0, min(bw - width - abs(tilt_x) - 1.0, x0))
    y1 = max(abs(tilt_y) + 1.0, min(bh - height - abs(tilt_y) - 1.0, y0))
    x2 = x1 + width
    y2 = y1 + height

    src = np.array(
        [
            [x1 - tilt_x, y1 - tilt_y],
            [x2 + tilt_x, y1 + tilt_y],
            [x2 - tilt_x, y2 - tilt_y],
            [x1 + tilt_x, y2 + tilt_y],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bg_overscan, matrix, (width, height), flags=cv2.INTER_LINEAR)


def get_head_normalized(pose_landmarks_list, face_landmarks_list=None) -> tuple[float, float] | None:
    if face_landmarks_list:
        nose = face_landmarks_list[0][1]
        return float(nose.x), float(nose.y)
    if pose_landmarks_list:
        lm = pose_landmarks_list[0]
        if (lm[0].visibility or 0.0) > 0.35:
            return float(lm[0].x), float(lm[0].y)
        if min(lm[11].visibility or 0.0, lm[12].visibility or 0.0) > 0.35:
            return float(0.5 * (lm[11].x + lm[12].x)), float(0.5 * (lm[11].y + lm[12].y))
    return None


def resize_to_width(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if target_width <= 0 or width <= target_width:
        return frame_bgr
    target_height = max(1, int(round(height * (target_width / width))))
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def apply_green_screen(
    frame_bgr: np.ndarray,
    segmentation_result: ImageSegmenterResult,
    background_bgr: np.ndarray,
    prev_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not segmentation_result.confidence_masks:
        return frame_bgr, prev_mask

    raw_mask = np.squeeze(segmentation_result.confidence_masks[0].numpy_view()).astype(np.float32)
    if prev_mask is not None and prev_mask.shape == raw_mask.shape:
        raw_mask = 0.75 * raw_mask + 0.25 * prev_mask

    fg_binary = (raw_mask > 0.45).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=4)
    if num_labels > 1:
        min_area = raw_mask.shape[0] * raw_mask.shape[1] * 0.03
        keep = np.zeros(num_labels, dtype=np.float32)
        for idx, area in enumerate(stats[1:, cv2.CC_STAT_AREA], start=1):
            if area >= min_area:
                keep[idx] = 1.0
        cleaned = raw_mask * keep[labels]
    else:
        cleaned = raw_mask

    smooth_mask = cv2.GaussianBlur(cleaned, (5, 5), 0)
    alpha = np.clip((smooth_mask - 0.35) / 0.45, 0.0, 1.0)
    if alpha.shape[:2] != frame_bgr.shape[:2]:
        alpha = cv2.resize(alpha, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = cv2.blendLinear(frame_bgr, background_bgr, alpha, 1.0 - alpha)
    return blended, raw_mask


def draw_landmarks_overlay(
    frame_bgr: np.ndarray,
    pose_result: vision.PoseLandmarkerResult,
    hand_result: vision.HandLandmarkerResult,
    pose_style: dict,
    hand_landmark_style: dict,
    hand_connection_style: dict,
    mirror: bool,
) -> None:
    height, width = frame_bgr.shape[:2]

    for pose_landmarks in pose_result.pose_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame_bgr,
            pose_landmarks,
            vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_style,
        )

    for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
        vision.drawing_utils.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_style,
            connection_drawing_spec=hand_connection_style,
        )

        for tip_idx in FINGERTIP_INDICES:
            tip = hand_landmarks[tip_idx]
            px = int(tip.x * width)
            py = int(tip.y * height)
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(frame_bgr, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

        if idx < len(hand_result.handedness) and hand_result.handedness[idx]:
            label = hand_result.handedness[idx][0].category_name
            if mirror:
                label = "Right" if label == "Left" else "Left"
            wrist = hand_landmarks[0]
            wx = int(wrist.x * width)
            wy = min(height - 10, int(wrist.y * height) + 24)
            cv2.putText(
                frame_bgr,
                label,
                (max(10, wx - 25), wy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def build_inference_image(frame_bgr: np.ndarray, target_width: int = 640) -> mp.Image:
    height, width = frame_bgr.shape[:2]
    if width > target_width:
        target_height = max(1, int(round(height * (target_width / width))))
        resized = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_bgr
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_hud(
    frame_bgr: np.ndarray,
    fps: float,
    hand_count: int,
    pose_detected: bool,
    bg_mode: str,
    show_landmarks: bool,
    status: DetectionStatus | None = None,
) -> None:
    lines = [
        f"FPS: {fps:.1f} | Pose: {'ON' if pose_detected else 'NONE'} | Hands: {hand_count}",
        f"Background: {bg_mode} | Landmarks: {'ON' if show_landmarks else 'OFF'}",
        "[G] Background  [L] Landmarks  [M] Mirror  [T] Test Jutsu  [Q/ESC] Quit",
    ]
    if status is not None:
        checks = (
            f"W:{int(status.wrists_close)} "
            f"R:{int(status.hands_raised)} "
            f"U:{int(status.hand_upright)} "
            f"IM:{int(status.index_middle_up)} "
            f"C:{int(status.ring_pinky_curled)}"
        )
        lines.append(f"Jutsu: {checks} | Score: {status.score:.2f} | Count: {status.trigger_count}")

    box_height = 24 + len(lines) * 22
    box_right = min(frame_bgr.shape[1] - 10, 520)
    roi = frame_bgr[10 : 10 + box_height, 10:box_right]
    roi[:] = ((roi.astype(np.uint16) * 115 + 20 * 140) >> 8).astype(np.uint8)
    for idx, text in enumerate(lines):
        cv2.putText(
            frame_bgr,
            text,
            (20, 32 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def draw_action_badge(frame_bgr: np.ndarray, active: bool, trigger_count: int) -> None:
    if not active:
        return
    height, width = frame_bgr.shape[:2]
    text = f"ACTION HAPPENED  #{trigger_count}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x2 = width - 20
    x1 = max(20, x2 - text_w - 36)
    if width >= 820:
        y1, y2 = 16, 16 + text_h + 24
    else:
        y2 = height - 16
        y1 = y2 - text_h - 24
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (16, 140, 45), -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (180, 255, 190), 2)
    cv2.putText(
        frame_bgr,
        text,
        (x1 + 18, y2 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


class CameraStream:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.lock = threading.Lock()
        self.ok = False
        self.frame: np.ndarray | None = None
        self.seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                with self.lock:
                    self.ok = False
                break
            with self.lock:
                self.ok = True
                self.frame = frame
                self.seq += 1

    def read(self, last_seq: int, timeout: float = 1.0) -> tuple[bool, np.ndarray | None, int]:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.lock:
                if self.frame is not None and self.seq != last_seq:
                    return True, self.frame.copy(), self.seq
                if not self.running:
                    return False, None, last_seq
            time.sleep(0.002)
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy(), self.seq

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def build_landmarkers(model_paths: dict[str, Path]):
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["pose_landmarker_lite.task"])),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["hand_landmarker.task"])),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    seg_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["selfie_segmenter_landscape.tflite"])),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.HandLandmarker.create_from_options(hand_options),
        vision.ImageSegmenter.create_from_options(seg_options),
    )


def build_face_landmarker(model_paths: dict[str, Path]) -> vision.FaceLandmarker:
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["face_landmarker.task"])),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(face_options)


def run_raw_camera(camera_index: int = 0) -> None:
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at index {camera_index}")

    window_name = "Raw Camera"
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_camera(
    camera_index: int = 0,
    raw: bool = False,
    target_width: int = 854,
    virtualcam: bool = False,
) -> None:
    if raw:
        run_raw_camera(camera_index)
        return

    vcam_module = None
    if virtualcam:
        import pyvirtualcam
        vcam_module = pyvirtualcam

    model_paths = ensure_models()
    pose_landmarker, hand_landmarker, segmenter = build_landmarkers(model_paths)
    face_landmarker = build_face_landmarker(model_paths)
    cap = open_camera(camera_index)
    if not cap.isOpened():
        pose_landmarker.close()
        hand_landmarker.close()
        segmenter.close()
        face_landmarker.close()
        raise RuntimeError(f"Unable to open camera at index {camera_index}")

    stream = CameraStream(cap)
    pose_style = dict(vision.drawing_styles.get_default_pose_landmarks_style())
    hand_landmark_style = dict(vision.drawing_styles.get_default_hand_landmarks_style())
    hand_connection_style = dict(vision.drawing_styles.get_default_hand_connections_style())

    bg_modes = ("custom_bg", "studio_green", "solid_green", "camera")
    bg_index = 0
    show_landmarks = True
    mirror = True
    last_timestamp_ms = 0
    prev_time = time.perf_counter()
    fps = 0.0
    custom_bg_hi: np.ndarray | None = None
    studio_bg: np.ndarray | None = None
    solid_bg: np.ndarray | None = None
    prev_mask: np.ndarray | None = None
    smooth_nx = 0.5
    smooth_ny = 0.5
    last_seq = 0
    detector = BambooJutsuDetector()
    effect = BambooJutsuEffect(toggle_cooldown=1.0)
    badge_until = 0.0
    last_expr = FaceExpression()
    vcam = None

    window_name = "Bamboo Jutsu"

    try:
        while True:
            ok, frame_bgr, last_seq = stream.read(last_seq)
            if not ok or frame_bgr is None:
                break

            frame_bgr = resize_to_width(frame_bgr, target_width)
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            height, width = frame_bgr.shape[:2]
            if studio_bg is None or studio_bg.shape[:2] != (height, width):
                custom_bg_hi = load_overscan_background(CUSTOM_BG_PATH, width, height)
                studio_bg = load_green_screen(width, height)
                solid_bg = np.full((height, width, 3), CHROMA_GREEN_BGR, dtype=np.uint8)
            if virtualcam and vcam is None:
                try:
                    vcam = vcam_module.Camera(
                        width=width,
                        height=height,
                        fps=30,
                        fmt=vcam_module.PixelFormat.BGR,
                        backend="obs",
                    )
                    print(f"Virtual camera active: {vcam.device} ({width}x{height})")
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to start OBS Virtual Camera. Make sure OBS Studio is installed "
                        "and no other app is locking OBS Virtual Camera."
                    ) from exc

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = fps * 0.85 + (1.0 / dt) * 0.15 if fps > 0 else 1.0 / dt

            timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
            last_timestamp_ms = timestamp_ms

            mp_image = build_inference_image(frame_bgr, target_width=480)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            status = detector.update(pose_result.pose_landmarks, hand_result.hand_landmarks)

            if status.triggered:
                badge_until = now + 1.5
                cx, cy, bw, bh = get_person_box(prev_mask, pose_result.pose_landmarks, width, height)
                effect.toggle(cx, cy, bw, bh, now)

            bg_mode = bg_modes[bg_index]
            face_landmarks = None
            if effect.is_bamboo_active(now):
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                face_landmarks = face_result.face_landmarks
                if face_result.face_blendshapes:
                    last_expr = extract_face_expression(face_result)

            head_norm = get_head_normalized(pose_result.pose_landmarks, face_landmarks)
            if head_norm is not None:
                smooth_nx = smooth_nx * 0.82 + head_norm[0] * 0.18
                smooth_ny = smooth_ny * 0.82 + head_norm[1] * 0.18

            if bg_mode == "camera":
                bg_frame = frame_bgr
            elif bg_mode == "custom_bg":
                bg_frame = warp_background_perspective(custom_bg_hi, width, height, smooth_nx, smooth_ny)
            elif bg_mode == "studio_green":
                bg_frame = studio_bg
            else:
                bg_frame = solid_bg

            if effect.is_bamboo_active(now):
                cx, _, _, bh = get_person_box(None, pose_result.pose_landmarks, width, height, face_landmarks)
                output, bamboo_on = effect.render(frame_bgr, bg_frame, cx, bh, last_expr, now)
            else:
                if bg_mode == "camera":
                    composited = frame_bgr.copy()
                    prev_mask = None
                else:
                    seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
                    composited, prev_mask = apply_green_screen(frame_bgr, seg_result, bg_frame, prev_mask)
                cx, _, _, bh = get_person_box(prev_mask, pose_result.pose_landmarks, width, height)
                output, bamboo_on = effect.render(composited, bg_frame, cx, bh, last_expr, now)

            if vcam is not None:
                vcam.send(np.ascontiguousarray(output))

            if show_landmarks and not bamboo_on:
                draw_landmarks_overlay(
                    output,
                    pose_result,
                    hand_result,
                    pose_style,
                    hand_landmark_style,
                    hand_connection_style,
                    mirror,
                )

            draw_hud(
                output,
                fps,
                len(hand_result.hand_landmarks),
                bool(pose_result.pose_landmarks),
                bg_mode,
                show_landmarks,
                status,
            )
            draw_action_badge(output, bamboo_on or now < badge_until, status.trigger_count)

            cv2.imshow(window_name, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("g"):
                bg_index = (bg_index + 1) % len(bg_modes)
            elif key == ord("l"):
                show_landmarks = not show_landmarks
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("t"):
                detector.trigger_count += 1
                badge_until = now + 1.5
                cx, cy, bw, bh = get_person_box(prev_mask, pose_result.pose_landmarks, width, height)
                effect.toggle(cx, cy, bw, bh, now)
    finally:
        if vcam is not None:
            vcam.close()
        pose_landmarker.close()
        hand_landmarker.close()
        segmenter.close()
        face_landmarker.close()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboo Jutsu real-time webcam substitution filter")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--raw", action="store_true", help="Show raw camera feed without processing")
    parser.add_argument("--width", type=int, default=854, help="Output frame width in pixels (default: 854)")
    parser.add_argument("--virtualcam", action="store_true", help="Stream clean output to OBS Virtual Camera")
    args = parser.parse_args()
    run_camera(args.camera, raw=args.raw, target_width=args.width, virtualcam=args.virtualcam)
