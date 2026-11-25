from dataclasses import dataclass
import math
from pathlib import Path
import random

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BAMBOO_PATH = ASSETS_DIR / "bamboo_log.png"
SMOKE_SHEET_PATH = ASSETS_DIR / "smoke_1.png"
STALK_CENTER_X_RATIO = 0.36


@dataclass
class SmokePuff:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    grow: float
    born: float
    life: float


@dataclass
class FaceExpression:
    blink_left: float = 0.0
    blink_right: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    jaw_open: float = 0.0
    smile: float = 0.0
    pucker: float = 0.0
    head_tilt: float = 0.0
    has_face: bool = False


def extract_face_expression(face_result) -> FaceExpression:
    if not face_result or not face_result.face_blendshapes:
        return FaceExpression()

    scores = {cat.category_name: cat.score for cat in face_result.face_blendshapes[0]}
    blink_left = scores.get("eyeBlinkLeft", 0.0)
    blink_right = scores.get("eyeBlinkRight", 0.0)
    gaze_x = (
        (scores.get("eyeLookInLeft", 0.0) - scores.get("eyeLookOutLeft", 0.0))
        + (scores.get("eyeLookOutRight", 0.0) - scores.get("eyeLookInRight", 0.0))
    ) * 0.5
    gaze_y = (
        (scores.get("eyeLookDownLeft", 0.0) + scores.get("eyeLookDownRight", 0.0))
        - (scores.get("eyeLookUpLeft", 0.0) + scores.get("eyeLookUpRight", 0.0))
    ) * 0.5

    raw_jaw = scores.get("jawOpen", 0.0)
    funnel = scores.get("mouthFunnel", 0.0)
    smile = 0.5 * (scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0))
    pucker = scores.get("mouthPucker", 0.0)

    head_tilt = 0.0
    lip_gap = 0.0
    if face_result.face_landmarks:
        lm = face_result.face_landmarks[0]
        if len(lm) > 263:
            left_eye = lm[33]
            right_eye = lm[263]
            head_tilt = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
            eye_dist = math.sqrt(
                (right_eye.x - left_eye.x) ** 2
                + (right_eye.y - left_eye.y) ** 2
                + (right_eye.z - left_eye.z) ** 2
            )
            if eye_dist > 1e-4:
                lip_gap = math.sqrt(
                    (lm[14].x - lm[13].x) ** 2
                    + (lm[14].y - lm[13].y) ** 2
                    + (lm[14].z - lm[13].z) ** 2
                ) / eye_dist

    jaw_boost = max(0.0, (raw_jaw - 0.015) * 3.0)
    lip_boost = max(0.0, (lip_gap - 0.025) * 4.2)
    jaw_open = min(1.0, max(jaw_boost, lip_boost) + funnel * 0.55)

    return FaceExpression(
        blink_left=blink_left,
        blink_right=blink_right,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        jaw_open=jaw_open,
        smile=smile,
        pucker=pucker,
        head_tilt=head_tilt,
        has_face=True,
    )


def load_bamboo_sprite() -> tuple[np.ndarray, np.ndarray]:
    if BAMBOO_PATH.exists():
        bgra = cv2.imread(str(BAMBOO_PATH), cv2.IMREAD_UNCHANGED)
        if bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4:
            return bgra[..., :3], bgra[..., 3]
    raise FileNotFoundError(f"Missing bamboo sprite at {BAMBOO_PATH}")


def load_smoke_frames() -> list[tuple[np.ndarray, np.ndarray]]:
    if not SMOKE_SHEET_PATH.exists():
        raise FileNotFoundError(f"Missing smoke sprite sheet at {SMOKE_SHEET_PATH}")
    sheet = cv2.imread(str(SMOKE_SHEET_PATH), cv2.IMREAD_UNCHANGED)
    if sheet is None or sheet.ndim != 3 or sheet.shape[2] != 4:
        raise ValueError(f"Invalid RGBA smoke sheet at {SMOKE_SHEET_PATH}")

    frame_size = sheet.shape[0]
    count = max(1, sheet.shape[1] // frame_size)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(count):
        tile = sheet[:, i * frame_size : (i + 1) * frame_size]
        frames.append((tile[..., :3], tile[..., 3]))
    return frames


def get_person_box(
    mask: np.ndarray | None,
    pose_landmarks_list,
    width: int,
    height: int,
    face_landmarks_list=None,
) -> tuple[int, int, int, int]:
    if pose_landmarks_list:
        lm = pose_landmarks_list[0]
        ls_vis = lm[11].visibility or 0.0
        rs_vis = lm[12].visibility or 0.0
        if ls_vis > 0.35 and rs_vis > 0.35:
            cx = int(0.5 * (lm[11].x + lm[12].x) * width)
            top_y = int(max(10, (lm[0].y - 0.14) * height))
            bot_y = height - 10
            cy = (top_y + bot_y) // 2
            sw = max(90, int(abs(lm[11].x - lm[12].x) * width * 1.8))
            return cx, cy, sw, max(160, bot_y - top_y)
        if (lm[0].visibility or 0.0) > 0.4:
            cx = int(lm[0].x * width)
            top_y = int(max(10, (lm[0].y - 0.14) * height))
            bot_y = height - 10
            return cx, (top_y + bot_y) // 2, int(width * 0.28), max(160, bot_y - top_y)

    if face_landmarks_list:
        nose = face_landmarks_list[0][1]
        cx = int(nose.x * width)
        top_y = int(max(10, (nose.y - 0.16) * height))
        bot_y = height - 10
        return cx, (top_y + bot_y) // 2, int(width * 0.28), max(160, bot_y - top_y)

    if mask is not None:
        ys, xs = np.where(mask > 0.45)
        if len(xs) > 40:
            mh, mw = mask.shape[:2]
            sx = width / mw
            sy = height / mh
            x1 = int(xs.min() * sx)
            x2 = int(xs.max() * sx)
            y1 = int(ys.min() * sy)
            y2 = int(ys.max() * sy)
            return (x1 + x2) // 2, (y1 + y2) // 2, max(90, x2 - x1), max(150, y2 - y1)

    return width // 2, int(height * 0.55), int(width * 0.3), int(height * 0.7)


def _blit_rgba_nearest(
    canvas: np.ndarray,
    sprite_bgr: np.ndarray,
    sprite_alpha: np.ndarray,
    x: int,
    y: int,
    target_w: int,
    target_h: int,
    opacity: float = 1.0,
) -> None:
    h, w = canvas.shape[:2]
    x2 = x + target_w
    y2 = y + target_h
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = target_w - max(0, x2 - w)
    sy2 = target_h - max(0, y2 - h)
    if sx1 >= sx2 or sy1 >= sy2:
        return

    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(w, x2), min(h, y2)

    bgr_up = cv2.resize(sprite_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    alpha_up = cv2.resize(sprite_alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    alpha_crop = (alpha_up[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0) * opacity
    if opacity >= 0.99:
        mask = alpha_crop > 0.5
        roi = canvas[dy1:dy2, dx1:dx2]
        roi[mask] = bgr_up[sy1:sy2, sx1:sx2][mask]
        return

    roi = canvas[dy1:dy2, dx1:dx2]
    blended = cv2.blendLinear(bgr_up[sy1:sy2, sx1:sx2], roi, alpha_crop, 1.0 - alpha_crop)
    roi[:] = blended


def _blit_pre_scaled_rgba(
    canvas: np.ndarray,
    bgr_img: np.ndarray,
    alpha_img: np.ndarray,
    x: int,
    y: int,
) -> None:
    h, w = canvas.shape[:2]
    th, tw = bgr_img.shape[:2]
    x2, y2 = x + tw, y + th
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = tw - max(0, x2 - w)
    sy2 = th - max(0, y2 - h)
    if sx1 >= sx2 or sy1 >= sy2:
        return

    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(w, x2), min(h, y2)
    mask = alpha_img[sy1:sy2, sx1:sx2] > 127
    roi = canvas[dy1:dy2, dx1:dx2]
    roi[mask] = bgr_img[sy1:sy2, sx1:sx2][mask]


def _draw_googly_eye(
    bgr: np.ndarray,
    alpha: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    blink: float,
    pupil_dx: float,
    pupil_dy: float,
) -> None:
    outline = (18, 36, 18)
    if blink > 0.45:
        cv2.ellipse(bgr, (cx, cy), (radius, max(3, radius // 5)), 0, 0, 180, outline, 3, cv2.LINE_4)
        cv2.ellipse(alpha, (cx, cy), (radius, max(3, radius // 5)), 0, 0, 180, 255, 3, cv2.LINE_4)
        return

    squash_y = max(4, int(round(radius * (1.0 - min(0.75, blink * 1.4)))))
    cv2.ellipse(bgr, (cx, cy), (radius + 2, squash_y + 2), 0, 0, 360, outline, -1, cv2.LINE_4)
    cv2.ellipse(bgr, (cx, cy), (radius, squash_y), 0, 0, 360, (248, 250, 248), -1, cv2.LINE_4)
    cv2.ellipse(alpha, (cx, cy), (radius + 2, squash_y + 2), 0, 0, 360, 255, -1, cv2.LINE_4)

    pupil_r = max(4, int(round(radius * 0.48)))
    max_off_x = max(1, radius - pupil_r - 2)
    max_off_y = max(1, squash_y - min(pupil_r, squash_y - 1) - 1)
    px = cx + int(round(max(-1.0, min(1.0, pupil_dx)) * max_off_x))
    py = cy + int(round(max(-1.0, min(1.0, pupil_dy)) * max_off_y))
    pr_y = min(pupil_r, max(2, squash_y - 2))

    cv2.ellipse(bgr, (px, py), (pupil_r, pr_y), 0, 0, 360, (16, 18, 22), -1, cv2.LINE_4)
    glint_r = max(1, pupil_r // 3)
    cv2.circle(bgr, (px - max(1, pupil_r // 3), py - max(1, pr_y // 3)), glint_r, (255, 255, 255), -1, cv2.LINE_4)


def _draw_synced_mouth(
    bgr: np.ndarray,
    alpha: np.ndarray,
    cx: int,
    cy: int,
    mouth_w: int,
    expr: FaceExpression,
) -> None:
    outline = (18, 36, 18)
    if expr.jaw_open < 0.06:
        curve = int(round(4 + expr.smile * 10))
        cv2.ellipse(bgr, (cx, cy - curve // 2), (mouth_w, max(3, curve)), 0, 15, 165, outline, 3, cv2.LINE_4)
        cv2.ellipse(alpha, (cx, cy - curve // 2), (mouth_w, max(3, curve)), 0, 15, 165, 255, 3, cv2.LINE_4)
        return

    open_h = max(4, int(round(3 + min(1.0, expr.jaw_open) * mouth_w * 0.95)))
    open_w = max(5, int(round(mouth_w * (0.65 if expr.pucker > 0.35 else 0.92 + expr.smile * 0.25))))
    cv2.ellipse(bgr, (cx, cy), (open_w + 2, open_h + 2), 0, 0, 360, outline, -1, cv2.LINE_4)
    cv2.ellipse(bgr, (cx, cy), (open_w, open_h), 0, 0, 360, (32, 24, 62), -1, cv2.LINE_4)
    cv2.ellipse(alpha, (cx, cy), (open_w + 2, open_h + 2), 0, 0, 360, 255, -1, cv2.LINE_4)

    tongue_h = max(2, open_h // 2)
    tongue_w = max(3, int(open_w * 0.65))
    cv2.ellipse(bgr, (cx, cy + open_h // 2 - 1), (tongue_w, tongue_h), 0, 180, 360, (98, 92, 214), -1, cv2.LINE_4)


class BambooJutsuEffect:
    def __init__(self, toggle_cooldown: float = 1.0) -> None:
        self.bamboo_bgr, self.bamboo_alpha = load_bamboo_sprite()
        self.smoke_frames = load_smoke_frames()
        self.toggle_cooldown = toggle_cooldown
        self.puffs: list[SmokePuff] = []
        self.mode_active = False
        self.swap_time = 0.0
        self.last_toggle_time = -10.0
        self.bamboo_x = 427.0
        self.bamboo_y = 120.0
        self.bamboo_vx = 0.0
        self.bamboo_vy = 0.0
        self.jiggle_x = 0.0
        self.jiggle_y = 0.0
        self.jiggle_vx = 0.0
        self.jiggle_vy = 0.0
        self.last_physics_time = 0.0
        self.bounced = False
        self.last_target_h = 220
        self.baseline_tilt: float | None = None
        self.last_tilt = 0.0

    def spawn_smoke(self, cx: int, cy: int, box_w: int, box_h: int, now: float, count: int = 24) -> None:
        spread_x = max(50, int(box_w * 0.48))
        spread_y = max(60, int(box_h * 0.45))
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(25.0, 145.0)
            self.puffs.append(
                SmokePuff(
                    x=cx + random.uniform(-spread_x, spread_x),
                    y=cy + random.uniform(-spread_y, spread_y),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed - random.uniform(15.0, 55.0),
                    size=random.uniform(95.0, 175.0),
                    grow=random.uniform(70.0, 150.0),
                    born=now + random.uniform(0.0, 0.12),
                    life=random.uniform(0.55, 0.95),
                )
            )

    def toggle(self, cx: int, cy: int, box_w: int, box_h: int, now: float) -> bool:
        if now - self.last_toggle_time < self.toggle_cooldown:
            return self.mode_active
        self.last_toggle_time = now

        if not self.mode_active:
            self.mode_active = True
            self.swap_time = now + 0.08
            self.bamboo_x = float(cx)
            self.bamboo_y = float(max(10, cy - box_h // 2))
            self.bamboo_vx = 0.0
            self.bamboo_vy = 0.0
            self.jiggle_x = 0.0
            self.jiggle_y = 0.0
            self.jiggle_vx = 0.0
            self.jiggle_vy = 0.0
            self.last_physics_time = now
            self.bounced = False
            self.baseline_tilt = None
            self.last_tilt = 0.0
            self.spawn_smoke(cx, cy, box_w, box_h, now, count=28)
        else:
            self.mode_active = False
            puff_x = int(round(self.bamboo_x))
            puff_y = int(round(self.bamboo_y + self.last_target_h * 0.5))
            self.spawn_smoke(puff_x, puff_y, max(140, box_w), max(180, self.last_target_h), now, count=24)

        return self.mode_active

    def is_bamboo_active(self, now: float) -> bool:
        return self.mode_active and now >= self.swap_time

    def _draw_bamboo(
        self,
        canvas: np.ndarray,
        target_cx: int,
        target_box_h: int,
        expr: FaceExpression,
        now: float,
    ) -> None:
        h, w = canvas.shape[:2]
        target_h = max(120, min(int(h * 0.58), int(target_box_h * 0.62)))
        self.last_target_h = target_h
        aspect = self.bamboo_bgr.shape[1] / self.bamboo_bgr.shape[0]
        target_w = max(56, int(round(target_h * aspect)))

        dt = max(0.001, min(0.05, now - self.last_physics_time))
        self.last_physics_time = now

        prev_x = self.bamboo_x
        prev_vx = self.bamboo_vx
        clamped_target_x = max(target_w * 0.35, min(w - target_w * 0.65, float(target_cx)))
        self.bamboo_x += (clamped_target_x - self.bamboo_x) * min(1.0, dt * 18.0)
        self.bamboo_vx = (self.bamboo_x - prev_x) / dt
        accel_x = (self.bamboo_vx - prev_vx) / dt

        ground_y = float(h - target_h - 4)
        prev_vy = self.bamboo_vy
        if self.bamboo_y < ground_y or abs(self.bamboo_vy) > 5.0:
            self.bamboo_vy += 2600.0 * dt
            self.bamboo_y += self.bamboo_vy * dt
            if self.bamboo_y >= ground_y:
                self.bamboo_y = ground_y
                if abs(self.bamboo_vy) > 140.0:
                    if not self.bounced:
                        self.bounced = True
                        self.spawn_smoke(int(round(self.bamboo_x)), h - 18, target_w, 30, now, count=6)
                    self.bamboo_vy = -self.bamboo_vy * 0.34
                else:
                    self.bamboo_vy = 0.0
        else:
            self.bamboo_y = ground_y
        accel_y = (self.bamboo_vy - prev_vy) / dt

        speed_x = abs(self.bamboo_vx)
        crawl_hop = 0.0
        crawl_tilt = 0.0
        if speed_x > 25.0 and self.bamboo_y >= ground_y - 2.0:
            crawl_hop = abs(math.sin(now * 26.0)) * min(14.0, speed_x * 0.032)
            crawl_tilt = max(-14.0, min(14.0, -self.bamboo_vx * 0.032)) + math.sin(now * 26.0) * min(6.0, speed_x * 0.015)

        relative_tilt = 0.0
        if expr.has_face:
            if self.baseline_tilt is None:
                self.baseline_tilt = expr.head_tilt
            elif abs(expr.head_tilt - self.baseline_tilt) < 12.0:
                self.baseline_tilt += (expr.head_tilt - self.baseline_tilt) * min(1.0, dt * 0.8)
            relative_tilt = expr.head_tilt - self.baseline_tilt
            if abs(relative_tilt) < 3.0:
                relative_tilt = 0.0

        total_tilt = max(-22.0, min(22.0, crawl_tilt - relative_tilt * 0.45))
        self.last_tilt = total_tilt

        self.jiggle_vx += (-accel_x * 0.00055 - 120.0 * self.jiggle_x - 10.0 * self.jiggle_vx) * dt
        self.jiggle_vy += (-accel_y * 0.00055 - 120.0 * self.jiggle_y - 10.0 * self.jiggle_vy) * dt
        self.jiggle_x = max(-0.85, min(0.85, self.jiggle_x + self.jiggle_vx * dt))
        self.jiggle_y = max(-0.85, min(0.85, self.jiggle_y + self.jiggle_vy * dt))

        bamboo_up = cv2.resize(self.bamboo_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        alpha_up = cv2.resize(self.bamboo_alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        eye_r = max(9, int(round(target_w * 0.11)))
        eye_y = int(round(target_h * 0.54))
        left_eye_x = int(round(target_w * 0.25))
        right_eye_x = int(round(target_w * 0.47))
        pupil_dx = expr.gaze_x * 1.6 + self.jiggle_x
        pupil_dy = expr.gaze_y * 1.6 + self.jiggle_y

        _draw_googly_eye(bamboo_up, alpha_up, left_eye_x, eye_y, eye_r, expr.blink_left, pupil_dx, pupil_dy)
        _draw_googly_eye(bamboo_up, alpha_up, right_eye_x, eye_y, eye_r, expr.blink_right, pupil_dx, pupil_dy)

        mouth_x = int(round(target_w * STALK_CENTER_X_RATIO))
        mouth_y = int(round(target_h * 0.67))
        mouth_w = max(10, int(round(target_w * 0.14)))
        _draw_synced_mouth(bamboo_up, alpha_up, mouth_x, mouth_y, mouth_w, expr)

        pad = int(round(max(target_w, target_h) * 0.22))
        pw = target_w + pad * 2
        ph = target_h + pad * 2
        padded_bgr = np.zeros((ph, pw, 3), dtype=np.uint8)
        padded_alpha = np.zeros((ph, pw), dtype=np.uint8)
        padded_bgr[pad : pad + target_h, pad : pad + target_w] = bamboo_up
        padded_alpha[pad : pad + target_h, pad : pad + target_w] = alpha_up

        if abs(total_tilt) > 0.5:
            pivot = (pad + target_w * STALK_CENTER_X_RATIO, pad + target_h * 0.5)
            mat = cv2.getRotationMatrix2D(pivot, total_tilt, 1.0)
            padded_bgr = cv2.warpAffine(padded_bgr, mat, (pw, ph), flags=cv2.INTER_NEAREST)
            padded_alpha = cv2.warpAffine(padded_alpha, mat, (pw, ph), flags=cv2.INTER_NEAREST)

        x1 = int(round(self.bamboo_x - target_w * STALK_CENTER_X_RATIO)) - pad
        y1 = int(round(self.bamboo_y - crawl_hop)) - pad
        _blit_pre_scaled_rgba(canvas, padded_bgr, padded_alpha, x1, y1)

    def _draw_smoke(self, canvas: np.ndarray, now: float) -> None:
        alive: list[SmokePuff] = []
        num_frames = len(self.smoke_frames)
        for p in self.puffs:
            age = now - p.born
            if age < 0:
                alive.append(p)
                continue
            if age >= p.life:
                continue
            alive.append(p)
            t = age / p.life
            frame_idx = min(num_frames - 1, int(t * num_frames))
            bgr, alpha = self.smoke_frames[frame_idx]
            sz = max(24, int(round(p.size + p.grow * age)))
            px = int(round(p.x + p.vx * age - sz * 0.5))
            py = int(round(p.y + p.vy * age - sz * 0.5))
            opacity = max(0.0, min(1.0, (1.0 - t) * 1.2))
            _blit_rgba_nearest(canvas, bgr, alpha, px, py, sz, sz, opacity=opacity)
        self.puffs = alive

    def render(
        self,
        composited_bgr: np.ndarray,
        background_bgr: np.ndarray,
        target_cx: int,
        target_box_h: int,
        expr: FaceExpression,
        now: float,
    ) -> tuple[np.ndarray, bool]:
        bamboo_on = self.is_bamboo_active(now)
        if bamboo_on:
            output = background_bgr.copy()
            self._draw_bamboo(output, target_cx, target_box_h, expr, now)
        else:
            output = composited_bgr

        if self.puffs:
            self._draw_smoke(output, now)

        return output, bamboo_on
