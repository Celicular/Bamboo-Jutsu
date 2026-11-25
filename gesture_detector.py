from collections import deque
from dataclasses import dataclass
import math


@dataclass
class DetectionStatus:
    active: bool
    triggered: bool
    trigger_count: int
    score: float
    wrists_close: bool
    hands_raised: bool
    hand_upright: bool
    index_middle_up: bool
    ring_pinky_curled: bool


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _segment_cos(a, b, c) -> float:
    v1x, v1y = b.x - a.x, b.y - a.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return (v1x * v2x + v1y * v2y) / (n1 * n2)


def _evaluate_pose(pose_landmarks) -> tuple[bool, bool, bool]:
    if not pose_landmarks or len(pose_landmarks) < 17:
        return False, False, False

    ls = pose_landmarks[11]
    rs = pose_landmarks[12]
    le = pose_landmarks[13]
    re = pose_landmarks[14]
    lw = pose_landmarks[15]
    rw = pose_landmarks[16]

    if min(ls.visibility or 1.0, rs.visibility or 1.0) < 0.3:
        return False, False, False
    if min(lw.visibility or 1.0, rw.visibility or 1.0) < 0.3:
        return False, False, False

    scale = max(_dist(ls, rs), 0.12)
    wrist_dist = _dist(lw, rw)
    wrists_close = wrist_dist <= 0.75 * scale

    shoulder_y = 0.5 * (ls.y + rs.y)
    wrist_y = 0.5 * (lw.y + rw.y)
    near_chest = (shoulder_y - 0.85 * scale) <= wrist_y <= (shoulder_y + 0.55 * scale)

    elbows_below = True
    if (le.visibility or 0.0) > 0.4 and (re.visibility or 0.0) > 0.4:
        elbows_below = lw.y < le.y + 0.1 * scale and rw.y < re.y + 0.1 * scale

    hands_raised = near_chest and elbows_below
    return True, wrists_close, hands_raised


def _is_finger_extended_up(lm, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int, palm_size: float) -> bool:
    wrist = lm[0]
    mcp = lm[mcp_idx]
    pip = lm[pip_idx]
    tip = lm[tip_idx]
    straight = _segment_cos(mcp, pip, tip) > 0.65
    pointing_up = tip.y < pip.y - 0.1 * palm_size and pip.y < mcp.y
    extended = _dist(wrist, tip) > _dist(wrist, pip) + 0.15 * palm_size
    return straight and pointing_up and extended


def _is_finger_curled(lm, pip_idx: int, tip_idx: int, middle_pip_y: float, ref_len: float, palm_size: float) -> bool:
    wrist = lm[0]
    pip = lm[pip_idx]
    tip = lm[tip_idx]
    shorter_than_index_middle = _dist(wrist, tip) < 0.82 * ref_len
    folded = _dist(wrist, tip) < _dist(wrist, pip) + 0.22 * palm_size or tip.y > middle_pip_y - 0.05 * palm_size
    return shorter_than_index_middle and folded


def _evaluate_hand(hand_landmarks) -> tuple[bool, bool, bool]:
    lm = hand_landmarks
    wrist = lm[0]
    palm_size = max(_dist(wrist, lm[9]), _dist(wrist, lm[5]), 0.05)

    hand_upright = lm[9].y < wrist.y - 0.35 * palm_size and lm[5].y < wrist.y - 0.3 * palm_size
    index_up = _is_finger_extended_up(lm, 5, 6, 7, 8, palm_size)
    middle_up = _is_finger_extended_up(lm, 9, 10, 11, 12, palm_size)
    tips_close = _dist(lm[8], lm[12]) <= 0.75 * palm_size
    index_middle_up = index_up and middle_up and tips_close

    ref_len = max(_dist(wrist, lm[8]), _dist(wrist, lm[12]), palm_size)
    middle_pip_y = lm[10].y
    ring_curled = _is_finger_curled(lm, 14, 16, middle_pip_y, ref_len, palm_size)
    pinky_curled = _is_finger_curled(lm, 18, 20, middle_pip_y, ref_len, palm_size)
    ring_pinky_curled = ring_curled and pinky_curled

    return hand_upright, index_middle_up, ring_pinky_curled


class BambooJutsuDetector:
    def __init__(self, window_size: int = 6, trigger_frames: int = 4, release_frames: int = 2) -> None:
        self.history: deque[bool] = deque(maxlen=window_size)
        self.trigger_frames = trigger_frames
        self.release_frames = release_frames
        self.active = False
        self.trigger_count = 0

    def update(self, pose_landmarks_list, hand_landmarks_list) -> DetectionStatus:
        pose_valid, wrists_close, hands_raised = (
            _evaluate_pose(pose_landmarks_list[0]) if pose_landmarks_list else (False, False, False)
        )

        best_hand = (False, False, False)
        best_hand_score = -1
        for hand_landmarks in hand_landmarks_list:
            upright, idx_mid, curled = _evaluate_hand(hand_landmarks)
            hand_score = int(upright) + int(idx_mid) + int(curled)
            if hand_score > best_hand_score:
                best_hand_score = hand_score
                best_hand = (upright, idx_mid, curled)

        hand_upright, index_middle_up, ring_pinky_curled = best_hand
        checks = (wrists_close, hands_raised, hand_upright, index_middle_up, ring_pinky_curled)
        score = sum(1 for c in checks if c) / len(checks)

        hand_ok = hand_upright and index_middle_up and ring_pinky_curled
        pose_ok = (wrists_close and hands_raised) if pose_valid else True
        matched = hand_ok and pose_ok

        self.history.append(matched)
        positives = sum(1 for item in self.history if item)

        triggered = False
        if not self.active and positives >= self.trigger_frames:
            self.active = True
            self.trigger_count += 1
            triggered = True
        elif self.active and positives < self.release_frames:
            self.active = False

        return DetectionStatus(
            active=self.active,
            triggered=triggered,
            trigger_count=self.trigger_count,
            score=score,
            wrists_close=wrists_close,
            hands_raised=hands_raised,
            hand_upright=hand_upright,
            index_middle_up=index_middle_up,
            ring_pinky_curled=ring_pinky_curled,
        )
