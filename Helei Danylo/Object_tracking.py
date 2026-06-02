"""
Robust Object Tracker — Mode Script
====================================
Mode 1 (--mode 1, default):
  Normal tracking. Upper-left corner recorded every 4 frames → auto_corners[].
  Trajectory plot saved to plots/mode1/ on exit.

Mode 2 (--mode 2):
  Every 4 frames the video pauses so the user can draw a correction window.
  auto_corners[] ← tracker upper-left (only when user also draws).
  user_corners[] ← user upper-left.
  Tracking resumes from user position (original w/h kept).
  Trajectory plot + absolute error plot saved to plots/mode2/ on exit.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

BBox = Tuple[int, int, int, int]


class TrackState(Enum):
    TRACKING   = auto()
    OCCLUDED   = auto()
    REDETECTED = auto()
    LOST       = auto()


@dataclass
class TrackerConfig:
    tracker_type: str         = "CSRT"
    max_occluded_frames: int  = 60
    huber_delta: float        = 15.0
    kalman_noise_cov: float   = 5e-3
    kalman_meas_cov: float    = 5e-2
    velocity_decay: float     = 0.75
    hist_threshold: float     = 0.52
    iou_threshold: float      = 0.15
    fg_ratio_threshold: float = 0.20
    mog2_history: int         = 200
    mog2_var_threshold: float = 50.0
    mog2_detect_shadows: bool = False
    canny_low: int            = 40
    canny_high: int           = 120
    edge_dilate: int          = 3
    redetect_interval: int    = 3
    ransac_reproj_thresh: float = 4.0
    min_inliers: int          = 6
    template_scale: float     = 3.0
    median_window: int        = 5
    appearance_alpha: float   = 0.04


# ─── Foreground extractor ────────────────────────────────────────────────────

class ForestForegroundExtractor:
    def __init__(self, cfg: TrackerConfig):
        self.cfg  = cfg
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history       = cfg.mog2_history,
            varThreshold  = cfg.mog2_var_threshold,
            detectShadows = cfg.mog2_detect_shadows,
        )
        self._k3     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k5     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k_edge = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.edge_dilate, cfg.edge_dilate))

    def update_background(self, frame: np.ndarray, learn_rate: float = -1) -> None:
        self._mog.apply(frame, learningRate=learn_rate)

    def get_mask(self, frame: np.ndarray, bbox: BBox,
                 learn_rate: float = 0.0) -> np.ndarray:
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8)

        full_motion = self._mog.apply(frame, learningRate=learn_rate)
        _, full_motion = cv2.threshold(full_motion, 200, 255, cv2.THRESH_BINARY)
        full_motion = cv2.morphologyEx(full_motion, cv2.MORPH_CLOSE, self._k5)
        full_motion = cv2.morphologyEx(full_motion, cv2.MORPH_OPEN,  self._k3)

        pad = 10
        ex1, ey1 = max(0, x1 - pad), max(0, y1 - pad)
        ex2, ey2 = min(W, x2 + pad), min(H, y2 + pad)
        roi_gray  = cv2.cvtColor(frame[ey1:ey2, ex1:ex2], cv2.COLOR_BGR2GRAY)
        edges     = cv2.Canny(roi_gray, self.cfg.canny_low, self.cfg.canny_high)
        edges     = cv2.dilate(edges, self._k_edge)

        filled     = edges.copy()
        fh, fw     = filled.shape
        flood_mask = np.zeros((fh + 2, fw + 2), dtype=np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        filled = cv2.bitwise_not(filled)
        filled = cv2.bitwise_or(filled, edges)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, self._k5)

        full_edge           = np.zeros((H, W), dtype=np.uint8)
        full_edge[ey1:ey2, ex1:ex2] = filled

        combined     = cv2.bitwise_and(full_motion, full_edge)
        roi_combined = combined[y1:y2, x1:x2]
        if np.count_nonzero(roi_combined) < 20:
            roi_combined = full_motion[y1:y2, x1:x2]
        return roi_combined

    def warmup(self, cap: cv2.VideoCapture, n_frames: int = 30) -> None:
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print(f"[MOG2] Warming up on {n_frames} frames...", end=" ", flush=True)
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.update_background(frame, learn_rate=0.1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        print("done.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def calc_hist_masked(frame: np.ndarray, bbox: BBox,
                     mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi  = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    m = None
    if mask is not None and mask.size > 0:
        m = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
        if np.count_nonzero(m) < 10:
            return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], m, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def hist_dist(h1, h2) -> float:
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def fg_ratio(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / mask.size


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_bbox(bbox: BBox, W: int, H: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)


def center_bbox_on_mask(bbox: BBox, mask: np.ndarray, W: int, H: int,
                        frame: Optional[np.ndarray] = None,
                        ref_hist: Optional[np.ndarray] = None,
                        kalman_center: Optional[Tuple[int, int]] = None) -> BBox:
    x, y, w, h = bbox
    if mask is None or np.count_nonzero(mask) < 20:
        return bbox

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    blobs = [(i, stats[i, cv2.CC_STAT_AREA], centroids[i])
             for i in range(1, num_labels)
             if stats[i, cv2.CC_STAT_AREA] >= 15]
    if not blobs:
        return bbox

    # Single blob — original shift-limit logic, no scoring needed.
    if len(blobs) == 1:
        _, _, (cx_l, cy_l) = blobs[0]
        cx, cy = int(cx_l) + x, int(cy_l) + y
        shift = ((cx - (x + w // 2)) ** 2 + (cy - (y + h // 2)) ** 2) ** 0.5
        if shift > min(w, h) * 0.50:
            return bbox
        return clamp_bbox((cx - w // 2, cy - h // 2, w, h), W, H)

    # Multiple blobs — score each by proximity to Kalman + histogram match.
    diag = max(1.0, (w ** 2 + h ** 2) ** 0.5)
    kx = kalman_center[0] if kalman_center is not None else x + w // 2
    ky = kalman_center[1] if kalman_center is not None else y + h // 2

    best_score, best_cx, best_cy = -1.0, x + w // 2, y + h // 2
    for i, area, (cx_l, cy_l) in blobs:
        cx, cy = int(cx_l) + x, int(cy_l) + y

        # How close is this blob's centre to the Kalman-predicted position?
        prox = max(0.0, 1.0 - ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5 / diag)

        # How similar is this blob's colour to the tracked object's reference?
        h_score = 0.5  # neutral when no reference available
        if frame is not None and ref_hist is not None:
            blob_mask = (labels == i).astype(np.uint8) * 255
            h_blob = calc_hist_masked(frame, bbox, blob_mask)
            if h_blob is not None:
                h_score = max(0.0, 1.0 - hist_dist(ref_hist, h_blob))

        score = 0.5 * prox + 0.5 * h_score
        if score > best_score:
            best_score, best_cx, best_cy = score, cx, cy

    return clamp_bbox((best_cx - w // 2, best_cy - h // 2, w, h), W, H)


def refine_bbox_to_contour(frame: np.ndarray, bbox: BBox,
                            W: int, H: int, min_fill: float = 0.25) -> BBox:
    x, y, w, h = bbox
    H_f, W_f = frame.shape[:2]
    pad = max(4, int(min(w, h) * 0.15))
    x1 = max(0, x - pad);   y1 = max(0, y - pad)
    x2 = min(W_f, x+w+pad); y2 = min(H_f, y+h+pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return bbox
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw   = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 3)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k_open)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < w * h * min_fill:
        return bbox
    rx, ry, rw, rh = cv2.boundingRect(largest)
    new_cx = x1 + rx + rw // 2
    new_cy = y1 + ry + rh // 2
    return clamp_bbox((new_cx - w // 2, new_cy - h // 2, w, h), W_f, H_f)


# ─── Huber Kalman ────────────────────────────────────────────────────────────

class HuberKalmanTracker:
    def __init__(self, bbox: BBox, cfg: TrackerConfig):
        self.cfg = cfg
        self.kf  = cv2.KalmanFilter(6, 4)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,dt, 0],
            [0,1,0,0, 0,dt],
            [0,0,1,0, 0, 0],
            [0,0,0,1, 0, 0],
            [0,0,0,0, 1, 0],
            [0,0,0,0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((4, 6), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * cfg.kalman_noise_cov
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * cfg.kalman_meas_cov
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)
        cx, cy, w, h = self._c(bbox)
        self.kf.statePost = np.array([cx, cy, w, h, 0., 0.],
                                      dtype=np.float32).reshape(6, 1)

    @staticmethod
    def _c(b): return b[0]+b[2]/2, b[1]+b[3]/2, float(b[2]), float(b[3])

    @staticmethod
    def _b(s) -> BBox:
        cx, cy = float(s[0]), float(s[1])
        w,  h  = max(10., float(s[2])), max(10., float(s[3]))
        return (int(cx - w/2), int(cy - h/2), int(w), int(h))

    def predict(self) -> BBox:
        return self._b(self.kf.predict().flatten())

    def correct(self, bbox: BBox) -> BBox:
        cx, cy, w, h = self._c(bbox)
        z   = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        H   = self.kf.measurementMatrix
        xp  = self.kf.statePre
        inn = z - H @ xp
        wts = np.minimum(1.0, self.cfg.huber_delta / (np.abs(inn) + 1e-6))
        R_r = self.kf.measurementNoiseCov / (np.diag(wts.flatten()) + 1e-8)
        P   = self.kf.errorCovPre
        S   = H @ P @ H.T + R_r
        K   = P @ H.T @ np.linalg.inv(S)
        xp2 = xp + K @ inn
        self.kf.statePost    = xp2.astype(np.float32)
        IKH = np.eye(6) - K @ H
        self.kf.errorCovPost = (IKH @ P @ IKH.T + K @ R_r @ K.T).astype(np.float32)
        return self._b(xp2.flatten())

    def position_uncertainty(self) -> float:
        P = self.kf.errorCovPost
        return float(np.sqrt((P[0, 0] + P[1, 1]) / 2.))

    def decay_velocity(self):
        self.kf.statePost[4] *= self.cfg.velocity_decay
        self.kf.statePost[5] *= self.cfg.velocity_decay


# ─── Median filter ───────────────────────────────────────────────────────────

class MedianTrajectoryFilter:
    def __init__(self, w=5):
        self._buf: deque = deque(maxlen=w)

    def update(self, bbox: BBox) -> BBox:
        self._buf.append(bbox)
        m = np.median(np.array(self._buf, dtype=np.float32), axis=0)
        return (int(m[0]), int(m[1]), int(m[2]), int(m[3]))

    def reset(self):
        self._buf.clear()


# ─── RANSAC re-detector ──────────────────────────────────────────────────────

class RANSACRedetector:
    def __init__(self, cfg: TrackerConfig):
        self.cfg       = cfg
        self._orb      = cv2.ORB_create(nfeatures=500)
        self._bf       = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._tmpl_kp  = None
        self._tmpl_des = None
        self._tmpl_size: Optional[Tuple[int, int]] = None
        self._tmpl_crop: Optional[np.ndarray]      = None

    def set_template(self, frame: np.ndarray, bbox: BBox,
                     mask: Optional[np.ndarray] = None) -> None:
        x, y, w, h = bbox
        H, W = frame.shape[:2]
        roi  = frame[max(0, y):min(H, y+h), max(0, x):min(W, x+w)]
        if roi.size == 0:
            return
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        rh, rw  = gray.shape[:2]
        orb_mask = None
        if mask is not None and fg_ratio(mask) >= 0.15:
            orb_mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
        kp, des = self._orb.detectAndCompute(gray, orb_mask)
        if des is None or len(kp) < 8:
            kp, des = self._orb.detectAndCompute(gray, None)
        if des is not None and len(kp) >= 4:
            self._tmpl_kp   = kp
            self._tmpl_des  = des
            self._tmpl_size = (w, h)
            self._tmpl_crop = roi.copy()

    def detect(self, frame: np.ndarray, search_bbox: BBox) -> Optional[BBox]:
        H_f, W_f = frame.shape[:2]
        sx, sy, sw, sh = search_bbox
        sc  = self.cfg.template_scale
        rx1 = max(0,   int(sx - sw*(sc-1)/2))
        ry1 = max(0,   int(sy - sh*(sc-1)/2))
        rx2 = min(W_f, int(sx + sw*(sc+1)/2))
        ry2 = min(H_f, int(sy + sh*(sc+1)/2))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        result = self._ransac_detect(frame, roi, rx1, ry1, W_f, H_f)
        if result is not None:
            return result
        return self._tm_detect(roi, rx1, ry1, W_f, H_f)

    def _ransac_detect(self, frame, roi, rx1, ry1, W_f, H_f) -> Optional[BBox]:
        if self._tmpl_kp is None or self._tmpl_des is None:
            return None
        kp2, des2 = self._orb.detectAndCompute(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None)
        if des2 is None or len(kp2) < 4:
            return None
        matches = self._bf.knnMatch(self._tmpl_des, des2, k=2)
        good    = [m for m, n in matches
                   if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < self.cfg.min_inliers:
            return None
        src = np.float32([self._tmpl_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt          for m in good]).reshape(-1, 1, 2)
        M, msk = cv2.findHomography(src, dst, cv2.RANSAC,
                                    self.cfg.ransac_reproj_thresh)
        if M is None or int(msk.sum()) < self.cfg.min_inliers:
            return None
        tw, th    = self._tmpl_size
        corners   = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, M)
        px = projected[:, 0, 0] + rx1
        py = projected[:, 0, 1] + ry1
        return clamp_bbox(
            (int(np.min(px)), int(np.min(py)),
             int(np.max(px) - np.min(px)), int(np.max(py) - np.min(py))),
            W_f, H_f)

    def _tm_detect(self, roi, rx1, ry1, W_f, H_f) -> Optional[BBox]:
        if self._tmpl_crop is None or self._tmpl_size is None:
            return None
        tw, th = self._tmpl_size
        best_val, best_bbox = -1.0, None
        for scale in np.linspace(0.75, 1.25, 9):
            nw = max(8, int(tw * scale))
            nh = max(8, int(th * scale))
            if roi.shape[0] < nh or roi.shape[1] < nw:
                continue
            res = cv2.matchTemplate(roi, cv2.resize(self._tmpl_crop, (nw, nh)),
                                    cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val  = max_val
                best_bbox = (rx1 + max_loc[0], ry1 + max_loc[1], nw, nh)
        if best_val < 0.35:
            return None
        return clamp_bbox(best_bbox, W_f, H_f)


# ─── Main tracker ────────────────────────────────────────────────────────────

class RobustTracker:
    def __init__(self, cfg: TrackerConfig, fg_extractor: ForestForegroundExtractor):
        self.cfg         = cfg
        self._fg         = fg_extractor
        self.state       = TrackState.LOST
        self._tracker    = None
        self._kalman     = None
        self._redetector = RANSACRedetector(cfg)
        self._median     = MedianTrajectoryFilter(cfg.median_window)
        self._bbox:      Optional[BBox]       = None
        self._ref_hist:  Optional[np.ndarray] = None
        self._occ_count:         int = 0
        self._frame_idx:         int = 0
        self._tmpl_update_count: int = 0

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        H, W = frame.shape[:2]
        bbox           = clamp_bbox(bbox, W, H)
        mask           = self._fg.get_mask(frame, bbox, learn_rate=0.05)
        self._ref_hist = calc_hist_masked(frame, bbox, mask)
        self._bbox     = bbox
        self._kalman   = HuberKalmanTracker(bbox, self.cfg)
        self._redetector.set_template(frame, bbox, mask)
        self._median.reset()
        self._occ_count         = 0
        self._frame_idx         = 0
        self._tmpl_update_count = 0
        self._tracker           = self._new_cv_tracker()
        self._tracker.init(frame, bbox)
        self.state = TrackState.TRACKING

    def _new_cv_tracker(self):
        return (cv2.TrackerKCF_create() if self.cfg.tracker_type == "KCF"
                else cv2.TrackerCSRT_create())

    def update(self, frame: np.ndarray) -> Tuple[TrackState, Optional[BBox]]:
        if self._kalman is None:
            return TrackState.LOST, None
        self._frame_idx += 1
        H, W = frame.shape[:2]

        kp       = clamp_bbox(self._kalman.predict(), W, H)
        unc      = self._kalman.position_uncertainty()
        iou_thr  = max(0.05, self.cfg.iou_threshold  - unc * 0.002)
        hist_thr = min(0.65, self.cfg.hist_threshold + unc * 0.002)

        cv_bbox   = self._cv_update(frame)
        result = self._verify(frame, cv_bbox, kp, iou_thr, hist_thr)

        if result is not None:
            confirmed, mask = result
            _kp_center  = (kp[0] + kp[2] // 2, kp[1] + kp[3] // 2)
            confirmed   = center_bbox_on_mask(confirmed, mask, W, H,
                                              frame=frame,
                                              ref_hist=self._ref_hist,
                                              kalman_center=_kp_center)
            corrected = clamp_bbox(self._kalman.correct(confirmed), W, H)
            smoothed  = self._median.update(corrected)
            self._bbox = smoothed
            _h = calc_hist_masked(frame, smoothed, mask)
            if (_h is not None and self._ref_hist is not None
                    and hist_dist(self._ref_hist, _h) < hist_thr * 0.80):
                self._update_appearance(frame, smoothed, mask)
            self._occ_count = 0
            prev       = self.state
            self.state = (TrackState.REDETECTED
                          if prev in (TrackState.OCCLUDED, TrackState.LOST)
                          else TrackState.TRACKING)
            if self.state == TrackState.REDETECTED:
                self._restart_cv(frame, smoothed)
            return self.state, self._bbox

        self._occ_count += 1
        self._kalman.decay_velocity()
        self._fg.update_background(frame, learn_rate=0.005)

        is_lost = self._occ_count > self.cfg.max_occluded_frames
        self.state = TrackState.LOST if is_lost else TrackState.OCCLUDED
        self._bbox = kp

        if self._frame_idx % self.cfg.redetect_interval == 0:
            search_bbox = kp
            if is_lost:
                exp = max(0, int(unc * 3))
                sx, sy, sw, sh = kp
                search_bbox = clamp_bbox(
                    (sx - exp, sy - exp, sw + 2 * exp, sh + 2 * exp), W, H)
            found = self._redetector.detect(frame, search_bbox)
            if found is not None:
                mask   = self._fg.get_mask(frame, found, learn_rate=0.0)
                fgr    = fg_ratio(mask)
                h_cand = calc_hist_masked(frame, found, mask)
                hdist  = (hist_dist(self._ref_hist, h_cand)
                          if h_cand is not None and self._ref_hist is not None
                          else 999.)
                if fgr >= self.cfg.fg_ratio_threshold or (fgr >= self.cfg.fg_ratio_threshold * 0.5 and hdist < hist_thr * 0.75):
                    if hdist < hist_thr:
                        _kp_center = (kp[0] + kp[2] // 2, kp[1] + kp[3] // 2)
                        found      = center_bbox_on_mask(found, mask, W, H,
                                                         frame=frame,
                                                         ref_hist=self._ref_hist,
                                                         kalman_center=_kp_center)
                        found     = refine_bbox_to_contour(frame, found, W, H)
                        corrected = clamp_bbox(self._kalman.correct(found), W, H)
                        smoothed  = self._median.update(corrected)
                        self._bbox = smoothed
                        _h = calc_hist_masked(frame, smoothed, mask)
                        if (_h is not None and self._ref_hist is not None
                                and hist_dist(self._ref_hist, _h) < hist_thr * 0.80):
                            self._update_appearance(frame, smoothed, mask)
                        self._occ_count = 0
                        self.state = TrackState.REDETECTED
                        self._restart_cv(frame, smoothed)

        return self.state, self._bbox

    def _cv_update(self, frame) -> Optional[BBox]:
        if self._tracker is None:
            return None
        try:
            ok, raw = self._tracker.update(frame)
        except Exception:
            return None
        return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])) if ok else None

    def _restart_cv(self, frame, bbox):
        self._tracker = self._new_cv_tracker()
        try:
            self._tracker.init(frame, bbox)
        except Exception as e:
            print(f"[warn] CV reinit: {e}")
            self._tracker = None

    def _verify(self, frame, cv_bbox, kp, iou_thr, hist_thr):
        if cv_bbox is None or self._ref_hist is None:
            return None
        H, W   = frame.shape[:2]
        cv_bbox = clamp_bbox(cv_bbox, W, H)
        mask = self._fg.get_mask(frame, cv_bbox, learn_rate=0.0)
        if fg_ratio(mask) < self.cfg.fg_ratio_threshold:
            if not self._hist_ok(frame, cv_bbox, hist_thr * 0.65):
                return None
            mask = None
        if not self._hist_ok(frame, cv_bbox, hist_thr, mask):
            return None
        if iou(cv_bbox, kp) < iou_thr:
            h = calc_hist_masked(frame, cv_bbox, mask)
            if h is None or hist_dist(self._ref_hist, h) >= hist_thr * 0.55:
                return None
        return cv_bbox, mask

    def _hist_ok(self, frame, bbox, thresh, mask=None) -> bool:
        if self._ref_hist is None:
            return False
        h = calc_hist_masked(frame, bbox, mask)
        if h is None:
            return False
        return hist_dist(self._ref_hist, h) < thresh

    def _update_appearance(self, frame, bbox, mask=None):
        a = self.cfg.appearance_alpha
        h = calc_hist_masked(frame, bbox, mask)
        if h is not None and self._ref_hist is not None:
            self._ref_hist = (1 - a) * self._ref_hist + a * h
        self._tmpl_update_count += 1
        if self._tmpl_update_count % 5 == 0:
            self._redetector.set_template(frame, bbox, mask)


# ─── Visualisation ───────────────────────────────────────────────────────────

STATE_COLORS = {
    TrackState.TRACKING:   (0, 220, 0),
    TrackState.OCCLUDED:   (0, 165, 255),
    TrackState.REDETECTED: (0, 215, 255),
    TrackState.LOST:       (0, 0, 220),
}
STATE_LABELS = {
    TrackState.TRACKING:   "TRACKING",
    TrackState.OCCLUDED:   "OCCLUDED",
    TrackState.REDETECTED: "REDETECTED",
    TrackState.LOST:       "LOST",
}


def draw_result(frame, state, bbox, uncertainty=0.):
    vis   = frame.copy()
    color = STATE_COLORS[state]
    label = STATE_LABELS[state]
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), color,
                      3 if state == TrackState.TRACKING else 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (x+w//2, y+h//2), color,
                       cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        info = f"{label}  u={uncertainty:.1f}px"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x, y-th-8), (x+tw+4, y), color, -1)
        cv2.putText(vis, info, (x+2, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, f"[{label}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Mode Tracker — fixed window size, interactive correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video",  type=str, metavar="PATH")
    src.add_argument("--camera", type=int, default=0, metavar="ID")
    p.add_argument("--tracker",            type=str,   default="CSRT",
                   choices=["CSRT", "KCF"])
    p.add_argument("--max-occluded",       type=int,   default=60)
    p.add_argument("--huber-delta",        type=float, default=15.0)
    p.add_argument("--hist-threshold",     type=float, default=0.52)
    p.add_argument("--iou-threshold",      type=float, default=0.15)
    p.add_argument("--fg-ratio-threshold", type=float, default=0.20)
    p.add_argument("--mog2-history",       type=int,   default=200)
    p.add_argument("--mog2-var-threshold", type=float, default=50.0)
    p.add_argument("--canny-low",          type=int,   default=40)
    p.add_argument("--canny-high",         type=int,   default=120)
    p.add_argument("--redetect-interval",  type=int,   default=3)
    p.add_argument("--template-scale",     type=float, default=3.0)
    p.add_argument("--median-window",      type=int,   default=5)
    p.add_argument("--velocity-decay",     type=float, default=0.75)
    p.add_argument("--warmup-frames",      type=int,   default=30)
    p.add_argument("--output",             type=str,   default=None)
    p.add_argument("--mode",               type=int,   default=1, choices=[1, 2],
                   help="1 = normal tracking; 2 = interactive correction every 4 frames")
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {path}")


def _overlay_bbox(vis: np.ndarray, bbox: Optional[BBox],
                  label: str, color: Tuple[int, int, int]) -> None:
    """Draw a secondary tracker bbox directly onto vis (in-place)."""
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    cv2.drawMarker(vis, (x + w // 2, y + h // 2), color,
                   cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(vis, (x, y - th - 6), (x + tw + 2, y), color, -1)
    cv2.putText(vis, label, (x + 1, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _build_cfg(args, tracker_type: str) -> TrackerConfig:
    return TrackerConfig(
        tracker_type         = tracker_type,
        max_occluded_frames  = args.max_occluded,
        huber_delta          = args.huber_delta,
        hist_threshold       = args.hist_threshold,
        iou_threshold        = args.iou_threshold,
        fg_ratio_threshold   = args.fg_ratio_threshold,
        mog2_history         = args.mog2_history,
        mog2_var_threshold   = args.mog2_var_threshold,
        canny_low            = args.canny_low,
        canny_high           = args.canny_high,
        redetect_interval    = args.redetect_interval,
        template_scale       = args.template_scale,
        median_window        = args.median_window,
        velocity_decay       = args.velocity_decay,
    )


# ─── Run ─────────────────────────────────────────────────────────────────────

def run(args):
    source = args.video if args.video else args.camera
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    plot_dir = os.path.join("plots", f"mode{args.mode}")

    # ── Tracker setup ────────────────────────────────────────────────────────
    if args.mode == 2:
        csrt_cfg = _build_cfg(args, "CSRT")
        kcf_cfg  = _build_cfg(args, "KCF")
        fg_csrt  = ForestForegroundExtractor(csrt_cfg)
        fg_kcf   = ForestForegroundExtractor(kcf_cfg)
        if args.warmup_frames > 0 and args.video:
            fg_csrt.warmup(cap, args.warmup_frames)
            # rewind happens inside warmup; run KCF warmup from same position
            fg_kcf.warmup(cap, args.warmup_frames)
        csrt_tracker = RobustTracker(csrt_cfg, fg_csrt)
        kcf_tracker  = RobustTracker(kcf_cfg,  fg_kcf)
    else:
        cfg         = _build_cfg(args, args.tracker)
        fg_extractor = ForestForegroundExtractor(cfg)
        if args.warmup_frames > 0 and args.video:
            fg_extractor.warmup(cap, args.warmup_frames)
        tracker = RobustTracker(cfg, fg_extractor)

    W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    initialized = False
    init_w = init_h = None
    frame_count = 0

    # Mode 1 data
    auto_corners: List[Tuple[int, int]] = []

    # Mode 2 data — all lists stay index-aligned (appended together)
    csrt_corners: List[Tuple[int, int]] = []
    kcf_corners:  List[Tuple[int, int]] = []
    user_corners: List[Tuple[int, int]] = []
    csrt_states:  List[TrackState]      = []
    kcf_states:   List[TrackState]      = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_vid, H_vid))

    mode_label = "Mode 1 — normal tracking" if args.mode == 1 else \
                 "Mode 2 — CSRT + KCF parallel with interactive correction"
    print(f"\n[{mode_label}]")
    print("Select the object to track, then press SPACE or ENTER.  ESC — cancel and exit.")
    print("Keys during playback: 'q' — quit\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # ── Initialisation ────────────────────────────────────────────────
        if not initialized:
            bbox = cv2.selectROI("Select object", frame,
                                 fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select object")
            if bbox[2] > 0 and bbox[3] > 0:
                init_w, init_h = bbox[2], bbox[3]
                if args.mode == 2:
                    csrt_tracker.init(frame, bbox)
                    kcf_tracker.init(frame, bbox)
                else:
                    tracker.init(frame, bbox)
                initialized = True
                frame_count = 0
            else:
                print("Selection cancelled — exiting.")
                break
            continue

        frame_count += 1

        # ── Update trackers ───────────────────────────────────────────────
        if args.mode == 2:
            csrt_state, csrt_bbox = csrt_tracker.update(frame)
            kcf_state,  kcf_bbox  = kcf_tracker.update(frame)
            if csrt_bbox is not None:
                csrt_bbox = clamp_bbox((csrt_bbox[0], csrt_bbox[1], init_w, init_h),
                                       W_vid, H_vid)
            if kcf_bbox is not None:
                kcf_bbox = clamp_bbox((kcf_bbox[0], kcf_bbox[1], init_w, init_h),
                                      W_vid, H_vid)
        else:
            state, bbox = tracker.update(frame)
            if bbox is not None:
                bbox = clamp_bbox((bbox[0], bbox[1], init_w, init_h), W_vid, H_vid)

        # ── Every-4-frame logic ───────────────────────────────────────────
        if frame_count % 10 == 0:
            if args.mode == 1:
                if bbox is not None:
                    auto_corners.append((bbox[0], bbox[1]))

            else:  # mode 2
                hint = frame.copy()
                # Show both tracker bboxes on the hint frame
                if csrt_bbox:
                    _overlay_bbox(hint, csrt_bbox, "CSRT", (0, 220, 0))
                if kcf_bbox:
                    _overlay_bbox(hint, kcf_bbox,  "KCF",  (0, 165, 255))
                cv2.putText(hint, "Draw correction window (SPACE/ENTER confirm, ESC = exit)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                            cv2.LINE_AA)
                user_sel = cv2.selectROI("Correction", hint,
                                         fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Correction")

                if user_sel[2] == 0 and user_sel[3] == 0:
                    print("Correction cancelled — exiting.")
                    break

                # Record all three lists together to stay index-aligned.
                # Trackers are NOT re-initialised — they continue from where they stopped.
                if user_sel[2] > 0 or user_sel[3] > 0:
                    if csrt_bbox is not None:
                        csrt_corners.append((csrt_bbox[0], csrt_bbox[1]))
                        csrt_states.append(csrt_state)
                    if kcf_bbox is not None:
                        kcf_corners.append((kcf_bbox[0], kcf_bbox[1]))
                        kcf_states.append(kcf_state)
                    ux, uy = int(user_sel[0]), int(user_sel[1])
                    user_corners.append((ux, uy))

        # ── Visualise ─────────────────────────────────────────────────────
        if args.mode == 2:
            csrt_unc = csrt_tracker._kalman.position_uncertainty() \
                       if csrt_tracker._kalman else 0.
            vis = draw_result(frame, csrt_state, csrt_bbox, csrt_unc)
            _overlay_bbox(vis, kcf_bbox, "KCF", (0, 165, 255))
        else:
            unc = tracker._kalman.position_uncertainty() if tracker._kalman else 0.
            vis = draw_result(frame, state, bbox, unc)

        if writer:
            writer.write(vis)
        cv2.imshow("Mode Tracker", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[saved] {args.output}")
    cv2.destroyAllWindows()

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.mode == 1:
        if auto_corners:
            fig, ax = plt.subplots()
            xs, ys = zip(*auto_corners)
            ax.plot(xs, ys)
            ax.invert_yaxis()
            _save_fig(fig, os.path.join(plot_dir, "trajectory.png"))
            plt.show()
            plt.close(fig)

    else:  # mode 2
        n = min(len(csrt_corners), len(kcf_corners), len(user_corners))
        if n == 0:
            return

        indices = list(range(1, n + 1))

        csrt_dx  = [abs(csrt_corners[i][0] - user_corners[i][0]) for i in range(n)]
        csrt_dy  = [abs(csrt_corners[i][1] - user_corners[i][1]) for i in range(n)]
        csrt_euc = [(csrt_dx[i]**2 + csrt_dy[i]**2) ** 0.5       for i in range(n)]
        kcf_dx   = [abs(kcf_corners[i][0]  - user_corners[i][0]) for i in range(n)]
        kcf_dy   = [abs(kcf_corners[i][1]  - user_corners[i][1]) for i in range(n)]
        kcf_euc  = [(kcf_dx[i]**2  + kcf_dy[i]**2)  ** 0.5       for i in range(n)]

        _SC = {
            TrackState.TRACKING:   "green",
            TrackState.OCCLUDED:   "orange",
            TrackState.LOST:       "red",
            TrackState.REDETECTED: "red",
        }
        _LEGEND = [
            Line2D([0], [0], color="green",      linewidth=1.5, label="TRACKING"),
            Line2D([0], [0], color="orange",     linewidth=1.5, label="OCCLUDED"),
            Line2D([0], [0], color="red",        linewidth=1.5, label="LOST / REDETECTED"),
            Line2D([0], [0], color="deepskyblue", linewidth=1.5, label="User"),
        ]

        def _draw_traj(ax, corners, states, title):
            xs = [p[0] for p in corners[:n]]
            ys = [p[1] for p in corners[:n]]
            for i in range(n - 1):
                ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                        color=_SC[states[i]], linewidth=1.5)
            if n == 1:
                ax.scatter(xs, ys, color=_SC[states[0]], s=30)
            uxs = [p[0] for p in user_corners[:n]]
            uys = [p[1] for p in user_corners[:n]]
            ax.plot(uxs, uys, color="deepskyblue", linewidth=1.5)
            ax.invert_yaxis()
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_title(title)
            ax.legend(handles=_LEGEND)

        # ── Trajectory CSRT ──────────────────────────────────────────────
        fig_tc, ax_tc = plt.subplots(figsize=(7, 5))
        _draw_traj(ax_tc, csrt_corners, csrt_states, "CSRT trajectory vs User")
        fig_tc.tight_layout()
        _save_fig(fig_tc, os.path.join(plot_dir, "trajectory_csrt.png"))
        plt.show()
        plt.close(fig_tc)

        # ── Trajectory KCF ───────────────────────────────────────────────
        fig_tk, ax_tk = plt.subplots(figsize=(7, 5))
        _draw_traj(ax_tk, kcf_corners, kcf_states, "KCF trajectory vs User")
        fig_tk.tight_layout()
        _save_fig(fig_tk, os.path.join(plot_dir, "trajectory_kcf.png"))
        plt.show()
        plt.close(fig_tk)

        # ── Errors CSRT ──────────────────────────────────────────────────
        fig_ec, ax_ec = plt.subplots(figsize=(8, 5))
        ax_ec.plot(csrt_dx,  indices, color="steelblue", linestyle="-",  label="|Δx|")
        ax_ec.plot(csrt_dy,  indices, color="steelblue", linestyle="--", label="|Δy|")
        ax_ec.plot(csrt_euc, indices, color="steelblue", linestyle=":",  label="Euclidean")
        ax_ec.set_xlabel("Absolute error (px)")
        ax_ec.set_ylabel("Correction index")
        ax_ec.set_title("CSRT absolute errors vs User")
        ax_ec.legend()
        fig_ec.tight_layout()
        _save_fig(fig_ec, os.path.join(plot_dir, "errors_csrt.png"))
        plt.show()
        plt.close(fig_ec)

        # ── Errors KCF ───────────────────────────────────────────────────
        fig_ek, ax_ek = plt.subplots(figsize=(8, 5))
        ax_ek.plot(kcf_dx,  indices, color="darkorange", linestyle="-",  label="|Δx|")
        ax_ek.plot(kcf_dy,  indices, color="darkorange", linestyle="--", label="|Δy|")
        ax_ek.plot(kcf_euc, indices, color="darkorange", linestyle=":",  label="Euclidean")
        ax_ek.set_xlabel("Absolute error (px)")
        ax_ek.set_ylabel("Correction index")
        ax_ek.set_title("KCF absolute errors vs User")
        ax_ek.legend()
        fig_ek.tight_layout()
        _save_fig(fig_ek, os.path.join(plot_dir, "errors_kcf.png"))
        plt.show()
        plt.close(fig_ek)

        # ── Distance CSRT ────────────────────────────────────────────────
        fig_dc, ax_dc = plt.subplots(figsize=(8, 4))
        ax_dc.plot(indices, csrt_euc, color="steelblue", linewidth=1.5)
        ax_dc.set_xlabel("Correction index")
        ax_dc.set_ylabel("Distance (px)")
        ax_dc.set_title("CSRT — distance from tracker corner to User corner")
        ax_dc.grid(True, linestyle="--", alpha=0.4)
        fig_dc.tight_layout()
        _save_fig(fig_dc, os.path.join(plot_dir, "distance_csrt.png"))
        plt.show()
        plt.close(fig_dc)

        # ── Distance KCF ─────────────────────────────────────────────────
        fig_dk, ax_dk = plt.subplots(figsize=(8, 4))
        ax_dk.plot(indices, kcf_euc, color="darkorange", linewidth=1.5)
        ax_dk.set_xlabel("Correction index")
        ax_dk.set_ylabel("Distance (px)")
        ax_dk.set_title("KCF — distance from tracker corner to User corner")
        ax_dk.grid(True, linestyle="--", alpha=0.4)
        fig_dk.tight_layout()
        _save_fig(fig_dk, os.path.join(plot_dir, "distance_kcf.png"))
        plt.show()
        plt.close(fig_dk)


if __name__ == "__main__":
    run(parse_args())
