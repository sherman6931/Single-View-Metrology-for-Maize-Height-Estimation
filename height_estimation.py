from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


METHOD_NAME = "hough_voting_ransac_v1"

ROOT_DIR = Path(r"E:\maize\predict_full_image_best_one4.0")
CSV_BY_GROUP_DIR = ROOT_DIR / "csv_by_group"
ORIGINAL_IMAGE_ROOT = Path(r"E:\maize\orignaldata3-expand"

)

OUTPUT_DIR = ROOT_DIR / f"geom_{METHOD_NAME}"
OUT_ALL_CSV = OUTPUT_DIR / f"all_{METHOD_NAME}_height.csv"
OUT_OVERALL_CSV = OUTPUT_DIR / f"summary_{METHOD_NAME}_overall.csv"
OUT_GROUP_SUMMARY_CSV = OUTPUT_DIR / f"summary_{METHOD_NAME}_by_group.csv"
OUT_STATUS_CSV = OUTPUT_DIR / f"status_count_{METHOD_NAME}.csv"
OUT_CANDIDATE_SUMMARY_CSV = OUTPUT_DIR / f"summary_{METHOD_NAME}_by_horizon_candidate.csv"
OUT_DIAGNOSTICS_CSV = OUTPUT_DIR / f"horizon_diagnostics_{METHOD_NAME}.csv"
OUT_GROUP_CSV_DIR = OUTPUT_DIR / f"csv_by_group_{METHOD_NAME}"
OUT_VIS_DIR = OUTPUT_DIR / f"visual_{METHOD_NAME}"

SAVE_VISUAL = True

ERASE_PLANT_BBOX_BEFORE_HOUGH = True
ERASE_BBOX_EXPAND_RATIO = 0.12

USE_CLAHE = True
GAUSSIAN_BLUR_KSIZE = 5
CANNY_SIGMA = 0.33

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180.0
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 90
HOUGH_MAX_LINE_GAP = 40

MIN_LINE_LENGTH_PX = 80

RANSAC_ITER = 2500
RANSAC_DIST_THRESH_PX = 14.0
MAX_VP_ABS_FACTOR = 60

STRICT_KEYPOINT_BORDER = True
KEYPOINT_BORDER_PX = 8

MIN_VALID_HEIGHT_CM = 20.0
MAX_VALID_HEIGHT_CM = 350.0

Y_HOR_MIN_FACTOR = -0.65
Y_HOR_MAX_FACTOR = 1.35
Y_V_MAX_ABS_FACTOR = 25.0

DIAG_MIN_ANGLE = 8
DIAG_MAX_ANGLE = 82

VERTICAL_MIN_ANGLE = 70
VERTICAL_MAX_ANGLE = 110

HORIZONTAL_TOL_ANGLE = 14
MAX_HORIZON_TILT_DEG = 25

USE_STRUCTURAL_ROI = True
MASK_PLANT_BBOX_AFTER_HOUGH = True
BBOX_EXPAND_RATIO_AFTER_HOUGH = 0.12

MIN_DIAG_INLIERS = 2
MIN_VERTICAL_INLIERS = 4
MIN_VERTICAL_INLIER_RATIO = 0.020


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_line(line: np.ndarray) -> Optional[np.ndarray]:
    a, b, c = line
    norm = math.hypot(a, b)
    if norm < 1e-12:
        return None
    return np.array([a / norm, b / norm, c / norm], dtype=np.float64)


def line_from_points(x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return normalize_line(np.array([a, b, c], dtype=np.float64))


def intersection_of_lines(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-10:
        return None
    xy = p[:2] / p[2]
    if not np.all(np.isfinite(xy)):
        return None
    return xy.astype(np.float64)


def y_on_line_at_x(line: np.ndarray, x: float) -> float:
    a, b, c = line
    if abs(b) < 1e-9:
        return np.nan
    return float(-(a * x + c) / b)


def line_tilt_deg(line: np.ndarray) -> float:
    a, b, _ = line
    angle = math.degrees(math.atan2(-a, b)) % 180.0
    return float(min(angle, 180.0 - angle))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumsum = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    return float(values[np.searchsorted(cumsum, cutoff)])


def parse_camera_and_height_from_path(csv_path: Path) -> Tuple[float, float]:
    try:
        true_h = float(csv_path.parent.name)
        cam_h = float(csv_path.parent.parent.name)
        return cam_h, true_h
    except Exception:
        return np.nan, np.nan


def resolve_image_path(row: pd.Series) -> Optional[Path]:
    candidates: List[Path] = []

    image_path = row.get("image_path", "")
    if isinstance(image_path, str) and image_path.strip():
        candidates.append(Path(image_path))

    relative_path = row.get("relative_path", "")
    if isinstance(relative_path, str) and relative_path.strip():
        candidates.append(ORIGINAL_IMAGE_ROOT / relative_path)

    image_name = row.get("image_name", "")
    if isinstance(image_name, str) and image_name.strip():
        candidates.append(ORIGINAL_IMAGE_ROOT / image_name)

    for p in candidates:
        if p.exists():
            return p

    return candidates[0] if candidates else None


def get_image_name(row: pd.Series, img_path: Optional[Path], idx: int) -> str:
    image_name = row.get("image_name", "")
    if isinstance(image_name, str) and image_name.strip():
        return Path(image_name).name
    if img_path is not None:
        return img_path.name
    return f"row_{idx}.jpg"


def expanded_bbox_from_row(row: pd.Series, w: int, h: int, expand_ratio: float) -> Optional[Tuple[int, int, int, int]]:
    x1 = safe_float(row.get("bbox_x1"))
    y1 = safe_float(row.get("bbox_y1"))
    x2 = safe_float(row.get("bbox_x2"))
    y2 = safe_float(row.get("bbox_y2"))

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    x1 -= bw * expand_ratio
    x2 += bw * expand_ratio
    y1 -= bh * expand_ratio
    y2 += bh * expand_ratio

    return (
        int(max(0, round(x1))),
        int(max(0, round(y1))),
        int(min(w - 1, round(x2))),
        int(min(h - 1, round(y2))),
    )


def keypoints_are_valid(row: pd.Series, img_h: int) -> Tuple[bool, str]:
    top_x = safe_float(row.get("top_x"))
    top_y = safe_float(row.get("top_y"))
    base_x = safe_float(row.get("base_x"))
    base_y = safe_float(row.get("base_y"))

    if not np.all(np.isfinite([top_x, top_y, base_x, base_y])):
        return False, "invalid_keypoints_nan"

    if base_y <= top_y:
        return False, "invalid_keypoints_order"

    if STRICT_KEYPOINT_BORDER:
        if top_y <= KEYPOINT_BORDER_PX:
            return False, "invalid_keypoint_top_on_border"
        if base_y >= img_h - 1 - KEYPOINT_BORDER_PX:
            return False, "invalid_keypoint_base_on_border"

    return True, "keypoints_ok"


def erase_bbox_region(img_bgr: np.ndarray, row: pd.Series) -> np.ndarray:
    if not ERASE_PLANT_BBOX_BEFORE_HOUGH:
        return img_bgr

    h, w = img_bgr.shape[:2]
    bbox = expanded_bbox_from_row(row, w, h, ERASE_BBOX_EXPAND_RATIO)

    if bbox is None:
        return img_bgr

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return img_bgr

    out = img_bgr.copy()
    med = np.median(out.reshape(-1, 3), axis=0).astype(np.uint8)
    out[y1:y2, x1:x2] = med
    return out


def make_edges_for_hough(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if GAUSSIAN_BLUR_KSIZE and GAUSSIAN_BLUR_KSIZE >= 3:
        gray = cv2.GaussianBlur(gray, (GAUSSIAN_BLUR_KSIZE, GAUSSIAN_BLUR_KSIZE), 0)

    med = np.median(gray)
    lower = int(max(0, (1.0 - CANNY_SIGMA) * med))
    upper = int(min(255, (1.0 + CANNY_SIGMA) * med))

    if upper <= lower:
        lower, upper = 50, 150

    edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges


def add_segment(segments: List[Dict], x1: float, y1: float, x2: float, y2: float) -> None:
    length = math.hypot(x2 - x1, y2 - y1)
    if length < MIN_LINE_LENGTH_PX:
        return
    line = line_from_points(x1, y1, x2, y2)
    if line is None:
        return
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    slope = (y2 - y1) / (x2 - x1 + 1e-9)
    segments.append({
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        "length": float(length), "angle": float(angle), "slope": float(slope),
        "line": line, "mid_x": float(0.5 * (x1 + x2)), "mid_y": float(0.5 * (y1 + y2)),
    })


def detect_hough_segments(img_bgr: np.ndarray, row: pd.Series) -> Tuple[List[Dict], np.ndarray]:
    img_for_hough = erase_bbox_region(img_bgr, row)
    edges = make_edges_for_hough(img_for_hough)

    raw = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    segments: List[Dict] = []
    if raw is not None:
        for item in raw.reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in item]
            add_segment(segments, x1, y1, x2, y2)

    return segments, edges


def make_roi_masks(img_shape: Tuple[int, int, int], row: pd.Series) -> Dict[str, np.ndarray]:
    h, w = img_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    if USE_STRUCTURAL_ROI:
        horizon_mask = (yy < 0.72 * h) | (xx < 0.38 * w) | (xx > 0.62 * w)
        vertical_mask = (xx < 0.42 * w) | (xx > 0.58 * w) | (yy < 0.65 * h)
        horizontal_mask = yy < 0.72 * h
    else:
        horizon_mask = np.ones((h, w), dtype=bool)
        vertical_mask = np.ones((h, w), dtype=bool)
        horizontal_mask = np.ones((h, w), dtype=bool)

    if MASK_PLANT_BBOX_AFTER_HOUGH:
        bbox = expanded_bbox_from_row(row, w, h, BBOX_EXPAND_RATIO_AFTER_HOUGH)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            horizon_mask[y1:y2, x1:x2] = False
            vertical_mask[y1:y2, x1:x2] = False
            horizontal_mask[y1:y2, x1:x2] = False

    return {
        "horizon": horizon_mask.astype(np.uint8),
        "vertical": vertical_mask.astype(np.uint8),
        "horizontal": horizontal_mask.astype(np.uint8),
    }


def segment_midpoint_in_mask(seg: Dict, mask: np.ndarray) -> bool:
    h, w = mask.shape[:2]
    mx = int(round(seg["mid_x"]))
    my = int(round(seg["mid_y"]))
    if mx < 0 or mx >= w or my < 0 or my >= h:
        return False
    return bool(mask[my, mx] > 0)


def split_line_groups(segments: List[Dict], masks: Dict[str, np.ndarray]) -> Dict[str, List[Dict]]:
    diag_pos: List[Dict] = []
    diag_neg: List[Dict] = []
    vertical: List[Dict] = []
    horizontal: List[Dict] = []

    for s in segments:
        a = s["angle"]
        in_horizon_roi = segment_midpoint_in_mask(s, masks["horizon"])
        in_vertical_roi = segment_midpoint_in_mask(s, masks["vertical"])
        in_horizontal_roi = segment_midpoint_in_mask(s, masks["horizontal"])

        if VERTICAL_MIN_ANGLE <= a <= VERTICAL_MAX_ANGLE:
            if in_vertical_roi:
                vertical.append(s)
        elif a <= HORIZONTAL_TOL_ANGLE or a >= 180 - HORIZONTAL_TOL_ANGLE:
            if in_horizontal_roi:
                horizontal.append(s)
        elif DIAG_MIN_ANGLE <= a <= DIAG_MAX_ANGLE:
            if in_horizon_roi:
                diag_pos.append(s)
        elif 180 - DIAG_MAX_ANGLE <= a <= 180 - DIAG_MIN_ANGLE:
            if in_horizon_roi:
                diag_neg.append(s)

    return {
        "diag_pos": diag_pos,
        "diag_neg": diag_neg,
        "vertical": vertical,
        "horizontal": horizontal,
    }


def estimate_vp_ransac(segs: List[Dict], img_shape: Tuple[int, int], name: str) -> Dict:
    n = len(segs)
    if n < 2:
        return {"vp": None, "inlier_indices": [], "inlier_count": 0, "inlier_ratio": 0.0,
                "status": f"{name}: too_few_lines({n})"}

    h, w = img_shape[:2]
    max_abs = MAX_VP_ABS_FACTOR * max(w, h)
    lines = np.stack([s["line"] for s in segs], axis=0)
    weights = np.array([max(1.0, s["length"]) for s in segs], dtype=np.float64)
    prob = weights / np.sum(weights)
    rng = np.random.default_rng(20260628)

    best_xy = None
    best_inliers: List[int] = []
    best_score = -1.0

    for _ in range(RANSAC_ITER):
        try:
            i, j = rng.choice(n, size=2, replace=False, p=prob)
        except Exception:
            i, j = rng.choice(n, size=2, replace=False)

        xy = intersection_of_lines(lines[i], lines[j])
        if xy is None:
            continue

        x, y = float(xy[0]), float(xy[1])
        if abs(x) > max_abs or abs(y) > max_abs:
            continue

        p = np.array([x, y, 1.0], dtype=np.float64)
        d = np.abs(lines @ p)
        inliers = np.where(d < RANSAC_DIST_THRESH_PX)[0]
        if len(inliers) < 2:
            continue

        score = float(np.sum(weights[inliers]))
        if score > best_score:
            best_score = score
            best_xy = np.array([x, y], dtype=np.float64)
            best_inliers = inliers.tolist()

    if best_xy is None or len(best_inliers) < 2:
        return {"vp": None, "inlier_indices": [], "inlier_count": 0, "inlier_ratio": 0.0,
                "status": f"{name}: ransac_failed"}

    try:
        A = lines[best_inliers]
        _, _, vt = np.linalg.svd(A)
        p = vt[-1]
        if abs(p[2]) > 1e-10:
            refined = p[:2] / p[2]
            if np.all(np.isfinite(refined)) and abs(refined[0]) < max_abs and abs(refined[1]) < max_abs:
                best_xy = refined.astype(np.float64)
    except Exception:
        pass

    inlier_count = len(best_inliers)
    inlier_ratio = inlier_count / max(1, n)
    return {
        "vp": best_xy,
        "inlier_indices": best_inliers,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "status": f"{name}: ok_inliers={inlier_count}/{n}, ratio={inlier_ratio:.3f}",
    }


def make_candidate_record(source: str, line: Optional[np.ndarray], y_hor: float, tilt: float, score: float, reason: str) -> Dict:
    return {
        "source": source,
        "line": line,
        "y_hor": float(y_hor) if np.isfinite(y_hor) else np.nan,
        "tilt_deg": float(tilt) if np.isfinite(tilt) else np.nan,
        "score": float(score) if np.isfinite(score) else np.nan,
        "reason": reason,
    }


def candidate_from_two_vps(vp_pos_info: Dict, vp_neg_info: Dict, x_ref: float) -> Dict:
    vp_pos = vp_pos_info["vp"]
    vp_neg = vp_neg_info["vp"]

    if vp_pos is None or vp_neg is None:
        return make_candidate_record("two_diagonal_vps", None, np.nan, np.nan, np.nan, "missing_vp")
    if vp_pos_info["inlier_count"] < MIN_DIAG_INLIERS:
        return make_candidate_record("two_diagonal_vps", None, np.nan, np.nan, np.nan, f"diag_pos_inliers_too_few={vp_pos_info['inlier_count']}")
    if vp_neg_info["inlier_count"] < MIN_DIAG_INLIERS:
        return make_candidate_record("two_diagonal_vps", None, np.nan, np.nan, np.nan, f"diag_neg_inliers_too_few={vp_neg_info['inlier_count']}")

    line = line_from_points(vp_pos[0], vp_pos[1], vp_neg[0], vp_neg[1])
    if line is None:
        return make_candidate_record("two_diagonal_vps", None, np.nan, np.nan, np.nan, "line_from_vps_failed")

    tilt = line_tilt_deg(line)
    y_hor = y_on_line_at_x(line, x_ref)
    score = 130.0 + 12.0 * vp_pos_info["inlier_count"] + 12.0 * vp_neg_info["inlier_count"] - 2.5 * tilt
    return make_candidate_record("two_diagonal_vps", line, y_hor, tilt, score, "ok")


def candidate_from_horizontal_weighted(horizontal_segs: List[Dict], img_shape: Tuple[int, int], x_ref: float) -> Dict:
    if len(horizontal_segs) < 1:
        return make_candidate_record("horizontal_weighted_median", None, np.nan, np.nan, np.nan, "no_horizontal_lines")

    h, w = img_shape[:2]
    cx = w / 2.0
    cand = [s for s in horizontal_segs if s["mid_y"] < 0.72 * h]
    if len(cand) == 0:
        cand = horizontal_segs

    y_values = []
    slopes = []
    weights = []
    for s in cand:
        y = y_on_line_at_x(s["line"], cx)
        if np.isfinite(y):
            y_values.append(y)
            slopes.append(s["slope"])
            weights.append(s["length"])

    if len(y_values) == 0:
        return make_candidate_record("horizontal_weighted_median", None, np.nan, np.nan, np.nan, "no_valid_y_from_horizontal_lines")

    y_med = weighted_median(np.array(y_values), np.array(weights))
    m_med = weighted_median(np.array(slopes), np.array(weights))
    if not np.isfinite(y_med) or not np.isfinite(m_med):
        return make_candidate_record("horizontal_weighted_median", None, np.nan, np.nan, np.nan, "weighted_median_failed")

    b0 = y_med - m_med * cx
    line = normalize_line(np.array([-m_med, 1.0, -b0], dtype=np.float64))
    if line is None:
        return make_candidate_record("horizontal_weighted_median", None, np.nan, np.nan, np.nan, "line_normalize_failed")

    tilt = line_tilt_deg(line)
    y_hor = y_on_line_at_x(line, x_ref)
    score = 70.0 + 0.05 * float(np.sum(weights)) - 2.0 * tilt
    return make_candidate_record("horizontal_weighted_median", line, y_hor, tilt, score, "ok")


def candidate_from_hough_voting(horizontal_segs: List[Dict], img_shape: Tuple[int, int], x_ref: float) -> Dict:
    if len(horizontal_segs) < 1:
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "no_horizontal_lines")

    h, w = img_shape[:2]
    cx = w / 2.0
    cand = [s for s in horizontal_segs if s["mid_y"] < 0.75 * h]
    if len(cand) == 0:
        cand = horizontal_segs

    y_vals = []
    slopes = []
    weights = []
    for s in cand:
        y = y_on_line_at_x(s["line"], cx)
        if not np.isfinite(y):
            continue
        tilt = min(s["angle"], 180.0 - s["angle"])
        tilt_weight = max(0.1, 1.0 - tilt / max(1.0, HORIZONTAL_TOL_ANGLE))
        wt = s["length"] * tilt_weight
        y_vals.append(y)
        slopes.append(s["slope"])
        weights.append(wt)

    if len(y_vals) == 0:
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "no_valid_y")

    y_vals = np.array(y_vals, dtype=np.float64)
    slopes = np.array(slopes, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    bin_size = max(20.0, h / 160.0)
    y_min = Y_HOR_MIN_FACTOR * h
    y_max = Y_HOR_MAX_FACTOR * h
    mask = np.isfinite(y_vals) & (y_vals >= y_min) & (y_vals <= y_max) & (weights > 0)

    if not np.any(mask):
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "all_y_out_of_range")

    y_vals = y_vals[mask]
    slopes = slopes[mask]
    weights = weights[mask]

    bin_ids = np.floor((y_vals - y_min) / bin_size).astype(int)
    unique_bins = np.unique(bin_ids)

    best_bin = None
    best_score = -1.0
    for b in unique_bins:
        idx = bin_ids == b
        score = float(np.sum(weights[idx]))
        if score > best_score:
            best_score = score
            best_bin = b

    if best_bin is None:
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "no_best_bin")

    idx = bin_ids == best_bin
    y_vote = weighted_median(y_vals[idx], weights[idx])
    slope_vote = weighted_median(slopes[idx], weights[idx])

    if not np.isfinite(y_vote) or not np.isfinite(slope_vote):
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "vote_median_failed")

    b0 = y_vote - slope_vote * cx
    line = normalize_line(np.array([-slope_vote, 1.0, -b0], dtype=np.float64))
    if line is None:
        return make_candidate_record("hough_y_voting", None, np.nan, np.nan, np.nan, "line_normalize_failed")

    tilt = line_tilt_deg(line)
    y_hor = y_on_line_at_x(line, x_ref)
    return make_candidate_record("hough_y_voting", line, y_hor, tilt, 90.0 + best_score - 2.0 * tilt, f"ok; bin_size={bin_size:.2f}; voted_lines={int(np.sum(idx))}")


def candidate_from_longest_horizontal(horizontal_segs: List[Dict], img_shape: Tuple[int, int], x_ref: float) -> Dict:
    if len(horizontal_segs) < 1:
        return make_candidate_record("longest_horizontal_line", None, np.nan, np.nan, np.nan, "no_horizontal_lines")

    h, _ = img_shape[:2]
    cand = [s for s in horizontal_segs if s["mid_y"] < 0.72 * h]
    if len(cand) == 0:
        cand = horizontal_segs

    s = max(cand, key=lambda z: z["length"])
    line = s["line"]
    tilt = line_tilt_deg(line)
    y_hor = y_on_line_at_x(line, x_ref)
    score = 45.0 + 0.02 * s["length"] - 2.0 * tilt
    return make_candidate_record("longest_horizontal_line", line, y_hor, tilt, score, "ok")


def is_valid_y_hor(y_hor: float, img_h: int) -> bool:
    if not np.isfinite(y_hor):
        return False
    return (Y_HOR_MIN_FACTOR * img_h) <= y_hor <= (Y_HOR_MAX_FACTOR * img_h)


def is_valid_horizon_candidate(c: Dict, img_h: int) -> Tuple[bool, str]:
    if c is None:
        return False, "candidate_is_none"
    if c.get("line") is None:
        return False, f"no_line:{c.get('reason', '')}"
    y_hor = c.get("y_hor", np.nan)
    tilt = c.get("tilt_deg", np.nan)
    if not is_valid_y_hor(y_hor, img_h):
        return False, f"invalid_y_hor={y_hor}"
    if not np.isfinite(tilt) or tilt > MAX_HORIZON_TILT_DEG:
        return False, f"tilt_too_large={tilt}"
    return True, "valid"


def compute_height_full_eq29(H_cam_cm: float, y_top: float, y_bot: float, y_hor: float, y_v: float) -> Tuple[float, str]:
    vals = [H_cam_cm, y_top, y_bot, y_hor, y_v]
    if not np.all(np.isfinite(vals)):
        return np.nan, "height_full_invalid_non_finite"

    denom1 = y_bot - y_hor
    denom2 = y_top - y_v
    if abs(denom1) < 1e-6:
        return np.nan, "height_full_invalid_denominator_ybot_yhor"
    if abs(denom2) < 1e-6:
        return np.nan, "height_full_invalid_denominator_ytop_yv"

    est = H_cam_cm * (1.0 - ((y_top - y_hor) / denom1) * ((y_bot - y_v) / denom2))
    if not np.isfinite(est):
        return np.nan, "height_full_invalid_result_non_finite"
    return float(est), "height_full_ok"


def compute_height_yv_infinity(H_cam_cm: float, y_top: float, y_bot: float, y_hor: float) -> Tuple[float, str]:
    vals = [H_cam_cm, y_top, y_bot, y_hor]
    if not np.all(np.isfinite(vals)):
        return np.nan, "height_infinity_invalid_non_finite"

    denom = y_bot - y_hor
    if abs(denom) < 1e-6:
        return np.nan, "height_infinity_invalid_denominator_ybot_yhor"

    est = H_cam_cm * ((y_bot - y_top) / denom)
    if not np.isfinite(est):
        return np.nan, "height_infinity_invalid_result_non_finite"
    return float(est), "height_infinity_ok"


def is_plausible_height(h: float) -> bool:
    if not np.isfinite(h):
        return False
    return MIN_VALID_HEIGHT_CM <= h <= MAX_VALID_HEIGHT_CM


def choose_height_estimate(H_cam_cm: float, y_top: float, y_bot: float, y_hor: float, y_v: float, vertical_vp_reliable: bool) -> Tuple[float, str, str]:
    est_inf, st_inf = compute_height_yv_infinity(H_cam_cm, y_top, y_bot, y_hor)

    if vertical_vp_reliable:
        est_full, st_full = compute_height_full_eq29(H_cam_cm, y_top, y_bot, y_hor, y_v)
        if is_plausible_height(est_full):
            return est_full, "full_eq29", st_full
        if is_plausible_height(est_inf):
            return est_inf, "yv_infinity_fallback_after_full_invalid", f"{st_full}; {st_inf}"
        return est_full, "full_eq29_invalid_no_good_fallback", f"{st_full}; {st_inf}"

    return est_inf, "yv_infinity", st_inf


def evaluate_candidate_height(candidate: Dict, H_cam_cm: float, true_height_cm: float, y_top: float, y_bot: float, y_v: float, vertical_vp_reliable: bool, img_h: int) -> Dict:
    source = candidate.get("source", "unknown")
    valid_hor, hor_reason = is_valid_horizon_candidate(candidate, img_h)

    out = {
        f"{source}_candidate_reason": candidate.get("reason", ""),
        f"{source}_valid_horizon": bool(valid_hor),
        f"{source}_invalid_reason": hor_reason,
        f"{source}_y_hor_px": candidate.get("y_hor", np.nan),
        f"{source}_tilt_deg": candidate.get("tilt_deg", np.nan),
        f"{source}_score": candidate.get("score", np.nan),
        f"{source}_estimated_height_cm": np.nan,
        f"{source}_height_mode": "",
        f"{source}_height_status": "",
        f"{source}_plausible_height": False,
        f"{source}_abs_error_cm": np.nan,
        f"{source}_rel_error_percent": np.nan,
    }

    if not valid_hor:
        return out

    est, mode, st = choose_height_estimate(H_cam_cm, y_top, y_bot, candidate["y_hor"], y_v, vertical_vp_reliable)
    out[f"{source}_estimated_height_cm"] = est
    out[f"{source}_height_mode"] = mode
    out[f"{source}_height_status"] = st
    out[f"{source}_plausible_height"] = bool(is_plausible_height(est))

    if is_plausible_height(est) and np.isfinite(true_height_cm) and true_height_cm > 0:
        abs_err = abs(est - true_height_cm)
        rel_err = abs_err / true_height_cm * 100.0
        out[f"{source}_abs_error_cm"] = abs_err
        out[f"{source}_rel_error_percent"] = rel_err

    return out


def estimate_geometric_priors(img_bgr: np.ndarray, row: pd.Series) -> Dict:
    h, w = img_bgr.shape[:2]
    all_segments, edges = detect_hough_segments(img_bgr, row)
    masks = make_roi_masks(img_bgr.shape, row)
    groups = split_line_groups(all_segments, masks)

    top_x = safe_float(row.get("top_x"))
    base_x = safe_float(row.get("base_x"))
    x_ref = np.nanmean([top_x, base_x])
    if not np.isfinite(x_ref):
        x_ref = w / 2.0

    vp_pos_info = estimate_vp_ransac(groups["diag_pos"], (h, w), "vp_diag_pos")
    vp_neg_info = estimate_vp_ransac(groups["diag_neg"], (h, w), "vp_diag_neg")
    vp_v_info = estimate_vp_ransac(groups["vertical"], (h, w), "vp_vertical")

    candidates = {
        "hough_y_voting": candidate_from_hough_voting(groups["horizontal"], (h, w), x_ref),
        "two_diagonal_vps": candidate_from_two_vps(vp_pos_info, vp_neg_info, x_ref),
        "horizontal_weighted_median": candidate_from_horizontal_weighted(groups["horizontal"], (h, w), x_ref),
        "longest_horizontal_line": candidate_from_longest_horizontal(groups["horizontal"], (h, w), x_ref),
    }

    vp_v = vp_v_info["vp"]
    vertical_vp_reliable = False
    y_v = np.nan
    vertical_vp_status = vp_v_info["status"]

    if vp_v is not None:
        y_v_candidate = float(vp_v[1])
        if (
            vp_v_info["inlier_count"] >= MIN_VERTICAL_INLIERS
            and vp_v_info["inlier_ratio"] >= MIN_VERTICAL_INLIER_RATIO
            and abs(y_v_candidate) <= Y_V_MAX_ABS_FACTOR * h
        ):
            vertical_vp_reliable = True
            y_v = y_v_candidate
            vertical_vp_status += "; vertical_vp_used"
        else:
            vertical_vp_status += "; vertical_vp_unreliable_use_infinity"
            y_v = np.nan
    else:
        vertical_vp_status += "; vertical_vp_missing_use_infinity"
        y_v = np.nan

    status_text = " | ".join([
        f"hough_lines_total={len(all_segments)}",
        vp_pos_info["status"],
        vp_neg_info["status"],
        vertical_vp_status,
    ])

    return {
        "edges": edges,
        "all_segments": all_segments,
        "groups": groups,
        "candidates": candidates,
        "vp_diag_pos": vp_pos_info["vp"],
        "vp_diag_neg": vp_neg_info["vp"],
        "vp_vertical": vp_v,
        "vp_diag_pos_inliers": vp_pos_info["inlier_count"],
        "vp_diag_neg_inliers": vp_neg_info["inlier_count"],
        "vp_vertical_inliers": vp_v_info["inlier_count"],
        "vp_diag_pos_ratio": vp_pos_info["inlier_ratio"],
        "vp_diag_neg_ratio": vp_neg_info["inlier_ratio"],
        "vp_vertical_ratio": vp_v_info["inlier_ratio"],
        "y_v": float(y_v) if np.isfinite(y_v) else np.nan,
        "vertical_vp_reliable": bool(vertical_vp_reliable),
        "x_ref": float(x_ref),
        "status_text": status_text,
        "num_lines_total": len(all_segments),
        "num_lines_diag_pos": len(groups["diag_pos"]),
        "num_lines_diag_neg": len(groups["diag_neg"]),
        "num_lines_vertical": len(groups["vertical"]),
        "num_lines_horizontal": len(groups["horizontal"]),
    }


def select_main_candidate(priors: Dict, candidate_eval: Dict[str, Dict], img_h: int) -> Tuple[Optional[Dict], str]:
    order = ["hough_y_voting", "longest_horizontal_line", "horizontal_weighted_median", "two_diagonal_vps"]
    reasons = []

    for src in order:
        c = priors["candidates"].get(src)
        ev = candidate_eval.get(src, {})
        valid_hor, hor_reason = is_valid_horizon_candidate(c, img_h)

        if not valid_hor:
            reasons.append(f"{src}: horizon_invalid({hor_reason})")
            continue

        plausible = bool(ev.get(f"{src}_plausible_height", False))
        if not plausible:
            est = ev.get(f"{src}_estimated_height_cm", np.nan)
            reasons.append(f"{src}: height_not_plausible({est})")
            continue

        return c, f"selected={src}; " + " | ".join(reasons)

    return None, "no_plausible_candidate; " + " | ".join(reasons)


def draw_infinite_line(img: np.ndarray, line: Optional[np.ndarray], color, thickness=4) -> None:
    if line is None:
        return
    h, w = img.shape[:2]
    a, b, c = line
    pts = []
    if abs(b) > 1e-9:
        pts.append((0, int(round(-(a * 0 + c) / b))))
        pts.append((w - 1, int(round(-(a * (w - 1) + c) / b))))
    if abs(a) > 1e-9:
        pts.append((int(round(-(b * 0 + c) / a)), 0))
        pts.append((int(round(-(b * (h - 1) + c) / a)), h - 1))
    valid = []
    for x, y in pts:
        if -w <= x <= 2 * w and -h <= y <= 2 * h:
            valid.append((x, y))
    if len(valid) >= 2:
        cv2.line(img, valid[0], valid[1], color, thickness, cv2.LINE_AA)


def draw_point_if_reasonable(img: np.ndarray, pt: Optional[np.ndarray], label: str, color) -> None:
    if pt is None:
        return

    h, w = img.shape[:2]
    x, y = float(pt[0]), float(pt[1])
    margin = 0.5 * max(w, h)

    if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
        return

    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    cv2.circle(img, (xi, yi), 10, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (xi + 12, yi - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def make_visualization(img_bgr: np.ndarray, row: pd.Series, priors: Dict, main_candidate: Optional[Dict], est_height_cm: float, out_path: Path) -> None:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    colors = {
        "diag_pos": (255, 120, 0),
        "diag_neg": (0, 180, 255),
        "vertical": (0, 220, 0),
        "horizontal": (200, 0, 200),
    }

    for group_name, segs in priors.get("groups", {}).items():
        segs_sorted = sorted(segs, key=lambda s: s["length"], reverse=True)[:100]
        for s in segs_sorted:
            p1 = (int(round(s["x1"])), int(round(s["y1"])))
            p2 = (int(round(s["x2"])), int(round(s["y2"])))
            cv2.line(vis, p1, p2, colors.get(group_name, (180, 180, 180)), 2, cv2.LINE_AA)

    cand = priors.get("candidates", {})
    draw_infinite_line(vis, cand.get("hough_y_voting", {}).get("line"), (0, 255, 180), thickness=4)
    draw_infinite_line(vis, cand.get("longest_horizontal_line", {}).get("line"), (0, 160, 255), thickness=3)
    draw_infinite_line(vis, cand.get("horizontal_weighted_median", {}).get("line"), (200, 0, 200), thickness=3)
    draw_infinite_line(vis, cand.get("two_diagonal_vps", {}).get("line"), (255, 80, 0), thickness=3)

    if main_candidate is not None:
        draw_infinite_line(vis, main_candidate.get("line"), (0, 255, 255), thickness=7)

    top_x = safe_float(row.get("top_x"))
    top_y = safe_float(row.get("top_y"))
    base_x = safe_float(row.get("base_x"))
    base_y = safe_float(row.get("base_y"))

    if np.all(np.isfinite([top_x, top_y, base_x, base_y])):
        cv2.circle(vis, (int(round(top_x)), int(round(top_y))), 16, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, "top", (int(round(top_x)) + 10, int(round(top_y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(base_x)), int(round(base_y))), 16, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(vis, "base", (int(round(base_x)) + 10, int(round(base_y)) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(vis, (int(round(top_x)), int(round(top_y))), (int(round(base_x)), int(round(base_y))), (255, 255, 255), 5, cv2.LINE_AA)

    draw_point_if_reasonable(vis, priors.get("vp_diag_pos"), "VP1", (255, 120, 0))
    draw_point_if_reasonable(vis, priors.get("vp_diag_neg"), "VP2", (0, 180, 255))
    draw_point_if_reasonable(vis, priors.get("vp_vertical"), "VVP", (0, 220, 0))

    main_source = "none" if main_candidate is None else main_candidate.get("source", "unknown")
    text_lines = [
        f"method={METHOD_NAME}",
        f"main_horizon={main_source}",
        f"lines={priors.get('num_lines_total', 0)}",
        f"hor={priors.get('num_lines_horizontal', 0)}, diag+={priors.get('num_lines_diag_pos', 0)}, diag-={priors.get('num_lines_diag_neg', 0)}",
        f"vertical_vp_reliable={priors.get('vertical_vp_reliable', False)}",
        f"est_height_cm={est_height_cm:.2f}" if np.isfinite(est_height_cm) else "est_height_cm=nan",
        "cyan=main, green=hough_vote, orange=longest, purple=weighted, blue=two_vps",
    ]

    y0 = 55
    for t in text_lines:
        cv2.putText(vis, t, (40, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis, t, (40, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 43

    max_side = 1800
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    imwrite_unicode(out_path, vis)


def init_record(row: pd.Series) -> Dict:
    rec = row.to_dict()

    base_cols = {
        f"{METHOD_NAME}_status": "unknown",
        f"{METHOD_NAME}_warning": "",
        f"{METHOD_NAME}_image_used": "",
        f"{METHOD_NAME}_num_lines_total": np.nan,
        f"{METHOD_NAME}_num_lines_diag_pos": np.nan,
        f"{METHOD_NAME}_num_lines_diag_neg": np.nan,
        f"{METHOD_NAME}_num_lines_vertical": np.nan,
        f"{METHOD_NAME}_num_lines_horizontal": np.nan,
        f"{METHOD_NAME}_vp_diag_pos_inliers": np.nan,
        f"{METHOD_NAME}_vp_diag_neg_inliers": np.nan,
        f"{METHOD_NAME}_vp_vertical_inliers": np.nan,
        f"{METHOD_NAME}_vp_diag_pos_ratio": np.nan,
        f"{METHOD_NAME}_vp_diag_neg_ratio": np.nan,
        f"{METHOD_NAME}_vp_vertical_ratio": np.nan,
        f"{METHOD_NAME}_vertical_vp_reliable": False,
        f"{METHOD_NAME}_selected_horizon_source": "",
        f"{METHOD_NAME}_selected_horizon_reason": "",
        f"{METHOD_NAME}_horizon_tilt_deg": np.nan,
        f"{METHOD_NAME}_height_mode": "",
        f"{METHOD_NAME}_x_ref_px": np.nan,
        f"{METHOD_NAME}_y_hor_px": np.nan,
        f"{METHOD_NAME}_y_v_px": np.nan,
        f"{METHOD_NAME}_estimated_height_cm": np.nan,
        f"{METHOD_NAME}_abs_error_cm": np.nan,
        f"{METHOD_NAME}_rel_error_percent": np.nan,
        f"{METHOD_NAME}_visual_path": "",
    }

    for k, v in base_cols.items():
        rec[k] = v

    for src in ["hough_y_voting", "two_diagonal_vps", "horizontal_weighted_median", "longest_horizontal_line"]:
        rec[f"{src}_candidate_reason"] = ""
        rec[f"{src}_valid_horizon"] = False
        rec[f"{src}_invalid_reason"] = ""
        rec[f"{src}_y_hor_px"] = np.nan
        rec[f"{src}_tilt_deg"] = np.nan
        rec[f"{src}_score"] = np.nan
        rec[f"{src}_estimated_height_cm"] = np.nan
        rec[f"{src}_height_mode"] = ""
        rec[f"{src}_height_status"] = ""
        rec[f"{src}_plausible_height"] = False
        rec[f"{src}_abs_error_cm"] = np.nan
        rec[f"{src}_rel_error_percent"] = np.nan

    return rec


def fill_prior_record(rec: Dict, priors: Dict) -> None:
    rec[f"{METHOD_NAME}_num_lines_total"] = priors["num_lines_total"]
    rec[f"{METHOD_NAME}_num_lines_diag_pos"] = priors["num_lines_diag_pos"]
    rec[f"{METHOD_NAME}_num_lines_diag_neg"] = priors["num_lines_diag_neg"]
    rec[f"{METHOD_NAME}_num_lines_vertical"] = priors["num_lines_vertical"]
    rec[f"{METHOD_NAME}_num_lines_horizontal"] = priors["num_lines_horizontal"]
    rec[f"{METHOD_NAME}_vp_diag_pos_inliers"] = priors["vp_diag_pos_inliers"]
    rec[f"{METHOD_NAME}_vp_diag_neg_inliers"] = priors["vp_diag_neg_inliers"]
    rec[f"{METHOD_NAME}_vp_vertical_inliers"] = priors["vp_vertical_inliers"]
    rec[f"{METHOD_NAME}_vp_diag_pos_ratio"] = priors["vp_diag_pos_ratio"]
    rec[f"{METHOD_NAME}_vp_diag_neg_ratio"] = priors["vp_diag_neg_ratio"]
    rec[f"{METHOD_NAME}_vp_vertical_ratio"] = priors["vp_vertical_ratio"]
    rec[f"{METHOD_NAME}_vertical_vp_reliable"] = priors["vertical_vp_reliable"]
    rec[f"{METHOD_NAME}_x_ref_px"] = priors["x_ref"]
    rec[f"{METHOD_NAME}_y_v_px"] = priors["y_v"]


def process_one_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cam_h_from_path, true_h_from_path = parse_camera_and_height_from_path(csv_path)

    try:
        rel_group_dir = csv_path.parent.relative_to(CSV_BY_GROUP_DIR)
    except Exception:
        rel_group_dir = Path(csv_path.parent.name)

    out_group_dir = OUT_GROUP_CSV_DIR / rel_group_dir
    out_group_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_group_dir / f"{METHOD_NAME}_height.csv"

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=str(rel_group_dir)):
        rec = init_record(row)

        img_path = resolve_image_path(row)
        rec[f"{METHOD_NAME}_image_used"] = str(img_path) if img_path is not None else ""

        yolo_status = str(row.get("status", "")).lower()
        if yolo_status != "ok":
            rec[f"{METHOD_NAME}_status"] = "skip_no_yolo_keypoints"
            results.append(rec)
            continue

        if img_path is None or not img_path.exists():
            rec[f"{METHOD_NAME}_status"] = "image_not_found"
            rec[f"{METHOD_NAME}_warning"] = str(img_path)
            results.append(rec)
            continue

        img = imread_unicode(img_path)
        if img is None:
            rec[f"{METHOD_NAME}_status"] = "image_read_failed"
            rec[f"{METHOD_NAME}_warning"] = str(img_path)
            results.append(rec)
            continue

        img_h, _ = img.shape[:2]
        kp_ok, kp_status = keypoints_are_valid(row, img_h)
        if not kp_ok:
            rec[f"{METHOD_NAME}_status"] = kp_status
            results.append(rec)
            continue

        H_cam_cm = safe_float(row.get("camera_setting"))
        if not np.isfinite(H_cam_cm):
            H_cam_cm = cam_h_from_path

        true_height_cm = safe_float(row.get("true_height_cm"))
        if not np.isfinite(true_height_cm):
            true_height_cm = true_h_from_path

        top_y = safe_float(row.get("top_y"))
        base_y = safe_float(row.get("base_y"))

        priors = estimate_geometric_priors(img, row)
        fill_prior_record(rec, priors)

        candidate_eval: Dict[str, Dict] = {}

        for source, candidate in priors["candidates"].items():
            ev = evaluate_candidate_height(
                candidate=candidate,
                H_cam_cm=H_cam_cm,
                true_height_cm=true_height_cm,
                y_top=top_y,
                y_bot=base_y,
                y_v=priors["y_v"],
                vertical_vp_reliable=priors["vertical_vp_reliable"],
                img_h=img_h,
            )
            candidate_eval[source] = ev
            for k, v in ev.items():
                rec[k] = v

        main_candidate, select_reason = select_main_candidate(priors, candidate_eval, img_h)
        rec[f"{METHOD_NAME}_selected_horizon_reason"] = select_reason

        if main_candidate is None:
            rec[f"{METHOD_NAME}_status"] = "invalid_horizon_or_height"
            rec[f"{METHOD_NAME}_warning"] = priors["status_text"] + " | " + select_reason
            main_est_h_cm = np.nan
        else:
            main_source = main_candidate["source"]
            main_est_h_cm = rec.get(f"{main_source}_estimated_height_cm", np.nan)

            rec[f"{METHOD_NAME}_selected_horizon_source"] = main_source
            rec[f"{METHOD_NAME}_horizon_tilt_deg"] = main_candidate.get("tilt_deg", np.nan)
            rec[f"{METHOD_NAME}_height_mode"] = rec.get(f"{main_source}_height_mode", "")
            rec[f"{METHOD_NAME}_y_hor_px"] = main_candidate.get("y_hor", np.nan)
            rec[f"{METHOD_NAME}_estimated_height_cm"] = main_est_h_cm

            if is_plausible_height(main_est_h_cm):
                rec[f"{METHOD_NAME}_status"] = "ok"
            else:
                rec[f"{METHOD_NAME}_status"] = "invalid_height"

            if rec[f"{METHOD_NAME}_status"] == "ok" and np.isfinite(main_est_h_cm) and np.isfinite(true_height_cm) and true_height_cm > 0:
                abs_err = abs(main_est_h_cm - true_height_cm)
                rel_err = abs_err / true_height_cm * 100.0
                rec[f"{METHOD_NAME}_abs_error_cm"] = abs_err
                rec[f"{METHOD_NAME}_rel_error_percent"] = rel_err

            rec[f"{METHOD_NAME}_warning"] = priors["status_text"] + " | " + select_reason

        if SAVE_VISUAL:
            image_name = get_image_name(row, img_path, idx)
            tag = rec[f"{METHOD_NAME}_status"]
            vis_path = OUT_VIS_DIR / rel_group_dir / f"{Path(image_name).stem}_{METHOD_NAME}_{tag}.jpg"

            try:
                make_visualization(img, row, priors, main_candidate, main_est_h_cm, vis_path)
                rec[f"{METHOD_NAME}_visual_path"] = str(vis_path)
            except Exception as e:
                rec[f"{METHOD_NAME}_warning"] += f" | visual_failed={repr(e)}"

        results.append(rec)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_df


def summarize_main(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_col = f"{METHOD_NAME}_status"
    est_col = f"{METHOD_NAME}_estimated_height_cm"
    abs_col = f"{METHOD_NAME}_abs_error_cm"
    rel_col = f"{METHOD_NAME}_rel_error_percent"

    vc = all_df[status_col].value_counts(dropna=False)
    status_count = pd.DataFrame({"status": vc.index.astype(str), "count": vc.values})

    valid = all_df[(all_df[status_col] == "ok") & np.isfinite(all_df[est_col]) & np.isfinite(all_df[abs_col]) & np.isfinite(all_df[rel_col])].copy()

    if len(valid) == 0:
        overall = pd.DataFrame([{
            "method": METHOD_NAME,
            "total_rows": int(len(all_df)),
            "valid_count": 0,
            "MRE_percent": np.nan,
            "RMSE_cm": np.nan,
            "MAE_cm": np.nan,
            "median_RE_percent": np.nan,
            "RE_under_5_count": 0,
            "RE_under_10_count": 0,
            "RE_under_20_count": 0,
        }])
        by_group = pd.DataFrame()
        return overall, by_group, status_count

    overall = pd.DataFrame([{
        "method": METHOD_NAME,
        "total_rows": int(len(all_df)),
        "valid_count": int(len(valid)),
        "MRE_percent": float(valid[rel_col].mean()),
        "RMSE_cm": float(np.sqrt(np.mean((valid[est_col] - valid["true_height_cm"]) ** 2))),
        "MAE_cm": float(valid[abs_col].mean()),
        "median_RE_percent": float(valid[rel_col].median()),
        "RE_under_5_count": int((valid[rel_col] <= 5).sum()),
        "RE_under_10_count": int((valid[rel_col] <= 10).sum()),
        "RE_under_20_count": int((valid[rel_col] <= 20).sum()),
    }])

    by_group = (
        valid.groupby(["camera_setting", "true_height_cm"], dropna=False)
        .apply(lambda g: pd.Series({
            "method": METHOD_NAME,
            "valid_count": int(len(g)),
            "MRE_percent": float(g[rel_col].mean()),
            "RMSE_cm": float(np.sqrt(np.mean((g[est_col] - g["true_height_cm"]) ** 2))),
            "MAE_cm": float(g[abs_col].mean()),
            "median_RE_percent": float(g[rel_col].median()),
            "RE_under_5_count": int((g[rel_col] <= 5).sum()),
            "RE_under_10_count": int((g[rel_col] <= 10).sum()),
            "RE_under_20_count": int((g[rel_col] <= 20).sum()),
        }))
        .reset_index()
    )

    return overall, by_group, status_count


def summarize_candidates(all_df: pd.DataFrame) -> pd.DataFrame:
    sources = ["hough_y_voting", "two_diagonal_vps", "horizontal_weighted_median", "longest_horizontal_line"]
    rows = []

    for src in sources:
        est_col = f"{src}_estimated_height_cm"
        abs_col = f"{src}_abs_error_cm"
        rel_col = f"{src}_rel_error_percent"
        plausible_col = f"{src}_plausible_height"
        valid_col = f"{src}_valid_horizon"

        valid = all_df[
            (all_df[valid_col] == True)
            & (all_df[plausible_col] == True)
            & np.isfinite(all_df[est_col])
            & np.isfinite(all_df[rel_col])
        ].copy()

        rows.append({
            "method": METHOD_NAME,
            "horizon_candidate": src,
            "valid_horizon_count": int((all_df[valid_col] == True).sum()) if valid_col in all_df.columns else 0,
            "plausible_height_count": int((all_df[plausible_col] == True).sum()) if plausible_col in all_df.columns else 0,
            "metric_valid_count": int(len(valid)),
            "MRE_percent": float(valid[rel_col].mean()) if len(valid) else np.nan,
            "RMSE_cm": float(np.sqrt(np.mean((valid[est_col] - valid["true_height_cm"]) ** 2))) if len(valid) else np.nan,
            "MAE_cm": float(valid[abs_col].mean()) if len(valid) else np.nan,
            "median_RE_percent": float(valid[rel_col].median()) if len(valid) else np.nan,
            "RE_under_5_count": int((valid[rel_col] <= 5).sum()) if len(valid) else 0,
            "RE_under_10_count": int((valid[rel_col] <= 10).sum()) if len(valid) else 0,
            "RE_under_20_count": int((valid[rel_col] <= 20).sum()) if len(valid) else 0,
        })

    return pd.DataFrame(rows)


def export_best_cases(all_df: pd.DataFrame) -> None:
    status_col = f"{METHOD_NAME}_status"
    rel_col = f"{METHOD_NAME}_rel_error_percent"

    valid = all_df[(all_df[status_col] == "ok") & np.isfinite(all_df[rel_col])].copy()
    if len(valid) == 0:
        return

    best = valid.sort_values(rel_col, ascending=True).head(30)
    best.to_csv(OUTPUT_DIR / f"best_cases_top30_{METHOD_NAME}.csv", index=False, encoding="utf-8-sig")

    best_under_10 = valid[valid[rel_col] <= 10].sort_values(rel_col, ascending=True)
    best_under_10.to_csv(OUTPUT_DIR / f"best_cases_under10percent_{METHOD_NAME}.csv", index=False, encoding="utf-8-sig")


def export_horizon_diagnostics(all_df: pd.DataFrame) -> None:
    keep_cols = [
        "image_path", "image_name", "camera_setting", "true_height_cm", "status",
        f"{METHOD_NAME}_status", f"{METHOD_NAME}_selected_horizon_source",
        f"{METHOD_NAME}_estimated_height_cm", f"{METHOD_NAME}_rel_error_percent",
        f"{METHOD_NAME}_num_lines_total", f"{METHOD_NAME}_num_lines_diag_pos", f"{METHOD_NAME}_num_lines_diag_neg",
        f"{METHOD_NAME}_num_lines_vertical", f"{METHOD_NAME}_num_lines_horizontal",
        f"{METHOD_NAME}_vp_diag_pos_inliers", f"{METHOD_NAME}_vp_diag_neg_inliers", f"{METHOD_NAME}_vp_vertical_inliers",
        f"{METHOD_NAME}_vertical_vp_reliable", f"{METHOD_NAME}_selected_horizon_reason",
        "hough_y_voting_valid_horizon", "hough_y_voting_invalid_reason", "hough_y_voting_y_hor_px",
        "hough_y_voting_tilt_deg", "hough_y_voting_estimated_height_cm", "hough_y_voting_rel_error_percent",
        "two_diagonal_vps_valid_horizon", "two_diagonal_vps_invalid_reason", "two_diagonal_vps_y_hor_px",
        "two_diagonal_vps_tilt_deg", "two_diagonal_vps_estimated_height_cm", "two_diagonal_vps_rel_error_percent",
        "horizontal_weighted_median_valid_horizon", "horizontal_weighted_median_invalid_reason", "horizontal_weighted_median_y_hor_px",
        "horizontal_weighted_median_tilt_deg", "horizontal_weighted_median_estimated_height_cm", "horizontal_weighted_median_rel_error_percent",
        "longest_horizontal_line_valid_horizon", "longest_horizontal_line_invalid_reason", "longest_horizontal_line_y_hor_px",
        "longest_horizontal_line_tilt_deg", "longest_horizontal_line_estimated_height_cm", "longest_horizontal_line_rel_error_percent",
        f"{METHOD_NAME}_visual_path",
    ]

    existing = [c for c in keep_cols if c in all_df.columns]
    all_df[existing].to_csv(OUT_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GROUP_CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_BY_GROUP_DIR.rglob("best_one_keypoints.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 best_one_keypoints.csv：{CSV_BY_GROUP_DIR}")

    print("=" * 80)
    print(f"[INFO] method                       : {METHOD_NAME}")
    print(f"[INFO] input csv_by_group            : {CSV_BY_GROUP_DIR}")
    print(f"[INFO] found csv files               : {len(csv_files)}")
    print(f"[INFO] output dir                    : {OUTPUT_DIR}")
    print(f"[INFO] no external model             : True")
    print(f"[INFO] OpenCV HoughLinesP threshold  : {HOUGH_THRESHOLD}")
    print(f"[INFO] Hough min line length         : {HOUGH_MIN_LINE_LENGTH}")
    print(f"[INFO] erase bbox before Hough       : {ERASE_PLANT_BBOX_BEFORE_HOUGH}")
    print("=" * 80)

    all_parts = []
    for csv_path in csv_files:
        part = process_one_csv(csv_path)
        all_parts.append(part)

    all_df = pd.concat(all_parts, ignore_index=True)
    all_df.to_csv(OUT_ALL_CSV, index=False, encoding="utf-8-sig")

    overall, by_group, status_count = summarize_main(all_df)
    candidate_summary = summarize_candidates(all_df)

    overall.to_csv(OUT_OVERALL_CSV, index=False, encoding="utf-8-sig")
    by_group.to_csv(OUT_GROUP_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    status_count.to_csv(OUT_STATUS_CSV, index=False, encoding="utf-8-sig")
    candidate_summary.to_csv(OUT_CANDIDATE_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    export_best_cases(all_df)
    export_horizon_diagnostics(all_df)

    print("\n[DONE] Hough Voting + RANSAC 几何线索提取与高度估计完成")
    print(f"[OUT] all csv             : {OUT_ALL_CSV}")
    print(f"[OUT] overall summary     : {OUT_OVERALL_CSV}")
    print(f"[OUT] group summary       : {OUT_GROUP_SUMMARY_CSV}")
    print(f"[OUT] status count        : {OUT_STATUS_CSV}")
    print(f"[OUT] candidate summary   : {OUT_CANDIDATE_SUMMARY_CSV}")
    print(f"[OUT] diagnostics         : {OUT_DIAGNOSTICS_CSV}")
    print(f"[OUT] best top30          : {OUTPUT_DIR / f'best_cases_top30_{METHOD_NAME}.csv'}")
    print(f"[OUT] best under10        : {OUTPUT_DIR / f'best_cases_under10percent_{METHOD_NAME}.csv'}")
    print(f"[OUT] visual dir          : {OUT_VIS_DIR}")

    print("\n[OVERALL]")
    print(overall.to_string(index=False))

    print("\n[CANDIDATE SUMMARY]")
    print(candidate_summary.to_string(index=False))

    print("\n[STATUS COUNT]")
    print(status_count.to_string(index=False))


if __name__ == "__main__":
    main()
