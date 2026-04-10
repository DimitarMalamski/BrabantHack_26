# detection_by_shadow_baseline.py

import os
import json
import glob
import math
import warnings

import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
TRAIN_DIR = "/data/train_data/train_data"
TEST_DIR = "/data/test_data/test_data"
SAMPLE_SUBMISSION_PATH = "/data/submission_example.csv"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_image_paths(folder: str):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def clip_or_default(value, default=0.0):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return default
    return float(value)


def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


# =========================
# LABEL TRANSFORMS
# =========================
def extract_targets_from_json(label_data: dict, image_width: int):
    bbox = label_data["bbox"]

    xmin = float(bbox["top_left"][0])
    ymin = float(bbox["top_left"][1])
    xmax = float(bbox["top_right"][0])
    ymax = float(bbox["bottom_left"][1])

    width = xmax - xmin
    height = ymax - ymin
    y_bottom = ymax

    # Side:
    # 0 = left of frame
    # 1 = right of frame
    # If whole box is left of image => xmax <= 0
    # If whole box is right of image => xmin >= image_width
    if xmax <= 0:
        side = 0
        x_offset = -xmax
    elif xmin >= image_width:
        side = 1
        x_offset = xmin - image_width
    else:
        # Fallback if something unexpected happens
        center_x = (xmin + xmax) / 2.0
        side = 0 if center_x < image_width / 2 else 1
        x_offset = 0.0

    direction = int(label_data.get("walking_into_frame_bool", -1))

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": width,
        "height": height,
        "y_bottom": y_bottom,
        "side": side,
        "x_offset": x_offset,
        "direction": direction,
    }


# =========================
# SHADOW FEATURE EXTRACTION
# =========================
def compute_shadow_mask(img: np.ndarray):
    """
    Very simple synthetic-friendly shadow extraction:
    - grayscale
    - blur
    - inverse threshold for dark regions
    - morphology cleanup
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold on dark regions
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return gray, mask


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_pca_features(contour: np.ndarray):
    pts = contour.reshape(-1, 2).astype(np.float32)

    if len(pts) < 5:
        return {
            "pca_cx": 0.0,
            "pca_cy": 0.0,
            "major_dx": 1.0,
            "major_dy": 0.0,
            "major_len": 0.0,
            "minor_len": 0.0,
            "angle_deg": 0.0,
        }

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=np.empty((0)))
    cx, cy = mean[0]

    major = eigenvectors[0]
    minor = eigenvectors[1]

    major_dx, major_dy = float(major[0]), float(major[1])
    major_len = float(np.sqrt(eigenvalues[0][0])) if eigenvalues.shape[0] > 0 else 0.0
    minor_len = float(np.sqrt(eigenvalues[1][0])) if eigenvalues.shape[0] > 1 else 0.0
    angle_deg = float(np.degrees(np.arctan2(major_dy, major_dx)))

    return {
        "pca_cx": float(cx),
        "pca_cy": float(cy),
        "major_dx": major_dx,
        "major_dy": major_dy,
        "major_len": major_len,
        "minor_len": minor_len,
        "angle_deg": angle_deg,
    }


def line_border_intersections(cx, cy, dx, dy, width, height):
    """
    Intersections of a line p(t) = c + t d with image borders.
    Returns y at left border and y at right border, plus x at top/bottom.
    """
    result = {
        "y_at_left": -9999.0,
        "y_at_right": -9999.0,
        "x_at_top": -9999.0,
        "x_at_bottom": -9999.0,
    }

    if abs(dx) > 1e-6:
        t_left = (0 - cx) / dx
        t_right = ((width - 1) - cx) / dx
        result["y_at_left"] = cy + t_left * dy
        result["y_at_right"] = cy + t_right * dy

    if abs(dy) > 1e-6:
        t_top = (0 - cy) / dy
        t_bottom = ((height - 1) - cy) / dy
        result["x_at_top"] = cx + t_top * dx
        result["x_at_bottom"] = cx + t_bottom * dx

    return result


def extract_shadow_features(img: np.ndarray):
    h, w = img.shape[:2]
    gray, mask = compute_shadow_mask(img)
    contour = largest_contour(mask)

    # defaults in case contour extraction fails
    features = {
        "img_w": float(w),
        "img_h": float(h),
        "shadow_found": 0.0,
        "mask_ratio": 0.0,
        "contour_area": 0.0,
        "contour_perimeter": 0.0,
        "bbox_x": 0.0,
        "bbox_y": 0.0,
        "bbox_w": 0.0,
        "bbox_h": 0.0,
        "bbox_aspect": 0.0,
        "bbox_cx": 0.0,
        "bbox_cy": 0.0,
        "leftmost_x": 0.0,
        "leftmost_y": 0.0,
        "rightmost_x": 0.0,
        "rightmost_y": 0.0,
        "topmost_x": 0.0,
        "topmost_y": 0.0,
        "bottommost_x": 0.0,
        "bottommost_y": 0.0,
        "dist_left": float(w),
        "dist_right": float(w),
        "touches_left": 0.0,
        "touches_right": 0.0,
        "mean_gray_shadow": 0.0,
        "std_gray_shadow": 0.0,
        "pca_cx": 0.0,
        "pca_cy": 0.0,
        "major_dx": 1.0,
        "major_dy": 0.0,
        "major_len": 0.0,
        "minor_len": 0.0,
        "angle_deg": 0.0,
        "y_at_left": -9999.0,
        "y_at_right": -9999.0,
        "x_at_top": -9999.0,
        "x_at_bottom": -9999.0,
    }

    features["mask_ratio"] = float(np.count_nonzero(mask)) / float(mask.size)

    if contour is None or len(contour) < 5:
        return features

    features["shadow_found"] = 1.0

    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    x, y, bw, bh = cv2.boundingRect(contour)

    pts = contour.reshape(-1, 2)
    leftmost = pts[np.argmin(pts[:, 0])]
    rightmost = pts[np.argmax(pts[:, 0])]
    topmost = pts[np.argmin(pts[:, 1])]
    bottommost = pts[np.argmax(pts[:, 1])]

    shadow_pixels = gray[mask > 0]
    mean_gray = float(np.mean(shadow_pixels)) if len(shadow_pixels) > 0 else 0.0
    std_gray = float(np.std(shadow_pixels)) if len(shadow_pixels) > 0 else 0.0

    pca = fit_pca_features(contour)
    line_feats = line_border_intersections(
        pca["pca_cx"], pca["pca_cy"], pca["major_dx"], pca["major_dy"], w, h
    )

    features.update({
        "contour_area": float(area),
        "contour_perimeter": float(peri),
        "bbox_x": float(x),
        "bbox_y": float(y),
        "bbox_w": float(bw),
        "bbox_h": float(bh),
        "bbox_aspect": safe_div(bw, bh, 0.0),
        "bbox_cx": float(x + bw / 2.0),
        "bbox_cy": float(y + bh / 2.0),
        "leftmost_x": float(leftmost[0]),
        "leftmost_y": float(leftmost[1]),
        "rightmost_x": float(rightmost[0]),
        "rightmost_y": float(rightmost[1]),
        "topmost_x": float(topmost[0]),
        "topmost_y": float(topmost[1]),
        "bottommost_x": float(bottommost[0]),
        "bottommost_y": float(bottommost[1]),
        "dist_left": float(leftmost[0]),
        "dist_right": float((w - 1) - rightmost[0]),
        "touches_left": 1.0 if leftmost[0] <= 2 else 0.0,
        "touches_right": 1.0 if rightmost[0] >= w - 3 else 0.0,
        "mean_gray_shadow": mean_gray,
        "std_gray_shadow": std_gray,
    })

    features.update(pca)
    features.update(line_feats)

    # normalized features often help tree models
    normalized = {}
    for key in [
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_cx", "bbox_cy",
        "leftmost_x", "leftmost_y", "rightmost_x", "rightmost_y",
        "topmost_x", "topmost_y", "bottommost_x", "bottommost_y",
        "dist_left", "dist_right", "pca_cx", "pca_cy",
        "y_at_left", "y_at_right", "x_at_top", "x_at_bottom"
    ]:
        if "x" in key or "left" in key or "right" in key:
            normalized[f"{key}_norm"] = features[key] / max(w, 1)
        else:
            normalized[f"{key}_norm"] = features[key] / max(h, 1)

    normalized["contour_area_norm"] = features["contour_area"] / float(w * h)
    normalized["contour_perimeter_norm"] = features["contour_perimeter"] / float(w + h)
    normalized["major_len_norm"] = features["major_len"] / max(w, h)
    normalized["minor_len_norm"] = features["minor_len"] / max(w, h)

    features.update(normalized)
    return features


# =========================
# DATASET BUILDING
# =========================
def build_train_dataframe(train_dir: str):
    rows = []
    image_paths = get_image_paths(train_dir)

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(train_dir, f"{stem}.json")
        if not os.path.exists(json_path):
            continue

        img = load_image(img_path)
        h, w = img.shape[:2]

        feats = extract_shadow_features(img)
        label = extract_targets_from_json(load_json(json_path), image_width=w)

        row = {"id": stem}
        row.update(feats)
        row.update(label)
        rows.append(row)

    return pd.DataFrame(rows)


def build_test_dataframe(test_dir: str):
    rows = []
    image_paths = get_image_paths(test_dir)

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = load_image(img_path)
        feats = extract_shadow_features(img)

        row = {"id": stem}
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# IOU EVAL
# =========================
def compute_iou(box_a, box_b):
    axmin, aymin, axmax, aymax = box_a
    bxmin, bymin, bxmax, bymax = box_b

    inter_xmin = max(axmin, bxmin)
    inter_ymin = max(aymin, bymin)
    inter_xmax = min(axmax, bxmax)
    inter_ymax = min(aymax, bymax)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area_a = max(0.0, axmax - axmin) * max(0.0, aymax - aymin)
    area_b = max(0.0, bxmax - bxmin) * max(0.0, bymax - bymin)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# =========================
# MODEL TRAINING
# =========================
def train_models(train_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns if c not in {
        "id",
        "xmin", "ymin", "xmax", "ymax",
        "width", "height", "y_bottom", "side", "x_offset", "direction"
    }]

    X = train_df[feature_cols].fillna(0.0)

    y_side = train_df["side"]
    y_bottom = train_df["y_bottom"]
    y_height = train_df["height"]
    y_width = train_df["width"]
    y_xoffset = train_df["x_offset"]

    side_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    reg_bottom = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    reg_height = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    reg_width = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    reg_xoffset = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    side_clf.fit(X, y_side)
    reg_bottom.fit(X, y_bottom)
    reg_height.fit(X, y_height)
    reg_width.fit(X, y_width)
    reg_xoffset.fit(X, y_xoffset)

    models = {
        "feature_cols": feature_cols,
        "side_clf": side_clf,
        "reg_bottom": reg_bottom,
        "reg_height": reg_height,
        "reg_width": reg_width,
        "reg_xoffset": reg_xoffset,
    }
    return models


def validate_models(train_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns if c not in {
        "id",
        "xmin", "ymin", "xmax", "ymax",
        "width", "height", "y_bottom", "side", "x_offset", "direction"
    }]

    train_part, valid_part = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_STATE
    )

    models = train_models(train_part)
    preds = predict_dataframe(models, valid_part)

    ious = []
    for _, row in preds.iterrows():
        gt = valid_part[valid_part["id"] == row["id"]].iloc[0]
        pred_box = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
        gt_box = [gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"]]
        ious.append(compute_iou(pred_box, gt_box))

    print(f"Validation mean IoU: {np.mean(ious):.4f}")
    return models


# =========================
# PREDICTION + RECONSTRUCTION
# =========================
def reconstruct_bbox(side, y_bottom, height, width, x_offset, image_width):
    side = int(round(side))
    y_bottom = float(y_bottom)
    height = max(1.0, float(height))
    width = max(1.0, float(width))
    x_offset = max(0.0, float(x_offset))

    if side == 0:
        xmax = -x_offset
        xmin = xmax - width
    else:
        xmin = image_width + x_offset
        xmax = xmin + width

    ymax = y_bottom
    ymin = ymax - height

    return xmin, ymin, xmax, ymax


def predict_dataframe(models, df: pd.DataFrame):
    X = df[models["feature_cols"]].fillna(0.0)

    pred_side = models["side_clf"].predict(X)
    pred_bottom = models["reg_bottom"].predict(X)
    pred_height = models["reg_height"].predict(X)
    pred_width = models["reg_width"].predict(X)
    pred_xoffset = models["reg_xoffset"].predict(X)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        image_width = row["img_w"]

        xmin, ymin, xmax, ymax = reconstruct_bbox(
            side=pred_side[i],
            y_bottom=pred_bottom[i],
            height=pred_height[i],
            width=pred_width[i],
            x_offset=pred_xoffset[i],
            image_width=image_width,
        )

        rows.append({
            "id": row["id"],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "direction": -1,  # safe baseline
        })

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    print("Building train dataframe...")
    train_df = build_train_dataframe(TRAIN_DIR)
    print(f"Train rows: {len(train_df)}")

    print("Validating baseline...")
    models = validate_models(train_df)

    print("Retraining on full training set...")
    models = train_models(train_df)

    print("Building test dataframe...")
    test_df = build_test_dataframe(TEST_DIR)
    print(f"Test rows: {len(test_df)}")

    print("Predicting test set...")
    submission = predict_dataframe(models, test_df)

    # match sample submission order if possible
    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        if "id" in sample.columns:
            submission = sample[["id"]].merge(submission, on="id", how="left")

    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(submission.head())


if __name__ == "__main__":
    main()