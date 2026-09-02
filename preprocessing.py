"""
Core Machine Learning and Image Preprocessing Module
Used by BOTH the Tkinter Desktop Application and the Streamlit Web Application.
Preserves the Scikit-Learn Digits Dataset and KNeighborsClassifier pipeline.
"""

import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Global cache for dataset and model
_DATASET = None
_MODEL = None
_EVAL_RESULTS = None

# Representative sample indices per digit class in load_digits()
REPRESENTATIVE_INDICES = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 15,  # Clean canonical sample for 5
    6: 6,
    7: 7,
    8: 8,
    9: 9
}


def load_digits_dataset():
    """
    Loads and caches the standard Scikit-Learn 8x8 handwritten digits dataset.
    """
    global _DATASET
    if _DATASET is None:
        _DATASET = load_digits()
    return _DATASET


def get_knn_model(n_neighbors=3):
    """
    Trains and caches the KNeighborsClassifier on the digits dataset.
    k=3 is optimal based on cross-validation performance.
    """
    global _MODEL
    if _MODEL is None:
        dataset = load_digits_dataset()
        _MODEL = KNeighborsClassifier(n_neighbors=n_neighbors)
        _MODEL.fit(dataset.data, dataset.target)
    return _MODEL


def evaluate_k_values(k_list=(1, 3, 5, 7), n_splits=5):
    """
    Performs Stratified K-Fold Cross-Validation across candidate k values
    to measure real, empirical classification performance.
    """
    global _EVAL_RESULTS
    if _EVAL_RESULTS is not None:
        return _EVAL_RESULTS

    dataset = load_digits_dataset()
    X, y = dataset.data, dataset.target
    results = {}

    for k in k_list:
        clf = KNeighborsClassifier(n_neighbors=k)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)
        results[k] = {
            "mean_accuracy": float(scores.mean() * 100.0),
            "std_accuracy": float(scores.std() * 100.0),
            "fold_scores": [float(s * 100.0) for s in scores]
        }

    # Also calculate a held-out 80/20 train/test split score for k=3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf_3 = KNeighborsClassifier(n_neighbors=3)
    clf_3.fit(X_train, y_train)
    test_acc_3 = float(clf_3.score(X_test, y_test) * 100.0)

    _EVAL_RESULTS = {
        "k_eval": results,
        "test_accuracy_k3": test_acc_3,
        "selected_k": 3
    }
    return _EVAL_RESULTS


def get_real_dataset_sample(digit_class, sample_offset=0):
    """
    Retrieves a real 8x8 handwritten digit sample from load_digits()
    for the specified digit class (0–9).
    """
    dataset = load_digits_dataset()
    
    if sample_offset == 0 and digit_class in REPRESENTATIVE_INDICES:
        selected_idx = REPRESENTATIVE_INDICES[digit_class]
    else:
        indices = np.where(dataset.target == digit_class)[0]
        if len(indices) == 0:
            return None
        selected_idx = int(indices[sample_offset % len(indices)])
    
    raw_8x8 = dataset.images[selected_idx]
    raw_flat = dataset.data[selected_idx]
    target_label = int(dataset.target[selected_idx])
    
    # Run through the KNN model
    model = get_knn_model()
    flat_reshaped = raw_flat.reshape(1, -1)
    prediction = int(model.predict(flat_reshaped)[0])
    probabilities = model.predict_proba(flat_reshaped)[0]
    confidence = float(np.max(probabilities) * 100.0)
    
    # Convert 8x8 numpy array to PIL Image for visualization
    vis_8x8 = Image.fromarray((raw_8x8 / 16.0 * 255.0).astype(np.uint8), mode="L")
    
    return {
        "digit_class": digit_class,
        "sample_index": int(selected_idx),
        "target_label": target_label,
        "raw_8x8": raw_8x8,
        "flat_array": raw_flat,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "vis_image": vis_8x8
    }


def preprocess_digit(image, model=None, pad_factor=1.28):
    """
    Shared robust image preprocessing and prediction pipeline:
    1. Input normalization (PIL Image or NumPy RGBA/RGB/Grayscale).
    2. Grayscale conversion.
    3. Auto-detection of canvas polarity (inverts light bg with dark strokes).
    4. Blank canvas detection.
    5. Bounding box extraction of non-zero foreground strokes.
    6. Aspect-ratio preserving square padding (margin factor ~1.28).
    7. Geometric centering.
    8. Anti-aliased downsampling to 8x8.
    9. Pixel scaling to [0, 16] range matching sklearn Digits distribution.
    10. Flattening to 64 features.
    11. KNeighborsClassifier prediction & probability extraction.
    """
    if model is None:
        model = get_knn_model()

    # 1. Convert input to PIL Image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:
            # RGBA from canvas
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        pil_img = image.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # 2. Grayscale conversion
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # 3. Canvas polarity check:
    # If background is bright (e.g. white canvas with black ink), invert so
    # background is dark (0) and digit strokes are bright (0-255).
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    is_inverted = False
    if np.mean(border_pixels) > 127.0:
        gray = ImageOps.invert(gray)
        arr = 255.0 - arr
        is_inverted = True

    # 4. Blank canvas check (no significant stroke detected)
    stroke_mask = arr > 30.0
    if np.max(arr) < 30.0 or np.sum(stroke_mask) < 25:
        return None, "empty"

    # 5. Extract bounding box of foreground strokes
    coords = np.argwhere(stroke_mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = gray.crop((x0, y0, x1, y1))
    w, h = cropped.size

    # 6. Pad to square preserving aspect ratio & geometry
    max_side = max(w, h)
    padded_size = max(int(max_side * pad_factor), 10)
    padded = Image.new("L", (padded_size, padded_size), 0)
    offset_x = (padded_size - w) // 2
    offset_y = (padded_size - h) // 2
    padded.paste(cropped, (offset_x, offset_y))

    # 7. Downscale to 8x8 using Area/Box or Bilinear
    small = padded.resize((8, 8), Image.Resampling.BOX)
    small_arr = np.array(small, dtype=np.float32)

    # 8. Normalize and scale to [0, 16] range matching load_digits()
    max_val = np.max(small_arr)
    if max_val > 0:
        scaled_8x8 = (small_arr / max_val) * 16.0
    else:
        scaled_8x8 = np.zeros((8, 8), dtype=np.float32)

    # 9. Flatten to 64 features
    flat_array = scaled_8x8.reshape(1, -1)

    # 10. Model Prediction & Confidence Analysis
    prediction = int(model.predict(flat_array)[0])
    probabilities = model.predict_proba(flat_array)[0]
    confidence = float(np.max(probabilities) * 100.0)

    # Calculate top-3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {"digit": int(idx), "probability": float(probabilities[idx] * 100.0)}
        for idx in top_indices
    ]

    is_low_confidence = bool(confidence < 50.0)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
        "is_low_confidence": is_low_confidence,
        "scaled_8x8": scaled_8x8,
        "flat_array": flat_array[0],
        "cropped_image": cropped,
        "centered_image": padded,
        "small_image": small,
        "is_inverted": is_inverted
    }, "success"
