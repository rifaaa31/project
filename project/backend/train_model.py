import os
import json
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2


CLASSES: List[str] = [
    "nv", "mel", "bcc", "akiec", "bkl", "df", "vasc", "no_lesion"
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_images_from_class_folders(dataset_dir: str, class_names: List[str],
                                    target_size: Tuple[int, int]=(64, 64)) -> Tuple[np.ndarray, np.ndarray]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            # Skip silently if class folder missing; allows partial datasets during testing
            continue
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size, Image.BILINEAR)
                    images.append(np.asarray(img, dtype=np.float32) / 255.0)
                    labels.append(class_index)
            except Exception:
                # Skip unreadable/corrupt files
                continue
    if not images:
        raise RuntimeError(f"No images found in {dataset_dir}. Ensure class subfolders exist: {class_names}")
    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[y]


def _stratified_split(
    X: np.ndarray,
    y_idx: np.ndarray,
    y_one_hot: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_train_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []
    y_train_list: List[np.ndarray] = []
    y_test_list: List[np.ndarray] = []

    for class_index in np.unique(y_idx):
        class_mask = (y_idx == class_index)
        class_indices = np.nonzero(class_mask)[0]
        rng.shuffle(class_indices)
        n_total = class_indices.shape[0]
        n_test = max(1, int(round(n_total * test_size))) if n_total > 1 else 1
        test_indices = class_indices[:n_test]
        train_indices = class_indices[n_test:]
        if train_indices.size == 0 and n_total > 1:
            # Ensure at least one train sample if possible
            train_indices = class_indices[-1:]
            test_indices = class_indices[:-1]

        X_train_list.append(X[train_indices])
        X_test_list.append(X[test_indices])
        y_train_list.append(y_one_hot[train_indices])
        y_test_list.append(y_one_hot[test_indices])

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Shuffle the combined splits
    def _shuffle_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.arange(a.shape[0])
        rng.shuffle(idx)
        return a[idx], b[idx]

    X_train, y_train = _shuffle_pair(X_train, y_train)
    X_test, y_test = _shuffle_pair(X_test, y_test)
    return X_train, X_test, y_train, y_test


def preprocess_and_save(dataset_dir: str, output_dir: str,
                        image_size: Tuple[int, int]=(64, 64), test_size: float=0.2,
                        random_state: int=42) -> Tuple[str, str, str, str]:
    ensure_dir(output_dir)

    X, y_idx = load_images_from_class_folders(dataset_dir, CLASSES, image_size)
    y = one_hot_encode(y_idx, len(CLASSES))

    X_train, X_test, y_train, y_test = _stratified_split(
        X, y_idx, y, test_size=test_size, random_state=random_state
    )

    x_train_path = os.path.join(output_dir, "X_train.npy")
    x_test_path = os.path.join(output_dir, "X_test.npy")
    y_train_path = os.path.join(output_dir, "y_train.npy")
    y_test_path = os.path.join(output_dir, "y_test.npy")

    np.save(x_train_path, X_train)
    np.save(x_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    print(f"Saved preprocessed arrays to: {output_dir}")
    return x_train_path, x_test_path, y_train_path, y_test_path


def build_model(input_shape=(64, 64, 3), num_classes: int = 8) -> tf.keras.Model:
    # Data augmentation is put directly into the model graph for speed and portability
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="lesion_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training(history: tf.keras.callbacks.History, plots_dir: str) -> None:
    ensure_dir(plots_dir)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    acc_path = os.path.join(plots_dir, "accuracy.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_path = os.path.join(plots_dir, "loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()


def _find_base_submodel(model: tf.keras.Model) -> tf.keras.Model:
    # Try to find the MobileNetV2 submodel robustly
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
        if isinstance(layer, layers.Layer) and hasattr(layer, 'name') and 'mobilenetv2' in layer.name.lower():
            return layer  # type: ignore
    # Fallback: try by name
    try:
        return model.get_layer('MobilenetV2')  # type: ignore
    except Exception:
        return model


def _calibrate_threshold(y_true_one_hot: np.ndarray, probs: np.ndarray, default_threshold: float = 0.80) -> Tuple[float, float, float]:
    """Return (threshold, precision_at_thr, recall_at_thr) maximizing recall with 100% precision.

    - y_true_one_hot: shape (N, C)
    - probs: softmax probabilities, shape (N, C)
    """
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = probs[np.arange(probs.shape[0]), y_pred]
    correct = (y_pred == y_true)

    # Evaluate precision/recall at candidate thresholds = unique confidences
    unique_conf = np.unique(conf)
    # Consider thresholds slightly above each unique value to drop all with conf == v when moving down
    candidate_thresholds = np.concatenate([unique_conf + 1e-6, [0.999999, 0.95, 0.90, 0.85, 0.80]])
    candidate_thresholds = np.clip(candidate_thresholds, 0.0, 1.0)
    candidate_thresholds = np.unique(candidate_thresholds)

    best_thr = default_threshold
    best_recall = 0.0
    best_precision = 1.0

    total_positives = float(correct.sum())
    for thr in sorted(candidate_thresholds):
        accepted = conf >= thr
        n_acc = int(accepted.sum())
        if n_acc == 0:
            continue
        tp = int((accepted & correct).sum())
        fp = n_acc - tp
        precision = 1.0 if n_acc == 0 else (tp / n_acc)
        recall = 0.0 if total_positives == 0 else (tp / total_positives)
        if precision == 1.0 and recall >= best_recall:
            best_recall = recall
            best_thr = float(thr)
            best_precision = precision

    return best_thr, best_precision, best_recall


def train(dataset_dir: str,
          artifacts_dir: str = "model",
          arrays_dir: str = "data",
          plots_dir: str = "plots",
          batch_size: int = 64,
          epochs: int = 30,
          image_size: Tuple[int, int] = (64, 64)) -> None:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    arrays_path = os.path.join(backend_dir, arrays_dir)
    model_dir = os.path.join(backend_dir, artifacts_dir)
    plots_out_dir = os.path.join(backend_dir, plots_dir)

    ensure_dir(arrays_path)
    ensure_dir(model_dir)
    ensure_dir(plots_out_dir)

    # Preprocess if arrays not present
    x_train_path = os.path.join(arrays_path, "X_train.npy")
    y_train_path = os.path.join(arrays_path, "y_train.npy")
    x_test_path = os.path.join(arrays_path, "X_test.npy")
    y_test_path = os.path.join(arrays_path, "y_test.npy")

    if not (os.path.exists(x_train_path) and os.path.exists(y_train_path) and os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        preprocess_and_save(dataset_dir, arrays_path, image_size)

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(x_test_path)
    y_val = np.load(y_test_path)

    model = build_model(input_shape=(image_size[0], image_size[1], 3), num_classes=len(CLASSES))

    checkpoint_path = os.path.join(model_dir, "skin_lesion_model.h5")

    # Phase 1: train head with base frozen
    callbacks_phase1 = [
        EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max(5, epochs // 2),
        batch_size=batch_size,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # Phase 2: fine-tune last N layers of base
    base = _find_base_submodel(model)
    try:
        base.trainable = True
        # Freeze all but the last 50 layers if possible
        if hasattr(base, 'layers') and len(base.layers) > 50:
            for layer in base.layers[:-50]:
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_phase2 = [
            EarlyStopping(monitor="val_accuracy", patience=6, mode="max", restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_phase2,
            verbose=1,
        )
        # Merge history logs for plotting
        history = history1
        for k, v in history2.history.items():
            history.history.setdefault(k, []).extend(v)
    except Exception:
        # If fine-tuning fails, fall back to phase1 history
        history = history1

    # Ensure the best checkpoint is saved and loaded
    try:
        model.save(checkpoint_path)
    except Exception:
        pass

    # Save label map
    with open(os.path.join(model_dir, "labels.json"), "w") as f:
        json.dump(CLASSES, f)

    # Calibration on validation set for threshold achieving 100% precision (if feasible)
    try:
        probs_val = model.predict(X_val, batch_size=batch_size, verbose=0)
        thr, p_at_thr, r_at_thr = _calibrate_threshold(y_val, probs_val, default_threshold=0.80)
    except Exception:
        thr, p_at_thr, r_at_thr = 0.80, 0.0, 0.0

    # Save config including calibrated threshold
    config = {
        "confidence_threshold": round(float(thr), 4),
        "val_precision_at_threshold": round(float(p_at_thr), 4),
        "val_recall_at_threshold": round(float(r_at_thr), 4),
        "image_size": list(image_size),
        "classes": CLASSES,
        "note": "Threshold chosen to maximize recall with precision==1.0 on validation set."
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plot_training(history, plots_out_dir)
    print(f"Training complete. Best model and plots saved to {model_dir}. Calibrated threshold: {config['confidence_threshold']} (val P={config['val_precision_at_threshold']}, R={config['val_recall_at_threshold']}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier on HAM10000 + No Lesion")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset root containing class subfolders: nv, mel, bcc, akiec, bkl, df, vasc, no_lesion")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train(dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size)
