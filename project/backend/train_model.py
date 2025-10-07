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


def build_model(input_shape=(64, 64, 3), num_classes: int=8) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
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

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "skin_lesion_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save label map
    with open(os.path.join(model_dir, "labels.json"), "w") as f:
        json.dump(CLASSES, f)

    plot_training(history, plots_out_dir)
    print("Training complete. Best model and plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier on HAM10000 + No Lesion")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset root containing class subfolders: nv, mel, bcc, akiec, bkl, df, vasc, no_lesion")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train(dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size)
