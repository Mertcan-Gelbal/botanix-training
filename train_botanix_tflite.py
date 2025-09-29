#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Botanix — Jetson için TFLite’e uygun eğitim (hazır split: train/val/test)
Klasör yapısı:
  ./data/botanix-dataset/
      train/<class>/*.jpg
      val/<class>/*.jpg
      test/<class>/*.jpg

Çalıştırma:
  python3 train_botanix_tflite.py \
    --data_dir ./data/botanix-dataset \
    --out_dir ./outputs \
    --epochs 15 --img_size 256 --batch_size 64
"""

import os, time, json, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------
# Yardımcılar
# -----------------------
def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def enable_memory_growth():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if gpus:
            print(f"[INFO] GPU sayısı: {len(gpus)} — memory_growth=AÇIK")
    except Exception as e:
        print("[WARN] GPU memory growth ayarlanamadı:", e)

def parse_args():
    p = argparse.ArgumentParser(description="Botanix TFLite Trainer (Jetson)")
    p.add_argument("--data_dir", type=str, default="./data/botanix-dataset",
                   help="train/val/test klasörlerini içeren kök dizin")
    p.add_argument("--out_dir", type=str, default="./outputs",
                   help="Model ve TFLite çıktıları")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--int8_samples", type=int, default=300,
                   help="INT8 kalibrasyon örnek sayısı")
    p.add_argument("--mixed_precision", action="store_true",
                   help="TF sürümü uygunsa mixed precision")
    p.add_argument("--debug_max_per_class", type=int, default=0,
                   help="Her sınıftan max N örnek kullan (0=hepsi)")
    return p.parse_args()

# -----------------------
# Veri okuma (hazır split’ten)
# -----------------------
def list_classes(split_dir: Path) -> List[str]:
    """Sınıf isimlerini alt klasörlerden al (sadece dizinler)."""
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"Sınıf bulunamadı: {split_dir}")
    return classes

def collect_items(split_root: Path, classes: List[str], max_per_class: int = 0) -> Dict[str, np.ndarray]:
    """Belirli split (train/val/test) altında dosyaları topla."""
    items: List[Tuple[str, str]] = []
    for c in classes:
        cls_dir = split_root / c
        files = []
        if cls_dir.exists():
            for f in cls_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    files.append(str(f))
        if max_per_class > 0:
            files = files[:max_per_class]
        items += [(fp, c) for fp in files]
    if not items:
        raise RuntimeError(f"Boş split: {split_root}")
    random.shuffle(items)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return {
        "filepath":  np.array([p for p, _ in items]),
        "label":     np.array([l for _, l in items]),
        "label_idx": np.array([class_to_idx[l] for _, l in items], dtype=np.int32),
    }

def build_pipelines(train, val, test, img_size, batch_size, seed):
    AUTO = tf.data.AUTOTUNE
    IMG_SIZE = (img_size, img_size)

    def decode_image(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
        return img

    layers = tf.keras.layers
    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")

    def make_ds(split, training=False):
        paths, labels = split["filepath"], split["label_idx"]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda p,l: (decode_image(p), l), num_parallel_calls=AUTO)
        if training:
            ds = ds.map(lambda x,y: (augment(x, training=True), y), num_parallel_calls=AUTO)
        ds = ds.batch(batch_size).prefetch(AUTO)
        return ds

    return make_ds(train, True), make_ds(val, False), make_ds(test, False)

# -----------------------
# Model
# -----------------------
def build_model(num_classes, img_size, dropout=0.25):
    input_shape = (img_size, img_size, 3)
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base.trainable = True
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=True)                # BN güncellensin
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Activation('linear', dtype='float32')(x)  # mixed precision güvenliği
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name='mnv3_large_botanix')

# -----------------------
# TFLite export
# -----------------------
def export_tflite(saved_dir: Path, rep_paths: List[str], img_size: int, out_dir: Path,
                  int8_samples: int = 300):
    out_dir.mkdir(parents=True, exist_ok=True)

    def representative_dataset():
        tmp = list(rep_paths)
        random.shuffle(tmp)
        sample = tmp[:min(int8_samples, len(tmp))]
        for p in sample:
            img = tf.io.read_file(p)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (img_size, img_size), antialias=True)
            img = tf.expand_dims(img, 0)
            yield [img]

    # FP16
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        tflite_fp16 = conv.convert()
        (out_dir / "model_float16.tflite").write_bytes(tflite_fp16)
        print("[OK] TFLite FP16 yazıldı:", out_dir / "model_float16.tflite")
    except Exception as e:
        print("[WARN] FP16 TFLite oluşturulamadı:", e)

    # INT8 full-integer
    try:
        conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tflite_int8 = conv.convert()
        (out_dir / "model_int8.tflite").write_bytes(tflite_int8)
        print("[OK] TFLite INT8 yazıldı:", out_dir / "model_int8.tflite")
    except Exception as e:
        print("[WARN] INT8 TFLite oluşturulamadı:", e)

# -----------------------
# Callback: terminal epoch logu
# -----------------------
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()
        print(f"\n[Epoch {epoch+1}] başlıyor...")

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        logs = logs or {}
        msg = " | ".join([f"{k}={v:.4f}" for k,v in logs.items() if isinstance(v, (int, float))])
        print(f"[Epoch {epoch+1}] bitti ({dt:.1f}s) -> {msg}")

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seeds(42)
    enable_memory_growth()

    if args.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision: AÇIK")
        except Exception as e:
            print("[WARN] Mixed precision açılamadı:", e)

    data_dir = Path(args.data_dir).resolve()       # ./data/botanix-dataset
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = data_dir / "train"
    val_root   = data_dir / "val"
    test_root  = data_dir / "test"

    print(f"[INFO] Veri dizini: {data_dir}")
    print(f"[INFO] Çıktı dizini: {out_dir}")

    # Sınıfları train’den al, val/test’in aynı sınıfları içerdiğini varsay
    classes = list_classes(train_root)
    # Splitleri hazır klasörlerden oku
    train = collect_items(train_root, classes, max_per_class=args.debug_max_per_class)
    val   = collect_items(val_root,   classes, max_per_class=args.debug_max_per_class) if val_root.exists() else None
    test  = collect_items(test_root,  classes, max_per_class=args.debug_max_per_class) if test_root.exists() else None

    if val is None:
        raise RuntimeError("Val split bulunamadı. Beklenen yol: {}/val".format(data_dir))
    if test is None:
        raise RuntimeError("Test split bulunamadı. Beklenen yol: {}/test".format(data_dir))

    num_classes = len(classes)
    print(f"[INFO] Sınıf sayısı: {num_classes}")
    print(f"[INFO] Örnek sayıları -> train: {len(train['filepath'])} | val: {len(val['filepath'])} | test: {len(test['filepath'])}")

    # Etiket dosyaları
    (out_dir / "labels.txt").write_text("\n".join(classes), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps({"class_names": classes}, ensure_ascii=False, indent=2), encoding="utf-8")

    # tf.data
    train_ds, val_ds, test_ds = build_pipelines(train, val, test, args.img_size, args.batch_size, seed=42)

    # Model
    model = build_model(num_classes, args.img_size, dropout=args.dropout)
    model.summary()

    # Optimizasyon
    if hasattr(tf.keras.optimizers, "AdamW"):
        opt = tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
        print("[INFO] Optimizer: AdamW")
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        print("[INFO] Optimizer: Adam (AdamW yok)")

    # Loss (label smoothing destek kontrolü)
    import inspect
    supports_ls = "label_smoothing" in inspect.signature(
        tf.keras.losses.SparseCategoricalCrossentropy.__init__
    ).parameters
    if supports_ls and args.label_smoothing > 0:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=args.label_smoothing)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        if args.label_smoothing > 0 and not supports_ls:
            print("[WARN] Bu TF kurulumunda SparseCCE için label_smoothing desteklenmiyor; smoothing devre dışı.")

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    # Callbacks
    ckpt_path = str(out_dir / "best_mnv3.h5")
    callbacks = [
        EpochLogger(),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv")),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, mode="max", min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=False, verbose=1),
    ]

    # Eğitim
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Test
    print("\n[TEST] Değerlendirme başlıyor...")
    metrics = model.evaluate(test_ds, verbose=1)
    print("[TEST] Sonuçlar:", dict(zip(model.metrics_names, metrics)))

    # Kayıt + TFLite
    saved_dir = out_dir / "saved_model"
    model.save(saved_dir, include_optimizer=False)
    print("[OK] SavedModel kaydedildi:", saved_dir)

    try:
        export_tflite(saved_dir, rep_paths=list(map(str, train["filepath"])), img_size=args.img_size,
                      out_dir=out_dir, int8_samples=args.int8_samples)
    except Exception as e:
        print("[WARN] TFLite export sırasında sorun:", e)

    print("\n[OK] Eğitim ve dışa aktarma tamamlandı.")
    print(f"Çıktılar: {out_dir}\n- best_mnv3.h5\n- saved_model/\n- model_float16.tflite (varsa)\n- model_int8.tflite (varsa)\n- labels.txt / labels.json\n- train_log.csv")

if __name__ == "__main__":
    main()
