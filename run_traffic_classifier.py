#!/usr/bin/env python3
"""
Standalone Traffic Sign Classifier

This script reproduces the entire Jupyter notebook workflow without requiring
Jupyter. It downloads the dataset, trains a LeNet-style CNN, evaluates it,
and allows inference on new images.

Usage:
    python run_traffic_classifier.py --help
    python run_traffic_classifier.py
    python run_traffic_classifier.py --epochs 20 --batch-size 256
    python run_traffic_classifier.py --eval-only --model-path lenet.keras
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Local imports ---
from src.data_handler import (
    download_and_extract_dataset,
    load_traffic_data,
    preprocess_images,
)
from src.model_handler import ModelTrainer, ModelInference


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Classifier (no Jupyter)")
    parser.add_argument(
        "--data-dir", default="dataset", help="Directory to store dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.003, help="Learning rate (default: 0.003)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training; only evaluate existing model"
    )
    parser.add_argument(
        "--model-path", default="lenet.keras", help="Path to saved model (default: lenet.keras)"
    )
    parser.add_argument(
        "--test-image", help="Path to local image for inference demo"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download dataset and exit"
    )
    return parser.parse_args()


def setup_dataset(data_dir: str):
    """Download dataset if necessary and return data loaders."""
    if not Path(data_dir).exists():
        print("Dataset not found; downloading...")
        download_and_extract_dataset(data_dir)
    else:
        print("Dataset already exists; skipping download.")

    train, valid, test = load_traffic_data(data_dir)
    print(f"Train: {len(train['features'])} images")
    print(f"Valid: {len(valid['features'])} images")
    print(f"Test : {len(test['features'])} images")
    return train, valid, test


def main():
    args = parse_args()

    # 1. Dataset --------------------------------------------------------------
    train, valid, test = setup_dataset(args.data_dir)
    if args.download_only:
        print("Dataset ready. Exiting.")
        return

    # 2. Pre-processing -------------------------------------------------------
    X_train = preprocess_images(train["features"])
    y_train = train["labels"]
    X_valid = preprocess_images(valid["features"])
    y_valid = valid["labels"]
    X_test = preprocess_images(test["features"])
    y_test = test["labels"]

    # 3. Training or Evaluation ----------------------------------------------
    trainer = ModelTrainer(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )

    if not args.eval_only:
        print("Starting training...")
        model, history = trainer.train(X_train, y_train, X_valid, y_valid)
        model.save(args.model_path)
        print(f"Training complete. Model saved to {args.model_path}")
    else:
        print("Skipping training (--eval-only)")

    # 4. Evaluation -----------------------------------------------------------
    print("Evaluating on test set...")
    inference = ModelInference(model_path=args.model_path)
    predictions = inference.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # 5. Optional inference demo ---------------------------------------------
    if args.test_image:
        if not Path(args.test_image).exists():
            print(f"Image file not found: {args.test_image}")
            sys.exit(1)
        img = cv2.imread(args.test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        img = preprocess_images(np.array([img]))
        pred = inference.predict(img)
        sign_names = pd.read_csv("signnames.csv")
        top5 = [(sign_names.iloc[i][1], p) for i, p in enumerate(pred[0])]
        top5 = sorted(top5, key=lambda x: x[1], reverse=True)[:5]
        print("Top-5 predictions:")
        for name, prob in top5:
            print(f"  {name:<35} {prob:.3f}")


if __name__ == "__main__":
    main()