"""
YOLOv8 Human Detection - Training Script
Train YOLOv8 model on custom human detection dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from typing import Optional


def train_model(data_yaml: str,
               model_size: str = "s",
               epochs: int = 20,
               imgsz: int = 640,
               batch: int = 16,
               device: str = "0",
               project: str = "runs",
               name: str = "train") -> Optional[YOLO]:
    """
    Train YOLOv8 model

    Args:
        data_yaml: Path to dataset configuration file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use (0 for GPU, cpu for CPU)
        project: Project name for saving results
        name: Experiment name

    Returns:
        Trained model or None if training fails
    """
    try:
        # Validate data file exists
        if not Path(data_yaml).exists():
            print(f"Error: Dataset config not found: {data_yaml}")
            return None

        # Load pretrained model
        model_name = f"yolov8{model_size}.pt"
        print(f"Loading pretrained model: {model_name}")
        model = YOLO(model_name)

        # Train model
        print(f"Starting training...")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Device: {device}")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            verbose=True
        )

        print("Training completed successfully!")
        return model

    except Exception as e:
        print(f"Error during training: {e}")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 Human Detection - Training Script"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to dataset configuration file (default: dataset/data.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n(nano), s(small), m(medium), l(large), x(xlarge) (default: s)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: 0 for GPU, cpu for CPU (default: 0)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project name for saving results (default: runs)"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name (default: train)"
    )

    args = parser.parse_args()

    # Train model
    model = train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )

    if model is None:
        print("Training failed. Exiting.")
        return

    print("\nTraining completed!")
    print(f"Best model saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()