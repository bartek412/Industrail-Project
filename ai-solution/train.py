from ultralytics import YOLO
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data.yaml file"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="Device to use (e.g. cpu, 0, 1, etc.)"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Path to pretrained weights file"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = YOLO("yolo11m.pt")

    # Prepare training configuration
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": 640,
        "device": args.device,
        "cache": True,
        "batch": 0.8,
        "perspective": 0.01,
        "auto_augment": "augmix",
        "single_cls": True,
        "cos_lr": True,
        "plots": True,
    }

    # Add pretrained weights if specified
    if args.pretrained:
        train_args["pretrained"] = args.pretrained

    # Start training
    model.train(**train_args)


if __name__ == "__main__":
    main()
