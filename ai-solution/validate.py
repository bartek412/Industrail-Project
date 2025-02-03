from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Validation Script")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data.yaml file"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="Device to use (e.g. cpu, 0, mps, etc.)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size for validation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model = YOLO(args.model)

    # Run validation
    metrics = model.val(
        data=args.data,
        device=args.device,
        batch=args.batch,
        save_json=True,
        plots=True,
    )

    # Print validation results
    print("Validation Results:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")


if __name__ == "__main__":
    main()
