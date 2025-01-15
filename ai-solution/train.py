from ultralytics import YOLO
import cv2


def main():
    model = YOLO("yolo11m.pt")
    results = model.train(
        data="/home/bart/projects/Industrial-Project/dataset/data.yaml",
        epochs=60,
        imgsz=640,
        device="0",
        cache=True,
        batch=0.8,
        perspective=0.01,
        auto_augment="augmix",
        # shear = 10,
        pretrained = '/home/bart/projects/Industrial-Project/train/runs/detect/train3/weights/best.pt',
        single_cls = True,
        cos_lr = True,
        plots= True,
    )


if __name__ == "__main__":
    main()
