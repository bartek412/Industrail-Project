from ultralytics import YOLO

# Load a model
model = YOLO("/home/bart/projects/Industrial-Project/train/runs/detect/train3/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(data="/home/bart/projects/Industrial-Project/dataset/data.yaml", device = "0", save_json = True)  # no arguments needed, dataset and settings remembered
print(f'---- WITHOUT FINE TUNING ----')
print(f'{metrics.box.map50=}') 

model = YOLO("/home/bart/projects/Industrial-Project/train/runs/detect/train5/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(data="/home/bart/projects/Industrial-Project/dataset/data.yaml", device = "0", save_json = True)  # no arguments needed, dataset and settings remembered
print(f'---- WITH FINE TUNING ----')
print(f'{metrics.box.map50=}') 
