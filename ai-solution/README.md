# YOLO for april tags

## Environment Setup

### 1. Install Anaconda or Miniconda
- Download and install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Navigate to the Project Directory
```bash
cd ai-solution
```

### 3. Create a Conda Environment
```bash
conda create --name yolo python=3.11 -y
```

### 4. Activate the Environment
```bash
conda activate yolo
```

### 5. Install YOLO Packages
#### For NVIDIA GPU:
```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

#### For Other Systems:
```bash
conda install -c pytorch -c conda-forge pytorch torchvision ultralytics
```

---

## Training Models

### Model 1: Training on a Large Roboflow Dataset

1. **Download the dataset** from the provided [SharePoint Link](https://tulodz-my.sharepoint.com/:u:/g/personal/253226_edu_p_lodz_pl/EZdEyy3Xk2NIgY1rdoQlB-MBBZyjEt9v7H1vJsCRps9KKg?e=SrsQDV).
2. **Unzip the dataset**.
3. **Update** the `path` parameter in `data.yaml` to the full path of the unzipped dataset.
4. **Run training**:
   ```bash
   python3 train.py --data <path_to_data.yaml> --epochs 80 --device 0
   ```
   - If you have an NVIDIA GPU, use `--device 0`.
   - If you have an Apple Silicon (M1/M2), use `--device "mps"`.
   - For other systems, use `--device "cpu"`.

---

### Model 2: Fine-Tuning with Project-Specific Data

1. **Run the dataset split script**:
   ```bash
   python3 create-dataset.py
   ```
   This will create the dataset (`dataset-our-videos`) in the parent folder.

2. **Start fine-tuning**:
   ```bash
   python3 train.py --data <path_to_data.yaml_from_dataset_our_videos> --pretrained <path_to_weights/best.pt> --epochs 30 --device 0
   ```

---

### Model 3: Training Exclusively on Project Data

1. **Run training only on the project dataset**:
   ```bash
   python3 train.py --data <path_to_data.yaml_from_dataset_our_videos> --epochs 80 --device 0
   ```

---

## Validating Models

```bash
python3 validate.py --model <path_to_weights/best.pt> --data ../dataset-our-videos/data.yaml --device 0
```

