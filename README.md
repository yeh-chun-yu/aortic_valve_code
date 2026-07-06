# Aortic Valve Detection

This repository contains the training, testing, inference, and ensemble evaluation code for aortic valve object detection in cardiac computed tomography (CT) images. The project is based on the Ultralytics YOLO framework and includes experiments with YOLOv9, YOLO11, YOLO12, and the proposed YOLOv12-P34 architecture.

The goal of this project is to automatically detect the aortic valve region in CT images using deep learning-based object detection models.

## Overview

Aortic valve detection in CT images is challenging because the target object is small, the surrounding cardiac structures are complex, and the valve appearance may vary across patients and CT slices.

This project evaluates multiple YOLO-based object detection models for single-class aortic valve detection. In addition to standard YOLO models, this repository includes YOLOv12-P34, a modified YOLOv12 architecture that keeps the P3 and P4 detection layers and removes the P5 detection layer to better focus on small and medium-sized targets.

## Features

- YOLOv9, YOLO11, and YOLO12 training scripts
- YOLOv12-P34 training scripts
- Five-fold cross-validation based on patient-level split
- Dataset preparation from challenge ZIP files
- Training YAML generation
- Model testing and evaluation scripts
- Single-model inference example
- Fold-based ensemble evaluation
- Modified Ultralytics model configuration files
- Single-class aortic valve object detection

## Repository Structure

```text
aortic_valve_code/
├── train.py                    # Train YOLOv9, YOLO11, and YOLO12 baseline models
├── test.py                     # Test YOLOv9, YOLO11, and YOLO12 baseline models
│
├── train_p34.py                # Train YOLOv12-P34 models
├── test_p34.py                 # Test YOLOv12-P34 models
│
├── ensemble.py                 # Ensemble evaluation for baseline YOLO models
├── ensemble_p34.py             # Ensemble evaluation for YOLOv12-P34 models
│
├── split_data.py               # Prepare dataset and generate YAML files
├── fold_patient_split.csv      # Patient-level 5-fold split file
│
├── pyproject.toml              # Project and dependency configuration
├── README.md                   # Project documentation
├── LICENSE                     # License file
│
└── ultralytics/                # Modified Ultralytics source code and model configs
    └── cfg/
        └── models/
            └── 12/
                ├── yolo12n-p34.yaml
                ├── yolo12s-p34.yaml
                ├── yolo12m-p34.yaml
                ├── yolo12l-p34.yaml
                └── yolo12x-p34.yaml
```

## Environment Setup with Anaconda

The recommended Python version is:

```text
Python 3.12.12
```

### 1. Create a Conda environment

```bash
conda create -n aortic_valve python=3.12.12 -y
conda activate aortic_valve
```

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.12
```

### 2. Install PyTorch

For GPU training, install the PyTorch version that matches your CUDA environment.

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you want to install the CPU version only:

```bash
pip install torch torchvision torchaudio
```

### 3. Clone this repository

```bash
git clone https://github.com/yeh-chun-yu/aortic_valve_code.git
cd aortic_valve_code
```

### 4. Install the project

Install the repository in editable mode:

```bash
pip install -e .
```

If additional packages are needed for dataset preparation, install them manually:

```bash
pip install pyyaml
```

### 5. Verify the installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics import success')"
```

If `torch.cuda.is_available()` returns `True`, the GPU can be used for training and inference.

## Dataset Preparation

Please prepare the challenge dataset ZIP files in the following structure:

```text
aortic_valve_code/
└── dataset_challenge_2/
    ├── training_image.zip
    ├── training_label.zip
    ├── testing_image.zip
    └── testing_label.zip
```

The patient-level fold split CSV should be placed in the project root:

```text
aortic_valve_code/
└── fold_patient_split.csv
```

### Supported CSV format

The recommended CSV format is:

```csv
split,fold0,fold1,fold2,fold3,fold4
train,patient0032,patient0019,patient0019,patient0019,patient0019
train,patient0047,patient0049,patient0049,patient0049,patient0049
val,patient0001,patient0002,patient0003,patient0004,patient0005
```

The script also supports long-format CSV:

```csv
fold,split,patient
0,train,patient0001
0,val,patient0009
1,train,patient0002
1,val,patient0010
```

### Run dataset preparation

Run:

```bash
python split_data.py
```

The script will:

1. Extract the four ZIP files.
2. Normalize the image and label folder structure.
3. Split the training data into five folds according to `fold_patient_split.csv`.
4. Copy the official testing set into each fold folder.
5. Generate YOLO training and testing YAML files.

### Generated dataset structure

After running `split_data.py`, the generated dataset will be placed outside the repository folder:

```text
../datasets_full/
├── fold0/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── fold1/
├── fold2/
├── fold3/
└── fold4/
```

### Generated YAML files

The following YAML files will be generated automatically:

```text
data_full_fold0.yaml
data_full_fold1.yaml
data_full_fold2.yaml
data_full_fold3.yaml
data_full_fold4.yaml
data_full_fold_test.yaml
```

Example training YAML:

```yaml
path: ../datasets_full/fold0
train: train/images
val: val/images
names:
  0: aortic_valve
```

Example testing YAML:

```yaml
path: ../datasets_full/fold0
train: train/images
val: test/images
names:
  0: aortic_valve
```

The training scripts use:

```text
data_full_fold0.yaml ~ data_full_fold4.yaml
```

The testing scripts use:

```text
data_full_fold_test.yaml
```

## Training

Before training, make sure the dataset has been prepared:

```bash
python split_data.py
```

### Train baseline YOLO models

Run:

```bash
python train.py
```

This script trains the following models with five-fold cross-validation:

```text
YOLOv9:  yolov9t, yolov9s, yolov9m, yolov9c, yolov9e
YOLO11:  yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
YOLO12:  yolo12n, yolo12s, yolo12m, yolo12l, yolo12x
```

The script uses:

```text
epochs = 200
batch = 16 for most models
batch = 8 for larger models
patience = 50
device = 0
```

Training outputs are saved under:

```text
runs/detect/last/
```

Example output path:

```text
runs/detect/last/yolo12l_fold0/weights/best.pt
```

### Train YOLOv12-P34 models

Run:

```bash
python train_p34.py
```

This script trains the following YOLOv12-P34 models:

```text
yolo12n-p34
yolo12s-p34
yolo12m-p34
yolo12l-p34
yolo12x-p34
```

Example output path:

```text
runs/detect/last/yolo12l-p34_fold0/weights/best.pt
```

## Testing and Evaluation

### Test baseline YOLO models

Run:

```bash
python test.py
```

This script loads the trained baseline YOLO weights from:

```text
runs/detect/last/yolo{model_name}_fold{fold}/weights/best.pt
```

and evaluates them using:

```text
data_full_fold_test.yaml
```

### Test YOLOv12-P34 models

Run:

```bash
python test_p34.py
```

This script loads the trained YOLOv12-P34 weights from:

```text
runs/detect/last/{model_name}_fold{fold}/weights/best.pt
```

and evaluates them using:

```text
data_full_fold_test.yaml
```

Evaluation outputs are saved under:

```text
runs/detect/test/
```

## Inference

You can run inference with a trained model using the Ultralytics command line interface.

Example:

```bash
yolo predict \
  model=./runs/detect/last/yolo12l-p34_fold0/weights/best.pt \
  source=../datasets_full/fold0/test/images \
  imgsz=640 \
  conf=0.25 \
  device=0 \
  save=True \
  save_txt=True
```

The prediction results will be saved under:

```text
runs/detect/predict/
```

You can also run inference with Python:

```python
from ultralytics import YOLO

model = YOLO("./runs/detect/last/yolo12l-p34_fold0/weights/best.pt")

results = model.predict(
    source="../datasets_full/fold0/test/images",
    imgsz=640,
    conf=0.25,
    device="0",
    save=True,
    save_txt=True,
)
```

## Ensemble Evaluation

This repository provides fold-based ensemble evaluation scripts.

### Ensemble baseline YOLO models

Run:

```bash
python ensemble.py
```

This script evaluates ensemble combinations for baseline YOLO models.

### Ensemble YOLOv12-P34 models

Run:

```bash
python ensemble_p34.py
```

This script evaluates ensemble combinations for YOLOv12-P34 models.

The ensemble scripts load trained fold weights and evaluate different top-N fold combinations.

Example weight path used by the baseline ensemble script:

```text
runs/detect/last/yolo12l_fold0/weights/best.pt
```

Example weight path used by the YOLOv12-P34 ensemble script:

```text
runs/detect/last/yolo12l-p34_fold0/weights/best.pt
```

## Evaluation Metrics

The models are evaluated using standard object detection metrics from Ultralytics YOLO:

```text
Precision
Recall
mAP50
mAP50-95
```

## Notes

The dataset ZIP files, extracted dataset folders, training outputs, and model weights should not be uploaded to GitHub.

Recommended `.gitignore` entries:

```gitignore
dataset_challenge_2/
datasets/
datasets_full/
runs/
wandb/
*.zip
*.pt
*.pth
*.onnx
*.engine
*.cache
__pycache__/
```

Before running training, testing, or ensemble evaluation, make sure that:

1. The Conda environment has been created and activated.
2. The required packages have been installed.
3. The four dataset ZIP files are placed under `dataset_challenge_2/`.
4. `fold_patient_split.csv` is placed in the project root.
5. `split_data.py` has been executed successfully.
6. The generated YAML files exist in the project root.
7. The trained weights exist before testing or ensemble evaluation.

## License

This repository is released under the AGPL-3.0 License.

## Acknowledgement

This project is based on the Ultralytics YOLO framework.

## Author

Chun-Yu Yeh
