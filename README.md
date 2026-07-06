# Aortic Valve Detection

This repository provides code for aortic valve object detection in cardiac CT images using Ultralytics YOLO models. It includes baseline YOLO models, the proposed YOLOv12-P34 models, 5-fold training, testing, inference, and ensemble evaluation scripts.

## Main Features

- Train and test YOLOv9, YOLO11, and YOLO12 models
- Train and test YOLOv12-P34 models
- Prepare challenge ZIP files into YOLO-format datasets
- Generate 5-fold YAML files based on `fold_patient_split.csv`
- Run single-model inference
- Run fold-based ensemble evaluation

## File Description

- `train.py`: Train baseline YOLOv9, YOLO11, and YOLO12 models.
- `test.py`: Test baseline YOLOv9, YOLO11, and YOLO12 models.
- `train_p34.py`: Train YOLOv12-P34 models.
- `test_p34.py`: Test YOLOv12-P34 models.
- `ensemble.py`: Ensemble evaluation for baseline YOLO models.
- `ensemble_p34.py`: Ensemble evaluation for YOLOv12-P34 models.
- `split_data.py`: Extract ZIP files, organize datasets, split data into 5 folds, and generate YAML files.
- `fold_patient_split.csv`: Patient-level 5-fold split file.
- `ultralytics/`: Modified Ultralytics source code and YOLOv12-P34 YAML model files.
- `pyproject.toml`: Project dependency configuration.

## Environment Setup with Anaconda

Recommended environment:

- Python 3.12.12
- CUDA-compatible GPU for training
- PyTorch
- Ultralytics YOLO

Create and activate the Conda environment:

```bash
conda create -n aortic_valve python=3.12.12 -y
conda activate aortic_valve
```

Check Python version:

```bash
python --version
```

## Install PyTorch

Install the PyTorch version that matches your CUDA version. Please refer to the official PyTorch installation pages:

- PyTorch official website: https://pytorch.org/
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/

Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU-only example:

```bash
pip install torch torchvision torchaudio
```

Check whether PyTorch can use GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Install This Repository

Clone this repository:

```bash
git clone https://github.com/yeh-chun-yu/aortic_valve_code.git
cd aortic_valve_code
```

Install in editable mode:

```bash
pip install -e .
```

If needed, install additional packages:

```bash
pip install pyyaml
```

## Dataset Preparation

Prepare the dataset ZIP files as follows:

```text
aortic_valve_code/
└── dataset_challenge_2/
    ├── training_image.zip
    ├── training_label.zip
    ├── testing_image.zip
    └── testing_label.zip
```

Also place `fold_patient_split.csv` in the repository root.

Run dataset preparation:

```bash
python split_data.py
```

The script will extract the ZIP files, split the training data into 5 folds according to `fold_patient_split.csv`, copy the testing data, and generate YAML files for training and testing.

Generated dataset folder:

```text
../datasets_full/
├── fold0/
├── fold1/
├── fold2/
├── fold3/
└── fold4/
```

Each fold contains:

- `train/images`
- `train/labels`
- `val/images`
- `val/labels`
- `test/images`
- `test/labels`

Generated YAML files:

- `data_full_fold0.yaml`
- `data_full_fold1.yaml`
- `data_full_fold2.yaml`
- `data_full_fold3.yaml`
- `data_full_fold4.yaml`
- `data_full_fold_test.yaml`

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

## Training

Train baseline YOLO models:

```bash
python train.py
```

Train YOLOv12-P34 models:

```bash
python train_p34.py
```

Training outputs are saved under:

```text
runs/detect/last/
```

## Testing

Test baseline YOLO models:

```bash
python test.py
```

Test YOLOv12-P34 models:

```bash
python test_p34.py
```

Testing outputs are saved under:

```text
runs/detect/test/
```

## Inference

Run inference with a trained model:

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

Python inference example:

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

Prediction results are saved under:

```text
runs/detect/predict/
```

## Ensemble Evaluation

Run baseline YOLO ensemble evaluation:

```bash
python ensemble.py
```

Run YOLOv12-P34 ensemble evaluation:

```bash
python ensemble_p34.py
```

The ensemble scripts load trained fold weights from `runs/detect/last/` and evaluate different top-N fold combinations.

## Evaluation Metrics

The models are evaluated using the following object detection metrics:

- Precision
- Recall
- mAP50
- mAP50-95

## Reference

- PyTorch official website: https://pytorch.org/
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- Ultralytics YOLO GitHub: https://github.com/ultralytics/ultralytics/tree/main

## GitHub Upload Notes

The following files or folders should not be uploaded to GitHub:

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

## License

This repository is released under the AGPL-3.0 License.

## Acknowledgement

This project is based on the Ultralytics YOLO framework.

## Author

Chun-Yu Yeh
