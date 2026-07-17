from ultralytics import YOLO
import os
import gc
import torch
from pathlib import Path


def main():
    model_list = [
        "ultralytics/cfg/models/12/yolo12n-p5.yaml",
        "ultralytics/cfg/models/12/yolo12s-p5.yaml",
        "ultralytics/cfg/models/12/yolo12m-p5.yaml",
        "ultralytics/cfg/models/12/yolo12l-p5.yaml",
        "ultralytics/cfg/models/12/yolo12x-p5.yaml",
    ]

    data_path = "data_full_fold_test.yaml"

    for model_path in model_list:
        model_stem = Path(model_path).stem  # yolo12s-p5

        for fold in range(5):
            print(f"\n===== Testing model: {model_stem} | fold{fold} =====")

            weight_path = f"./runs/detect/last/{model_stem}_fold{fold}/weights/best.pt"

            if not os.path.exists(weight_path):
                print(f"Weight file not found, skipping: {weight_path}")
                continue

            model = YOLO(weight_path)

            results = model.val(
                data=data_path,
                device="0",
                name=f"test/{model_stem}_fold{fold}",
            )

            del results
            del model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
