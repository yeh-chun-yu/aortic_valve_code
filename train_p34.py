from ultralytics import YOLO
import gc
import torch
from pathlib import Path


def main():
    model_list = [
        "ultralytics/cfg/models/12/yolo12n-p34.yaml",
        "ultralytics/cfg/models/12/yolo12s-p34.yaml",
        "ultralytics/cfg/models/12/yolo12m-p34.yaml",
        "ultralytics/cfg/models/12/yolo12l-p34.yaml",
        "ultralytics/cfg/models/12/yolo12x-p34.yaml"
    ]

    # Use a smaller batch size for larger models
    small_batch_models = {
        "yolo12x-p34.yaml",
    }

    for model_path in model_list:
        model_file = Path(model_path).name          # yolo12n-p34.yaml
        model_stem = Path(model_path).stem          # yolo12n-p34

        batch_size = 8 if model_file in small_batch_models else 16

        for fold in range(5):
            print(f"\n===== Model: {model_stem} | fold{fold} | batch={batch_size} =====")

            model = YOLO(model_path)

            results = model.train(
                data=f"data_full_fold{fold}.yaml",
                epochs=200,
                batch=batch_size,
                patience=50,
                name=f"last/{model_stem}_fold{fold}",
                device="0",
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
