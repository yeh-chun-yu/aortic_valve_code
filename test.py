from ultralytics import YOLO
import os
import gc
import torch


def main():
    model_list = [
        "12n", "12s", "12m", "12l", "12x",
        "11n", "11s", "11m", "11l", "11x",
        "v9t", "v9s", "v9m", "v9c", "v9e"
    ]

    for model_name in model_list:
        for fold in range(5):
            print(f"\n===== Testing model: {model_name} | fold{fold} =====")

            weight_path = f"./runs/detect/last/yolo{model_name}_fold{fold}/weights/best.pt"
            data_path = "data_full_fold_test.yaml"

            if not os.path.exists(weight_path):
                print(f"Weight file not found, skipping: {weight_path}")
                continue

            model = YOLO(weight_path)

            results = model.val(
                data=data_path,
                device="0",
                name=f"test/{model_name}_fold{fold}",
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
