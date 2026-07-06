from ultralytics.nn.tasks import Ensemble, load_checkpoint
from ultralytics import YOLO
import os
import gc
import torch


def build_ensemble(weight_paths, device="0"):
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    ensemble_model = Ensemble()

    for w in weight_paths:
        #print(f"載入權重: {w}")
        m, ckpt = load_checkpoint(w, device=device, fuse=True)
        ensemble_model.append(m)

    ensemble_model.names = ensemble_model[0].names
    ensemble_model.nc = ensemble_model[0].nc
    ensemble_model.yaml = ensemble_model[0].yaml
    ensemble_model.stride = ensemble_model[0].stride
    ensemble_model.task = "detect"
    ensemble_model.pt_path = weight_paths[0]
    ensemble_model.args = ensemble_model[0].args

    ensemble_model.fuse = lambda verbose=True: ensemble_model
    ensemble_model.is_fused = lambda thresh=10: True

    def ensemble_forward(self, x, augment=False, profile=False, visualize=False, embed=None, **kwargs):
        y = [
            module(x, augment=augment, profile=profile, visualize=visualize)[0]
            for module in self
        ]

        y = torch.cat(y, 2)

        return y, None

    ensemble_model.forward = ensemble_forward.__get__(ensemble_model, type(ensemble_model))

    yolo = YOLO(weight_paths[0])
    yolo.model = ensemble_model

    return yolo


def main():
    fold_rank = {
        "12n": [3, 4, 2, 1, 0],
        "12s": [2, 3, 4, 1, 0],
        "12m": [3, 0, 2, 4, 1],
        "12l": [3, 4, 0, 1, 2],
        "12x": [3, 2, 4, 0, 1],

        "11n": [3, 1, 0, 2, 4],  
        "11s": [3, 4, 0, 1, 2],
        "11m": [3, 0, 4, 2, 1],
        "11l": [3, 2, 4, 1, 0],
        "11x": [0, 3, 2, 1, 4],

        "v9t": [0, 4, 2, 1, 3],
        "v9s": [4, 3, 0, 2, 1],
        "v9m": [4, 3, 2, 0, 1],
        "v9c": [4, 2, 0, 3, 1],
        "v9e": [0, 4, 2, 3, 1],
    }

    data_path = "data_full_fold_test.yaml"
    device = "0"


    top_list = [1, 2, 3, 4, 5]

    for model_name, ranks in fold_rank.items():
        for top_n in top_list:
            selected_folds = ranks[:top_n]

            print(f"\n===== Ensemble 測試模型: {model_name} | top{top_n} folds {selected_folds} =====")

            weight_paths = []

            for fold in selected_folds:
                weight_path = f"./runs/detect/last/yolo{model_name}_fold{fold}/weights/best.pt"

                if not os.path.exists(weight_path):
                    print(f"找不到權重，跳過此 fold: {weight_path}")
                    continue

                weight_paths.append(weight_path)

            if len(weight_paths) < top_n:
                print(f"{model_name} top{top_n} 權重不足，跳過")
                continue

            #print(f"{model_name} 使用 {len(weight_paths)} 個 fold 做 ensemble")

            model = build_ensemble(weight_paths, device=device)

            results = model.val(
                iou=0.4,
                data=data_path,
                device=device,
                name=f"test/{model_name}_ensemble_top{top_n}_iou0.4",
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