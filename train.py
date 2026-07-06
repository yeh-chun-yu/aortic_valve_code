from ultralytics import YOLO
import gc
import torch

def main():
    model_list = [   
        "yolo12n.pt", "yolo11n.pt", "yolov9t.pt",
        "yolo12s.pt", "yolo11s.pt", "yolov9s.pt",
        "yolo12m.pt", "yolo11m.pt", "yolov9m.pt",
        "yolo12l.pt", "yolo11l.pt", "yolov9c.pt",
        "yolo12x.pt", "yolo11x.pt", "yolov9e.pt"
    ]

    small_batch_models = {"yolo12x.pt", "yolo11x.pt", "yolov9e.pt"}

    for model_name in model_list:
        # 根據模型大小決定 batch
        batch_size = 8 if model_name in small_batch_models else 16

        for fold in range(5):
            print(f"\n===== 模型: {model_name} | fold{fold} | batch={batch_size} =====")

            model = YOLO(model_name)

            results = model.train(
                data=f"data_full_fold{fold}.yaml",
                epochs=200,
                batch=batch_size,
                patience=50,
                name=f"last/{model_name[:-3]}_fold{fold}",
                device="0",
            )

            # 釋放記憶體
            del results
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
