import csv
import re
import shutil
import zipfile
from pathlib import Path

import yaml


# =========================
# Dataset ZIP configuration
# =========================
# Expected input files:
#
# dataset_challenge_2/
# ├── training_image.zip
# ├── training_label.zip
# ├── testing_image.zip
# └── testing_label.zip
#
# The fold split CSV should be placed in the project root:
#
# fold_patient_split.csv
#
# Supported CSV formats:
# 1. Wide format:
#    split,fold0,fold1,fold2,fold3,fold4
#    train,patient0001,patient0002,...
#    val,patient0009,patient0010,...
#
# 2. Long format:
#    fold,split,patient
#    0,train,patient0001
#    0,val,patient0009

DATASET_ROOT = Path("./dataset_challenge_2")
SPLIT_CSV = Path("./fold_patient_split.csv")

TRAINING_IMAGE_ZIP = DATASET_ROOT / "training_image.zip"
TRAINING_LABEL_ZIP = DATASET_ROOT / "training_label.zip"
TESTING_IMAGE_ZIP = DATASET_ROOT / "testing_image.zip"
TESTING_LABEL_ZIP = DATASET_ROOT / "testing_label.zip"

EXTRACT_ROOT = DATASET_ROOT / "_extracted"
PREPARED_ROOT = DATASET_ROOT / "_prepared"

TRAIN_IMG_ROOT = PREPARED_ROOT / "training" / "images"
TRAIN_LBL_ROOT = PREPARED_ROOT / "training" / "labels"
TEST_IMG_ROOT = PREPARED_ROOT / "testing" / "images"
TEST_LBL_ROOT = PREPARED_ROOT / "testing" / "labels"

# This path matches the generated YAML files:
# path: ../datasets_full/fold0
OUT_ROOT = Path("../datasets_full")
YAML_DATASET_ROOT = "../datasets_full"

FOLDS = 5
CLASS_NAMES = {0: "aortic_valve"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".txt"}

# Set to True to rebuild folders from scratch
RESET_PREPARED_DATA = True
RESET_OUTPUT_DATA = True


def remove_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)


def check_required_files():
    required_files = [
        TRAINING_IMAGE_ZIP,
        TRAINING_LABEL_ZIP,
        TESTING_IMAGE_ZIP,
        TESTING_LABEL_ZIP,
        SPLIT_CSV,
    ]

    missing_files = [str(p) for p in required_files if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(f"  - {p}" for p in missing_files)
            + "\n\nPlease place the four ZIP files under ./dataset_challenge_2/ "
            "and fold_patient_split.csv in the project root."
        )


def safe_extract_zip(zip_path: Path, extract_dir: Path):
    if extract_dir.exists():
        return

    print(f"[INFO] Extracting {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    extract_dir_resolved = extract_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = (extract_dir / member.filename).resolve()
            if not str(target_path).startswith(str(extract_dir_resolved)):
                raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}")
        zf.extractall(extract_dir)


def has_target_files(path: Path, valid_exts: set[str]) -> bool:
    return any(p.is_file() and p.suffix.lower() in valid_exts for p in path.rglob("*"))


def find_patient_dirs(root: Path, valid_exts: set[str]) -> list[Path]:
    patient_pattern = re.compile(r"^patient\d+$", re.IGNORECASE)

    patient_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and patient_pattern.match(path.name):
            if has_target_files(path, valid_exts):
                patient_dirs.append(path)

    patient_dirs = sorted(patient_dirs, key=lambda p: p.name.lower())

    if not patient_dirs:
        raise RuntimeError(f"No patient folders were found under: {root}")

    return patient_dirs


def normalize_zip_to_patient_folders(zip_path: Path, dst_root: Path, valid_exts: set[str]):
    extract_dir = EXTRACT_ROOT / zip_path.stem
    safe_extract_zip(zip_path, extract_dir)

    patient_dirs = find_patient_dirs(extract_dir, valid_exts)
    dst_root.mkdir(parents=True, exist_ok=True)

    for patient_dir in patient_dirs:
        dst_patient_dir = dst_root / patient_dir.name
        dst_patient_dir.mkdir(parents=True, exist_ok=True)

        for src_file in sorted(patient_dir.rglob("*")):
            if not src_file.is_file() or src_file.suffix.lower() not in valid_exts:
                continue

            dst_file = dst_patient_dir / src_file.name
            if dst_file.exists():
                print(f"[WARN] Duplicate file name found, overwriting: {dst_file}")

            shutil.copy2(src_file, dst_file)


def prepare_dataset_from_zips():
    check_required_files()

    if RESET_PREPARED_DATA:
        remove_dir(PREPARED_ROOT)

    TRAIN_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_LBL_ROOT.mkdir(parents=True, exist_ok=True)

    normalize_zip_to_patient_folders(TRAINING_IMAGE_ZIP, TRAIN_IMG_ROOT, IMAGE_EXTS)
    normalize_zip_to_patient_folders(TRAINING_LABEL_ZIP, TRAIN_LBL_ROOT, LABEL_EXTS)
    normalize_zip_to_patient_folders(TESTING_IMAGE_ZIP, TEST_IMG_ROOT, IMAGE_EXTS)
    normalize_zip_to_patient_folders(TESTING_LABEL_ZIP, TEST_LBL_ROOT, LABEL_EXTS)


def clean_patient_name(value: str) -> str | None:
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none"}:
        return None

    return value


def read_fold_split_csv(csv_path: Path) -> dict[int, dict[str, list[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"Empty CSV file: {csv_path}")

    fieldnames_lower = {name.lower(): name for name in fieldnames}

    folds = {
        fold: {
            "train": [],
            "val": [],
        }
        for fold in range(FOLDS)
    }

    # Long format: fold, split, patient
    if {"fold", "split", "patient"}.issubset(set(fieldnames_lower.keys())):
        fold_col = fieldnames_lower["fold"]
        split_col = fieldnames_lower["split"]
        patient_col = fieldnames_lower["patient"]

        for row in rows:
            fold_value = str(row[fold_col]).strip().lower().replace("fold", "")
            split = str(row[split_col]).strip().lower()
            patient = clean_patient_name(row[patient_col])

            if patient is None:
                continue

            fold = int(fold_value)
            if fold not in folds:
                raise RuntimeError(f"Invalid fold index in CSV: {fold}")

            if split not in {"train", "val"}:
                raise RuntimeError(f"Invalid split in CSV: {split}")

            folds[fold][split].append(patient)

        return folds

    # Wide format: split, fold0, fold1, fold2, fold3, fold4
    required_wide_columns = {"split"} | {f"fold{fold}" for fold in range(FOLDS)}
    if required_wide_columns.issubset(set(fieldnames_lower.keys())):
        split_col = fieldnames_lower["split"]
        fold_cols = {
            fold: fieldnames_lower[f"fold{fold}"]
            for fold in range(FOLDS)
        }

        for row in rows:
            split = str(row[split_col]).strip().lower()

            if split not in {"train", "val"}:
                continue

            for fold, fold_col in fold_cols.items():
                patient = clean_patient_name(row[fold_col])
                if patient is not None:
                    folds[fold][split].append(patient)

        return folds

    raise RuntimeError(
        "Unsupported CSV format. Please use either:\n"
        "1. split,fold0,fold1,fold2,fold3,fold4\n"
        "2. fold,split,patient"
    )


def validate_fold_split(folds: dict[int, dict[str, list[str]]]):
    available_train_patients = {
        p.name for p in TRAIN_IMG_ROOT.iterdir()
        if p.is_dir()
    } & {
        p.name for p in TRAIN_LBL_ROOT.iterdir()
        if p.is_dir()
    }

    for fold in range(FOLDS):
        train_patients = folds[fold]["train"]
        val_patients = folds[fold]["val"]

        if not train_patients:
            raise RuntimeError(f"Fold {fold} has no training patients in CSV.")

        if not val_patients:
            raise RuntimeError(f"Fold {fold} has no validation patients in CSV.")

        overlap = sorted(set(train_patients) & set(val_patients))
        if overlap:
            raise RuntimeError(f"Fold {fold} has overlapping train/val patients: {overlap}")

        missing = sorted((set(train_patients) | set(val_patients)) - available_train_patients)
        if missing:
            print(f"[WARN] Fold {fold} has patients not found in prepared training data: {missing}")


def list_image_files(patient_img_dir: Path) -> list[Path]:
    return sorted(
        p for p in patient_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def get_unique_base(base: str, patient: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base

    new_base = f"{patient}_{base}"
    counter = 1

    while new_base in used_names:
        new_base = f"{patient}_{base}_{counter}"
        counter += 1

    used_names.add(new_base)
    return new_base


def copy_patient_files(
    patients: list[str] | set[str],
    src_img_root: Path,
    src_lbl_root: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()

    for patient in sorted(patients):
        img_dir = src_img_root / patient
        lbl_dir = src_lbl_root / patient

        if not img_dir.is_dir():
            print(f"[WARN] Image folder not found, skipping patient: {img_dir}")
            continue

        if not lbl_dir.is_dir():
            print(f"[WARN] Label folder not found, creating empty labels for patient: {patient}")

        for img_path in list_image_files(img_dir):
            base = img_path.stem
            unique_base = get_unique_base(base, patient, used_names)

            dst_img = dst_img_dir / f"{unique_base}{img_path.suffix.lower()}"
            dst_lbl = dst_lbl_dir / f"{unique_base}.txt"

            shutil.copy2(img_path, dst_img)

            src_lbl = lbl_dir / f"{base}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
                print(f"[WARN] Label file not found, created empty label: {src_lbl}")


def write_data_yaml(yaml_path: Path, path_value: str, train: str, val: str):
    data = {
        "path": path_value.replace("\\", "/"),
        "train": train,
        "val": val,
        "names": CLASS_NAMES,
    }

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Wrote YAML: {yaml_path}")


def write_used_split_csv(folds: dict[int, dict[str, list[str]]], out_csv: Path):
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "split", "patient"])

        for fold in range(FOLDS):
            for split in ["train", "val"]:
                for patient in folds[fold][split]:
                    writer.writerow([fold, split, patient])

    print(f"[OK] Wrote used split CSV: {out_csv}")


def get_testing_patients() -> list[str]:
    image_patients = {
        p.name for p in TEST_IMG_ROOT.iterdir()
        if p.is_dir()
    }

    label_patients = {
        p.name for p in TEST_LBL_ROOT.iterdir()
        if p.is_dir()
    }

    patients = sorted(image_patients & label_patients)

    if not patients:
        raise RuntimeError("No matched testing patient folders were found.")

    return patients


def build_folds_from_csv():
    if RESET_OUTPUT_DATA:
        remove_dir(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    folds = read_fold_split_csv(SPLIT_CSV)
    validate_fold_split(folds)

    testing_patients = get_testing_patients()

    for fold in range(FOLDS):
        fold_dir = OUT_ROOT / f"fold{fold}"

        train_patients = folds[fold]["train"]
        val_patients = folds[fold]["val"]

        print(
            f"[FOLD {fold}] "
            f"train={len(train_patients)} patients, "
            f"val={len(val_patients)} patients, "
            f"test={len(testing_patients)} patients"
        )

        copy_patient_files(
            patients=train_patients,
            src_img_root=TRAIN_IMG_ROOT,
            src_lbl_root=TRAIN_LBL_ROOT,
            dst_img_dir=fold_dir / "train" / "images",
            dst_lbl_dir=fold_dir / "train" / "labels",
        )

        copy_patient_files(
            patients=val_patients,
            src_img_root=TRAIN_IMG_ROOT,
            src_lbl_root=TRAIN_LBL_ROOT,
            dst_img_dir=fold_dir / "val" / "images",
            dst_lbl_dir=fold_dir / "val" / "labels",
        )

        # Copy the same official testing set into every fold folder.
        # This makes the following YAML path valid for each fold:
        # path: ../datasets_full/foldX
        # val: test/images
        copy_patient_files(
            patients=testing_patients,
            src_img_root=TEST_IMG_ROOT,
            src_lbl_root=TEST_LBL_ROOT,
            dst_img_dir=fold_dir / "test" / "images",
            dst_lbl_dir=fold_dir / "test" / "labels",
        )

        yaml_path_value = f"{YAML_DATASET_ROOT}/fold{fold}"

        write_data_yaml(
            yaml_path=Path(f"./data_full_fold{fold}.yaml"),
            path_value=yaml_path_value,
            train="train/images",
            val="val/images",
        )

        print(f"[OK] Fold {fold} dataset created at: {fold_dir}")

    # The current test.py and test_p5.py use data_full_fold_test.yaml.
    # The testing set is the same for all folds, so this file points to fold0/test.
    write_data_yaml(
        yaml_path=Path("./data_full_fold_test.yaml"),
        path_value=f"{YAML_DATASET_ROOT}/fold0",
        train="train/images",
        val="test/images",
    )

    write_used_split_csv(folds, Path("./fold_patient_split_used.csv"))


def main():
    prepare_dataset_from_zips()
    build_folds_from_csv()
    print("[DONE] Dataset preparation is complete.")


if __name__ == "__main__":
    main()
