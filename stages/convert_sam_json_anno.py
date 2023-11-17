from config import load_config
import argparse

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from ASBEST_VEINS_LABELING.labelutilits._data_spliter import merge_yolo_annotation
from ASBEST_VEINS_LABELING.labelutilits._converter import Yolo2Coco
from ASBEST_VEINS_LABELING.labelutilits._anno_collector import merge_anno
from ASBEST_VEINS_LABELING.labelutilits._coords_transition import convert_sam_to_yolo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml.",
        default="./sam_to_json.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    convert_sam_to_yolo(config["path2samlabel"], config["path2yolo"], 200)

    Yolo2Coco(
        path_label=config["path2samlabel"],
        path_image=config["path2image"],
        path_save_json=config["path2savejson"],
    ).convert()
