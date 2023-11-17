from config import load_config
import argparse

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from ASBEST_VEINS_LABELING.labelutilits._data_spliter import (
    merge_yolo_annotation,
    k_fold_split_yolo,
)
from ASBEST_VEINS_LABELING.labelutilits._converter import Yolo2Coco, coco2box_keypoints
from ASBEST_VEINS_LABELING.labelutilits._anno_collector import merge_anno

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert and splite fold for train model"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml.",
        default="./convert_params.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # coco2box_keypoints(config["path2anno"], config["path2yoloformat"])
    k_fold_split_yolo(
        config["path2yoloformat"],
        config["path2image"],
        config["path2savefold"],
        3,
        config["random_seed"],
    )
