from config import load_config
import argparse

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from ASBEST_VEINS_LABELING.labelutilits._data_spliter import merge_yolo_annotation
from ASBEST_VEINS_LABELING.labelutilits._converter import Yolo2Coco
from ASBEST_VEINS_LABELING.labelutilits._anno_collector import merge_anno

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config yaml.",
        default="./sam_params.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    merge_yolo_annotation(
        config["merge"]["path1"],
        config["input"],
        config["merge"]["path2"],
        config["merge"]["path2save"],
        config["merge"]["iou_tresh"],
    )

    Yolo2Coco(
        path_label=config["merge"]["path2save"],
        path_image=config["input"],
        path_save_json=config["path2savejson"],
    ).convert()
    # add old annotation
    merge_anno(
        config["merge"]["path2df_old"],
        ["new_anno", "old_anno"],
        config["input"],
        config["merge"]["path2_dfmerge"],
    )
