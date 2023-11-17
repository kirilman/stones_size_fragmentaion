from dataclasses import dataclass
from os import path
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from torch import mode
from ultralytics.utils.ops import xywhn2xyxy
from pathlib import Path


def polygonFromMask(
    maskedArr,
):  # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask_util.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)
    return segmentation[0]  # , [x, y, w, h], area


def read_segmentation_labels(p):
    with open(p, "r") as f:
        lines = f.readlines()
    return [np.fromstring(line, sep=" ") for line in lines]


class SamNet:
    def __init__(self, type: str = "vit_l", path2checkpoint: str = "") -> None:
        self.predictor = SamPredictor(
            sam_model_registry[type](checkpoint=path2checkpoint).to("cuda:1")
        )

    def set_image(self, image: np.array):
        self.predictor.set_image(image)

    def process_image(
        self,
        image: np.array,
        point_coords: np.array = None,
        point_labels: np.array = None,
        box: np.array = None,
    ):
        if len(image.shape) > 2:
            h, w, _ = image.shape
        else:
            h, w = image.shape

        y_pred, _, _ = self.predictor.predict(
            point_coords, point_labels, box, multimask_output=False
        )
        polygone = np.array(polygonFromMask(y_pred[0].astype(np.uint8))).astype(
            np.float64
        )
        polygone[0::2] /= w
        polygone[1::2] /= h
        return polygone


class SamProcessor:
    def __init__(self, model_type, path2label, path2image, path2sam, path2save):
        self.sam = SamNet(model_type, path2sam)
        self.path2label = Path(path2label)
        self.path2image = Path(path2image)
        self.path2save = Path(path2save)

    def run(self):
        f_images = {x.stem: x for x in Path(self.path2image).glob("*")}
        for p in list(Path(self.path2label).glob("*.txt")):
            print(p)
            path2image = f_images[p.stem]
            lines = read_segmentation_labels(p)
            image = cv2.imread(str(path2image))
            h, w = self._image_size(image)
            self.sam.set_image(image)
            polygones = []
            for line in lines:
                xyxy = xywhn2xyxy(np.array(line[1:]), w, h)
                polygone = self.sam.process_image(image, box=xyxy)
                polygones.append(polygone)
            self._savetofile(self.path2save / p.name, polygones)

    def _image_size(self, image):
        if len(image.shape) > 2:
            h, w, _ = image.shape
        else:
            h, w = image.shape
        return h, w

    def _savetofile(self, fpath, polygones):
        with open(fpath, "w") as file:
            for poly in polygones:
                file.write("0 ")
                for p in poly:
                    file.write("{:.3f} ".format(p))
                file.write("\n")
