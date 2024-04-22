#convert segment predict from txt to json
# python /storage/reshetnikov/yolov8_rotate/asbest/asbestutills/_converter.py --type 'yolo2coco' --image_dir "/storage/reshetnikov/open_pits_merge/merge_fraction/split/images" \
# --inpt_dir "/storage/reshetnikov/yolov8_rotate/stages/runs/segment/segm/labels/" --save_dir "/storage/reshetnikov/yolov8_rotate/stages/runs/segment/segm/labels/predict.json"

from asbestutills.metrics.wasserstein import var_confidence,wasserstein
from pathlib import Path
PATH2SAVE = Path('/storage/reshetnikov/yolov8_rotate/stages/runs/splite_comp/validation')
PATH2ANNO = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/images/anno.json'
PATH2SOURCE = '/storage/reshetnikov/open_pits_merge/merge_fraction/split/train_split/val'
#box 
box_model = '/storage/reshetnikov/runs/fraction_obb/box_splite_8x/weights/best.pt'
max_det = 2500
# var_confidence(box_model,
#                PATH2SOURCE,
#                PATH2SAVE / 'box',
#                0.05, max_det = max_det)
# wasserstein(PATH2SAVE / 'box', PATH2ANNO)

# #obb
# box_model = '/storage/reshetnikov/runs/fraction_obb/obb_splite_8x/weights/best.pt'
# var_confidence(box_model,
#                PATH2SOURCE,
#                PATH2SAVE / 'obb',
#                0.05, max_det = max_det)
# wasserstein(PATH2SAVE / 'obb', PATH2ANNO)

# #knpt
# box_model = '/storage/reshetnikov/runs/fraction_obb/kpnt_splite_8x3/weights/best.pt'
# var_confidence(box_model,
#                PATH2SOURCE,
#                PATH2SAVE / 'kpnt',
#                0.05, max_det = max_det)
# wasserstein(PATH2SAVE / 'kpnt', PATH2ANNO)
# #segment
# box_model = '/storage/reshetnikov/runs/fraction_obb/segm_splite_8x/weights/best.pt'
# var_confidence(box_model,
#                PATH2SOURCE,
#                PATH2SAVE / 'segm',
#                0.05, max_det = max_det)
# wasserstein(PATH2SAVE / 'segm', PATH2ANNO)


box_model = '/storage/reshetnikov/yolov9/runs/train-seg/exp/weights/best.pt'
var_confidence(box_model,
               PATH2SOURCE,
               PATH2SAVE / 'segm_v9',
               0.05, max_det = max_det)
wasserstein(PATH2SAVE / 'segm_v9', PATH2ANNO)




