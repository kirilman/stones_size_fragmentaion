yolo pose train model="yolov8x-pose.pt" data="/storage/reshetnikov/stone_fractions/dataset/sam_pred/fold/Fold_0/config.yaml" \
imgsz=1024 batch=8 epochs=100 single_cls=True  optimizer='SGD' \
project="/storage/reshetnikov/runs/knpt2/" name="test_dataset" cache="ram" save device='cuda:2,3' patience=0 