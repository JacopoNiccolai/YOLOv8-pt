import os
import argparse
import yaml
import torch

from utils import util


import torch
from pathlib import Path
import cv2
import numpy as np

import torch
from pathlib import Path
import cv2
import numpy as np

# Optional: class names (adjust based on your dataset)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def inference():
    # Load the model
    checkpoint = torch.load('./weights/best.pt', map_location='cpu')
    model = checkpoint['model'].float()
    model.eval()
    model.half()

    # Load and preprocess the image
    img_path = 'data/images/2.jpg'
    img0 = cv2.imread(img_path)  # BGR
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = img_resized.transpose((2, 0, 1))  # HWC to CHW
    img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float16) / 255.0
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)  # [1, 3, H, W]

    # Inference
    with torch.no_grad():
        preds = model(img_tensor)[0]  # assuming model returns (predictions, ...) tuple

    # NMS (optional: only if your model doesn’t do this internally)
    preds = preds[preds[:, 4] > 0.25]  # confidence threshold
    if preds.shape[0] == 0:
        print("No detections.")
        return

    # Scale boxes to original image size
    h0, w0 = img0.shape[:2]
    scale_x, scale_y = w0 / 640, h0 / 640
    boxes = preds.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Draw boxes and labels
    for *xyxy, conf, cls in boxes:
        label = f'{CLASS_NAMES[int(cls)]} {conf:.2f}' if int(cls) < len(CLASS_NAMES) else f'{conf:.2f}'
        xyxy = [int(x) for x in xyxy]
        cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img0, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save and show image
    output_path = 'runs/detect/exp'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    save_path = f'{output_path}/result.jpg'
    cv2.imwrite(save_path, img0)
    print(f"Saved results to {save_path}")

    cv2.imshow('Detection', img0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()  # get args

    # the environment variable LOCAL_RANK tells the script which GPU the current process is using
    args.local_rank = int(os.getenv('LOCAL_RANK', 0))  # if not set, default to 0, so not distributed training
    # the environment variable WORLD_SIZE is the total number of processes (usually GPUs) involved in the training
    args.world_size = int(os.getenv('WORLD_SIZE', 1))   # if not set, default to 1, so single-process training

    if args.world_size > 1: # if multi-GPU training
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:    # only the main process (often called rank 0) runs the following
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()   # set random seed for reproducibility
    util.setup_multi_processes()    # set up multi-processes for distributed training

    # read parameters from yaml file, put them in a dictionary
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    #if args.train:
    #    train(args, params)
    inference()
        

if __name__ == "__main__":
    main()