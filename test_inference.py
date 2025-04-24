import os
import argparse
import yaml
import torch
import tqdm
import cv2

from utils import util
from torch.utils import data
from utils.dataset import Dataset


def inference(args, params):
    
    filenames = []
    
    for i in range(1, 2):
        filenames.append('data/images/' + str(i) + '.jpg')

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 1, False, collate_fn=Dataset.collate_fn)  
    
    model = torch.load('./weights/best.pt', map_location='cpu')['model'].float() 
    
    # model.half()  # needed if GPU is used
    model.eval()   
    
    # Create output folders
    txt_dir = 'inference_outputs/txt'
    img_dir = 'inference_outputs/img'
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inference loop
    for i, (samples, targets, paths) in enumerate(tqdm.tqdm(loader, desc='Running Inference')):
        samples = samples.to(device).float() / 255.0

        with torch.no_grad():
            outputs = model(samples)

        # outputs = util.non_max_suppression(outputs, 0.001, 0.65)
        outputs = util.non_max_suppression(outputs, 0.5, 0.65)

        for j, pred in enumerate(outputs):
            # Get original image path
            image_path = filenames[j]  # assumes loader returns image paths
            filename = filenames[j].split('/')[-1].split('.')[0]  # get filename without extension

            # Load original image for drawing
            original_img = cv2.imread(image_path)
            h_orig, w_orig = original_img.shape[:2]

            # Save predictions to TXT
            txt_path = os.path.join(txt_dir, f"{filename}.txt")
            with open(txt_path, 'w') as f:
                if pred is not None and len(pred):
                    for *xyxy, conf, cls in pred:
                        line = f"{int(cls)} {conf:.4f} {' '.join([f'{x:.2f}' for x in xyxy])}\n"
                        f.write(line)

                        # Draw boxes on image
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = f"{int(cls)} {conf:.2f}"
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Save annotated image
            img_save_path = os.path.join(img_dir, f"{filename}.jpg")
            cv2.imwrite(img_save_path, original_img)
    
    print('len(loader):', len(loader))
    print('len(dataset):', len(dataset))
    

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

    # CTRL k + C to comment out the following lines
    # CTRL k + U to uncomment the following lines

    # # the environment variable LOCAL_RANK tells the script which GPU the current process is using
    # args.local_rank = int(os.getenv('LOCAL_RANK', 0))  # if not set, default to 0, so not distributed training
    # # the environment variable WORLD_SIZE is the total number of processes (usually GPUs) involved in the training
    # args.world_size = int(os.getenv('WORLD_SIZE', 1))   # if not set, default to 1, so single-process training

    # if args.world_size > 1: # if multi-GPU training
    #     torch.cuda.set_device(device=args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # if args.local_rank == 0:    # only the main process (often called rank 0) runs the following
    #     if not os.path.exists('weights'):
    #         os.makedirs('weights')

    # util.setup_seed()   # set random seed for reproducibility
    # util.setup_multi_processes()    # set up multi-processes for distributed training

    # read parameters from yaml file, put them in a dictionary
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    #if args.train:
    #    train(args, params)
    inference(args, params)
        

if __name__ == "__main__":
    main()