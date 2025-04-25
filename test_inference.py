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
    
    for i in range(0, 11):
        filenames.append('data/images/' + str(i) + '.jpg')

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 5, False, collate_fn=Dataset.collate_fn)  
    
    # model = torch.load('./weights/best.pt', map_location='cpu')['model'].float() 
    checkpoint = torch.load('./weights/best.pt', map_location='cpu', weights_only=False)
    model = checkpoint['model'].float()
    
    # model.half()  # needed if GPU is used
    model.eval()   
    
    # Create output folders
    txt_dir = 'inference_outputs/txt'
    img_dir = 'inference_outputs/img'
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_bar = tqdm.tqdm(loader, desc='Running Inference')
    
    img_count = 0

    # Inference loop
    # for i, samples, targets) in enumerate(tqdm.tqdm(loader, desc='Running Inference')):
    for samples, targets, shapes in p_bar:
        samples = samples.to(device).float() / 255.0

        with torch.no_grad():
            outputs = model(samples)

        # outputs = util.non_max_suppression(outputs, 0.001, 0.65)
        outputs = util.non_max_suppression(outputs, 0.5, 0.65)
        
        for j, pred in enumerate(outputs):
            
            # Get original image path
            image_path = filenames[img_count]  # assumes loader returns image paths
            filename = filenames[img_count].split('/')[-1].split('.')[0]  # get filename without extension

            img_count += 1
            
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
                        
                        # scale coordinates to original image size
                        xyxy = [int(x) for x in xyxy]
                        xyxy[0] = int(xyxy[0] * w_orig / args.input_size)
                        xyxy[1] = int(xyxy[1] * h_orig / args.input_size)
                        xyxy[2] = int(xyxy[2] * w_orig / args.input_size)
                        xyxy[3] = int(xyxy[3] * h_orig / args.input_size)
                        
                        # Draw boxes on image
                        x1, y1, x2, y2 = [int(x) for x in xyxy]
                        label = f"{int(cls)}: {conf:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(original_img, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
    # parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('--epochs', default=500, type=int)
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--test', action='store_true')

    args = parser.parse_args()  # get args

    # CTRL k + C to comment out the following lines
    # CTRL k + U to uncomment the following lines

    # read parameters from yaml file, put them in a dictionary
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    inference(args, params)
        

if __name__ == "__main__":
    main()