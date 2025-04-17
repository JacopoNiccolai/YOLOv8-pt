
import torch
import os
import argparse
import yaml

from torch.utils import data

from utils import util
from utils.dataset import Dataset

def inference(args, params):
    print("Hello World! This is a test inference script.")
    
    # set device to CPU
    device = torch.device('cpu')
    # load the model
    model = torch.load('./weights/best.pt', map_location='cpu')['model'].float()
    
    # data handling
    filenames = []
    
    # add to filenames list all the files names in the directory data/images
    for i in range(1, 10):
        filenames.append(f"data/images/{i}.jpg")
        
    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    # load the model to the device
    model.to(device).eval()
    model.half()  # to FP16
    
    # inference
    for i, (img, img_path) in enumerate(loader):
        img = img.to(device).half()  # to FP16
        pred = model(img)[0]
        
        # print the shape of the prediction
        print(f"Prediction shape: {pred.shape}")
        
        # print the image path
        print(f"Image path: {img_path[0]}")
        
        # print the prediction  
        print(f"Prediction: {pred[j][0]} {pred[j][1]} {pred[j][2]} {pred[j][3]} {pred[j][4]} {pred[j][5]}")
        
    print("OK")

    

if __name__ == "__main__":
    print(torch.__version__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    # if args.train:
    #     train(args, params)
    # if args.test:
    #     test(args, params)
    
    inference(args, params)
