import argparse
from model import My_Model
from zrcrm_data import My_dataset
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import time
import pathlib
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infolder', type=str, default="./input")
    parser.add_argument('--outfolder', type=str, default="./output")
    parser.add_argument('--ckp', type=str, default="./snapshots/best.pth")
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    model = My_Model(only_output_target=True).to(args.device)

    model.load_state_dict(torch.load(args.ckp))
    model.eval()

    dataloader = DataLoader(
        dataset=My_dataset(args.infolder),
        batch_size=1)
    iterloader = iter(dataloader)

    out_dir = pathlib.Path(args.outfolder)
    out_dir.mkdir(parents=1, exist_ok=1)

    for i in range(dataloader.__len__()):
        start_time = time.time()
        with torch.no_grad():
            temp_out = model(next(iterloader))
        p_time = (time.time() - start_time)
        filename = dataloader.dataset.data_list[i].name
        print(f'process time for {filename}: {p_time}')
        Image.fromarray((torch.squeeze(temp_out, dim=0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_dir.joinpath(filename))







