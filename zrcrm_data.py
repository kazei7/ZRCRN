import pathlib
from typing import Any
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from zrcrm_loss import CannyDetector, get_state_dict, cal_huidu

class My_dataset(data.Dataset):
    def __init__(self, data_path, filename_pattern="*", with_ref=False, in_path='input', tar_path='target'):
        data_path = pathlib.Path(data_path)
        self.with_ref = with_ref
        if with_ref:
            # data_path shuould be built like this:
            # "data_path"
            # ├─"in_path"
            # └─"tar_path"
            # image file saved in "in_path" and "tar_path", paired data should be named same
            self.in_path = in_path
            self.tar_path = tar_path
            self.data_path = data_path
            self.data_list = sorted([str(f.name) for f in data_path.joinpath(in_path).glob(filename_pattern)])
        else:
            self.data_list = [f for f in sorted(data_path.glob(filename_pattern), key=lambda i: i.stem)]

    def __getitem__(self, index):
        if self.with_ref:
            temp = self.getitem_withref(index)
        else:
            temp = self.getitem_withoutref(index)
        return temp

    def getitem_withoutref(self, index):
        file_path = str(self.data_list[index])
        test = self.get_img(file_path)
        return test

    def getitem_withref(self, index):
        in_file_path = str(self.data_path.joinpath(self.in_path, self.data_list[index]))
        tar_file_path = str(self.data_path.joinpath(self.tar_path, self.data_list[index]))

        in_img = self.get_img(in_file_path)
        tar_img = self.get_img(tar_file_path)

        return (in_img, tar_img)

    def get_img(self, path:str):
        img = Image.open(path)
        img = img.convert("RGB")
        # img = img.resize(size, Image.Resampling.LANCZOS)
        img = (np.asarray(img) / 255.0)
        img = torch.from_numpy(img)
        return img.permute(2, 0, 1).to(torch.float32)

    def __len__(self):
        return len(self.data_list)

class Multi_input_dataset(data.Dataset):
    def __init__(self, data_path, train_folders:list, filename_pattern="*", with_ref=False, ref_folder=None):
        super(Multi_input_dataset, self).__init__()
        self.train_folders = train_folders
        # data_path shuould be built like this:
        # "data_path"
        # ├─want_folder[index]
        # └─ref_folder
        # image file saved in each want_folder, associated image file should be named same
        if with_ref:
            self.train_folders.append(ref_folder)
        self.data_path = pathlib.Path(data_path)
        self.data_list = [f.name for f in sorted(self.data_path.joinpath(train_folders[0]).glob(filename_pattern), key=lambda i: i.stem)]

    def __getitem__(self, index):
        input = {}
        for folder in self.train_folders:
            path = str(self.data_path.joinpath(folder, self.data_list[index]))
            input[folder] = self.get_img(path)
        return input

    def get_img(self, path: str):
        img = Image.open(path)
        # img = img.resize(size, Image.Resampling.LANCZOS)
        if img.mode == 'RGB':
            img = (np.asarray(img) / 255.0)
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).to(torch.float32)
        elif img.mode == 'L':
            img = (np.asarray(img) / 255.0)
            img = torch.from_numpy(img)
            img = img.unsqueeze(0).unsqueeze(0).to(torch.float32)
        return img

    def __len__(self):
        return len(self.data_list)
    



def prepare_assist_datasets(in_path, device='cpu', hparams={}):
    hparams.setdefault('want_exposure_min', 0.2)
    hparams.setdefault('adaptivesmooththreshold_low', 0.707)
    hparams.setdefault('adaptivesmooththreshold_up', 1.414)
    hparams.setdefault('adaptivesmoothscale', 2)
    hparams.setdefault('device', device)
    data_dict = {
        'exptar': "Get_exptar(E_min=hparams['want_exposure_min'], E_max=1)",
        'adaptivesmoothtar': "Get_adaptivesmoothtar(\
                                    threshold_low=hparams['adaptivesmooththreshold_low'],\
                                    threshold_up=hparams['adaptivesmooththreshold_up'],\
                                    k=hparams['adaptivesmoothscale'], device=hparams['device'])"
    }
    in_path = pathlib.Path(in_path)
    in_dataset = My_dataset(in_path.joinpath('train'), '*')
    for tar_folder, funcstr in data_dict.items():
        if tar_folder == 'exptar':
            tar_folder = hparams['train_subfolders'][1]
        if tar_folder == 'adaptivesmoothtar':
            tar_folder = hparams['train_subfolders'][2]
        if in_path.joinpath(tar_folder).is_dir():
            continue
        else:
            print(f"generating assist dataset: {tar_folder}")
            in_path.joinpath(tar_folder).mkdir(parents=True, exist_ok=True)
            func = eval(funcstr)
            for i in range(len(in_dataset)):
                filename = in_dataset.data_list[i].name
                with torch.no_grad():
                    temp = func(in_dataset[i].unsqueeze(0).to(device))
                Image.fromarray(torch.clamp(temp.squeeze() * 255, max=255, min=0).cpu().numpy().astype(np.uint8)).save(
                    in_path.joinpath(tar_folder, f'{filename}').absolute())


class Get_exptar(nn.Module):
    def __init__(self, E_min=0.5, E_max=1):
        super(Get_exptar, self).__init__()
        self.transmation = linear_transmation(max_tar=E_max, min_tar=E_min)

    def forward(self, in_img):
        x = cal_huidu(in_img)
        target = self.transmation(x)
        return target

class Get_adaptivesmoothtar(nn.Module):
    def __init__(self, filter_size=5, threshold_low=0.1, threshold_up=0.5, k=1.5, device='cuda'):
        super(Get_adaptivesmoothtar, self).__init__()
        self.canny_tidu = CannyDetector(filter_size=filter_size, device=device, mode='tidu')
        self.threshold_low = threshold_low
        self.threshold_up = threshold_up
        self.k = k
        self.canny_bianyuan = CannyDetector(filter_size=filter_size, device=device, mode='thickbianyuan')

    def forward(self, in_img):
        target_mask = self.canny_bianyuan(in_img, low_threshold=self.threshold_low, high_threshold=self.threshold_up)
        target_temp = self.canny_tidu(in_img)
        # divide 16.971 to normalize and save
        target = target_temp * target_mask * self.k / 16.971
        target = torch.clamp(target, max=1)
        return target

class linear_transmation(nn.Module):
    def __init__(self, max_tar, min_tar):
        super(linear_transmation, self).__init__()
        self.max_tar = max_tar
        self.min_tar = min_tar

    def forward(self, img):
        maxV = img.max()
        minV = img.min()
        return (img-minV)*(self.max_tar-self.min_tar)/(maxV-minV)+self.min_tar

if __name__ == '__main__':
    pass





