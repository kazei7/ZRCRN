import torch
import torch.nn as nn

# 输入4d的rgb_tensor，输出4d的灰度矩阵
def cal_huidu(x, mode='perception'):
    if mode == 'mean':
        return torch.mean(x, dim=1)
    elif mode == 'perception':
        return (0.299*x[:, 0, :, :] + 0.587*x[:, 1, :, :] + 0.114*x[:, 2, :, :]).unsqueeze(1)

class L_adaptivesmooth(nn.Module):
    def __init__(self, filter_size=5, device='cuda'):
        super(L_adaptivesmooth, self).__init__()
        self.canny_tidu = CannyDetector(filter_size=filter_size, device=device, mode='tidu')

    def forward(self, P1, tar):
        D_enhance_all_new = self.canny_tidu(P1)
        # read from file and mult 16.971 to recover exact target
        E = torch.pow(D_enhance_all_new-(16.971*tar), 2)
        return torch.mean(E)

class L_exp(nn.Module):
    def __init__(self):
        super(L_exp, self).__init__()

    def forward(self, x, target):
        mean = cal_huidu(x)
        temp = mean - target
        d = torch.mean(torch.pow(temp, 2))
        return d

class L_exp_const(L_exp):
    def __init__(self, want_exposure=0.6):
        super(L_exp_const, self).__init__()
        self.want_exposure = want_exposure

    def forward(self, x):
        mean = cal_huidu(x)
        temp = mean - torch.FloatTensor([self.want_exposure]).cuda()
        d = torch.mean(torch.pow(temp, 2))
        return d


import math
from scipy.signal.windows import gaussian
import numpy as np
# 参考项目：https://github.com/jm12138/CannyDetector/tree/main
class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1, device='cpu', mode='bianyuan'):
        super(CannyDetector, self).__init__()
        # 配置运行设备
        self.device = device
        self.filter_size = filter_size

        # 高斯滤波器
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2), bias=False).to(self.device)
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False).to(self.device)



        # Sobel 滤波器
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False).to(self.device)
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False).to(self.device)

        # 定向滤波器
        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False).to(self.device)

        # 连通滤波器
        self.connect_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False).to(self.device)

        # 初始化参数
        params = get_state_dict(filter_size=filter_size, std=std, map_func=lambda x:torch.from_numpy(x).to(self.device))
        self.load_state_dict(params)

        for key in params.keys():
            getattr(self, key.split('.')[0]).weight.requires_grad = True

        if mode == 'bianyuan':
            self.forward = self.forward_bianyuan
        elif mode == 'tidu':
            self.forward = self.forward_tidu
        elif mode == 'mohutidu':
            self.forward = self.forward_mohutidu
        elif mode == 'shuangyuzhi':
            self.forward = self.forward_shuangyuzhi
        elif mode == 'thickbianyuan':
            self.forward = self.forward_thickbianyuan


    @torch.no_grad()
    def forward_bianyuan(self, img, low_threshold=10.0, high_threshold=100.0):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        grad_orientation += 180.0
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)

        indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: 双阈值
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        return thresholded

    @torch.no_grad()
    def forward_thickbianyuan(self, img, low_threshold=10.0, high_threshold=100.0):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        # grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        # grad_orientation += 180.0
        # grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0

        thin_edges = grad_mag

        # Step4: 双阈值
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()

        return thresholded


    @torch.no_grad()
    def forward_mohutidu(self, img):
        # 拆分图像通道
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        # grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/math.pi))
        # grad_orientation += 180.0
        # grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0
        #
        # # Step3: 非最大抑制，边缘细化
        # all_filtered = self.directional_filter(grad_mag)
        #
        # inidices_positive = (grad_orientation / 45) % 8
        # inidices_negative = ((grad_orientation / 45) + 4) % 8
        #
        # batch, _, height, width = inidices_positive.shape
        # pixel_count = height * width * batch
        # pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)
        #
        # indices = (inidices_positive.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_positive = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))
        #
        # indices = (inidices_negative.reshape((-1, )) * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_negative = all_filtered.reshape((-1, ))[indices.long()].reshape((batch, 1, height, width))
        #
        # channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])
        #
        # is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        #
        # thin_edges = grad_mag.clone()
        # thin_edges[is_max==0] = 0.0

        return grad_mag


    def forward_tidu(self, img):
        # 拆分图像通道
        blurred_img_r = img[:, 0:1]  # red channel
        blurred_img_g = img[:, 1:2]  # green channel
        blurred_img_b = img[:, 2:3]  # blue channel

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        # torch.sqrt的输入参数+1e-8避免在backward时出现梯度爆炸问题
        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2 + 1e-8)
        grad_mag = grad_mag + torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2 + 1e-8)
        grad_mag = grad_mag + torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2 + 1e-8)
        # grad_orientation = (
        #             torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (180.0 / math.pi))
        # grad_orientation += 180.0
        # grad_orientation = torch.round(grad_orientation / 45.0) * 45.0
        #
        # # Step3: 非最大抑制，边缘细化
        # all_filtered = self.directional_filter(grad_mag)
        #
        # inidices_positive = (grad_orientation / 45) % 8
        # inidices_negative = ((grad_orientation / 45) + 4) % 8
        #
        # batch, _, height, width = inidices_positive.shape
        # pixel_count = height * width * batch
        # pixel_range = torch.Tensor([range(pixel_count)]).to(self.device)
        #
        # indices = (inidices_positive.reshape((-1,)) * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_positive = all_filtered.reshape((-1,))[indices.long()].reshape(
        #     (batch, 1, height, width))
        #
        # indices = (inidices_negative.reshape((-1,)) * pixel_count + pixel_range).squeeze()
        # channel_select_filtered_negative = all_filtered.reshape((-1,))[indices.long()].reshape(
        #     (batch, 1, height, width))
        #
        # channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])
        #
        # is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        #
        # thin_edges = grad_mag.clone()
        # thin_edges[is_max == 0] = 0.0

        return grad_mag

    def forward_shuangyuzhi(self, thin_edges, low_threshold=10.0, high_threshold=100.0):
        # Step4: 双阈值
        thresholded = thin_edges.clone()
        lower = thin_edges<low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges>high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(thin_edges>=low_threshold, thin_edges<=high_threshold)
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map>0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded>0.0).float()
        return thresholded

def get_state_dict(filter_size=5, std=1.0, map_func=lambda x:x, only_gaussian=False):
    generated_filters = gaussian(filter_size, std=std).reshape([1, filter_size
                                                   ]).astype(np.float32)

    generated_filters = generated_filters/generated_filters.sum()


    gaussian_filter_horizontal = generated_filters[None, None, ...]

    gaussian_filter_vertical = generated_filters.T[None, None, ...]

    sobel_filter_horizontal = np.array([[[
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]]]],
        dtype='float32'
    )

    sobel_filter_vertical = np.array([[[
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]]]],
        dtype='float32'
    )

    directional_filter = np.array(
        [[[[ 0.,  0.,  0.],
          [ 0.,  1., -1.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0., -1.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0., -1.,  0.]]],


        [[[ 0.,  0.,  0.],
          [ 0.,  1.,  0.],
          [-1.,  0.,  0.]]],


        [[[ 0.,  0.,  0.],
          [-1.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[-1.,  0.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0., -1.,  0.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]],


        [[[ 0.,  0., -1.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  0.]]]],
        dtype=np.float32
    )

    connect_filter = np.array([[[
        [1., 1., 1.],
        [1., 0., 1.],
        [1., 1., 1.]]]],
        dtype=np.float32
    )

    if only_gaussian:
        return {
        'gaussian_filter_horizontal.weight': map_func(gaussian_filter_horizontal),
        'gaussian_filter_vertical.weight': map_func(gaussian_filter_vertical)
    }
    else:
        return {
        'gaussian_filter_horizontal.weight': map_func(gaussian_filter_horizontal),
        'gaussian_filter_vertical.weight': map_func(gaussian_filter_vertical),
        'sobel_filter_horizontal.weight': map_func(sobel_filter_horizontal),
        'sobel_filter_vertical.weight': map_func(sobel_filter_vertical),
        'directional_filter.weight': map_func(directional_filter),
        'connect_filter.weight': map_func(connect_filter),
    }

class My_loss(nn.Module):
    def __init__(self, hparams={}):
        super(My_loss, self).__init__()
        hparams.setdefault('auto_multi_task_loss', 1)
        hparams.setdefault('lr_exp', 1)
        hparams.setdefault('lr_adaptivesmooth', 1)
        hparams.setdefault('lr_exp_const', 0)
        self.hparams = hparams

        all_loss_dict = {
            'lr_exp': ["L_exp()", "P1, exptar"],
            'lr_adaptivesmooth': ["L_adaptivesmooth()",\
                                  "P1, adaptivesmoothtar"],
            'lr_exp_const': ["L_exp_const(want_exposure=0.6)", "P1"],
        }

        self.all_loss = []
        self.all_input = []
        self.all_loss_name = []
        for loss, func in all_loss_dict.items():
            if loss in hparams:
                pass
            else:
                hparams[loss]=0
            if hparams[loss] != 0:
                self.all_loss.append(eval(func[0]))
                self.all_input.append("(" + func[1] + ")")
                self.all_loss_name.append(loss)

        calculation = ''
        if self.hparams['auto_multi_task_loss']:
            for i in self.all_loss_name:
                if calculation:
                    calculation = f"{calculation}, {i}"
                else:
                    calculation = f"[{i}"
            print(f"train_loss will include: {calculation}] ,weights will be auto calculated")
        else:
            for i in self.all_loss_name:
                if calculation:
                    calculation = calculation + ' + ' + str(self.hparams[i]) + '*' + i
                else:
                    calculation = str(self.hparams[i]) + '*' + i
            print(f"train_loss will be: {calculation}")


    def forward(self, P1=None, exptar=None, adaptivesmoothtar=None, logvars=None):
        loss = 0
        count = 0
        for func, input, loss_name in zip(self.all_loss, self.all_input, self.all_loss_name):
            if self.hparams['auto_multi_task_loss']:
                temp = eval("func" + input)
                if temp.isnan():
                    print(f"result for {loss_name} is nan, neglected!")
                else:
                    autoweight = torch.exp(-logvars[count])
                    loss += (temp * autoweight + logvars[count])
            else:
                temp = eval("func" + input)
                if temp.isnan():
                    print(f"result for {loss_name} is nan, neglected!")
                else:
                    loss += (temp * self.hparams[loss_name])
            count = count + 1
        return loss

if __name__ == "__main__":
    pass




