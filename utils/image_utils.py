#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import colour

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image_acescg_tensor = render_pkg["render"] # This is (C,H,W), ACEScg, on CUDA

        # Convert ACEScg Tensor to sRGB Tensor for display
        # 1. Detach, CPU, permute to H,W,C, NumPy
        img_acescg_hwc_np = net_image_acescg_tensor.detach().cpu().permute(1, 2, 0).numpy()

        # 2. Convert ACEScg to sRGB
        img_srgb_np = colour.RGB_to_RGB(img_acescg_hwc_np,
                                        colour.RGB_COLOURSPACES["ACEScg"],
                                        colour.RGB_COLOURSPACES["sRGB"])

        # 3. Clip
        img_srgb_np_clipped = np.clip(img_srgb_np, 0.0, 1.0)

        # 4. Convert back to PyTorch tensor (C,H,W) and original device
        net_image = torch.from_numpy(img_srgb_np_clipped).permute(2, 0, 1).to(net_image_acescg_tensor.device)

    if net_image.shape[0]==1 and output != 'alpha': # Alpha is already single channel, no colormap
        # Colormap expects single channel input, ensure it's not already 3-channel sRGB.
        # This path is typically for depth, normal (after processing), edge, curvature.
        # These should not go through the ACEScg->sRGB conversion above if they are not color images.
        # The logic above for 'else' branch specifically targets the 'render' (color) output.
        # So, if net_image here is single channel, it wasn't the 'render' output.
        net_image = colormap(net_image)
    return net_image