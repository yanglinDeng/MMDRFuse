from os import listdir
from os.path import join
import random
from torch import nn
from PIL import Image
from args_fusion import args
from scipy.misc import imread
from torchvision import transforms
from typing import Type, Union
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn.functional as F
import numpy as np
from math import exp

mse_value = torch.nn.MSELoss(reduction="sum")
if torch.cuda.is_available():
    mse_value= mse_value.cuda(args.device)

def showLossChart(path,savedName):
    plt.cla();
    plt.clf();
    if (path == ""):
        return;
    data = scio.loadmat(path)
    loss =data['Loss'][0];

    x_data = range(0,len(loss));
    y_data = loss;

    plt.plot(x_data,y_data);
    plt.xlabel("Step");
    plt.ylabel("Loss");
    plt.savefig(savedName);

def distillation(stu,tea):
    res = 0.
    for i in range(len(stu)):
        stu_new = torch.sum(stu[i],dim=1)
        tea_new = torch.sum(tea[i],dim=1)
        # print(stu_new.shape,tea_new.shape)
        stu_ord = torch.norm(stu_new,p=2)
        tea_ord = torch.norm(tea_new,p=2)
        if torch.cuda.is_available():
            stu_new = stu_new.cuda(args.device)
            tea_new = tea_new.cuda(args.device)
            stu_ord = stu_ord.cuda(args.device)
            tea_ord = tea_ord.cuda(args.device)
        print(stu_new.shape,tea_new.shape)
        res+=mse_value(stu_new/stu_ord,tea_new/tea_ord)
    return res/2

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def gmsd_value(dis_img:Type[Union[torch.Tensor,np.ndarray]],ref_img:Type[Union[torch.Tensor,np.ndarray]],c=170,device='cuda'):

    dis_img = dis_img.unsqueeze(0).unsqueeze(0)
    ref_img = ref_img.unsqueeze(0).unsqueeze(0)

    if torch.max(dis_img) <= 1:
        dis_img = dis_img * 255
    if torch.max(ref_img) <= 1:
        ref_img = ref_img * 255

    hx=torch.tensor([[1/3,0,-1/3]]*3,dtype=torch.float).unsqueeze(0).unsqueeze(0)#Prewitt算子
    if (args.cuda):
        hx = hx.cuda(int(args.device));
    ave_filter=torch.tensor([[0.25,0.25],[0.25,0.25]],dtype=torch.float).unsqueeze(0).unsqueeze(0)#均值滤波核
    if (args.cuda):
        ave_filter = ave_filter.cuda(int(args.device));
    down_step=2#下采样间隔
    hy=hx.transpose(2,3)

    dis_img=dis_img.float()
    if (args.cuda):
        dis_img = dis_img.cuda(int(args.device));
    ref_img=ref_img.float()
    if (args.cuda):
        ref_img = ref_img.cuda(int(args.device));


    ave_dis=F.conv2d(dis_img,ave_filter,stride=1)
    ave_ref=F.conv2d(ref_img,ave_filter,stride=1)

    ave_dis_down=ave_dis[:,:,0::down_step,0::down_step]
    ave_ref_down=ave_ref[:,:,0::down_step,0::down_step]

    mr_sq=F.conv2d(ave_ref_down,hx)**2+F.conv2d(ave_ref_down,hy)**2
    md_sq=F.conv2d(ave_dis_down,hx)**2+F.conv2d(ave_dis_down,hy)**2
    mr=torch.sqrt(mr_sq)
    md=torch.sqrt(md_sq)
    GMS=(2*mr*md+c)/(mr_sq+md_sq+c)
    GMSD=torch.std(GMS.view(-1))
    return GMSD.item()
#
def gmsd(img1,img2):
    num = img1.shape[0]
    value = 0.
    for i in range(num):
        img_ref = img1[i,0,:,:]
        img_dis = img2[i,0,:,:]
        value += gmsd_value(img_dis,img_ref)
    return value/num



def gradient(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    # kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1);
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=1);
    #showTensor(gradMap);
    return gradMap;     
    
def gradient2(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    #kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1);
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=1);
    #showTensor(gradMap);
    return gradMap;         

def loadPatchesPairPaths2(directory):
    imagePatchesIR = [];
    imagePatchesVIS = [];
    for i in range(0+1,args.trainNumber+1):
        irPatchPath = directory+"/IR/"+str(i)+".png";
        visPatchPath = directory+"/VIS_gray/"+str(i)+".png";
        imagePatchesIR.append(irPatchPath);
        imagePatchesVIS.append(visPatchPath);
    return imagePatchesIR,imagePatchesVIS;
    
def loadPatchesPairPaths():
    imagePatches = [];
    for i in range(0+1,args.trainNumber+1):
        imagePatches.append(str(i));
    return imagePatches;    

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])

    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()

def load_datasetPair(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        image_path = image_path[:-mod];    
    num_imgs-=mod
    original_img_path = image_path[:num_imgs]

    # random
    random.shuffle(original_img_path)
    batches = int(len(original_img_path) // BATCH_SIZE)
    return original_img_path, batches

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    image = image/255;
    return image


def get_train_images_auto2(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_train_images_auto(pre, paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(pre+"/"+path+".png", height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            # test_rgb = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
