import os
from torch.autograd import Variable
from scipy.misc import imread, imsave
from student_net import stu_net
from args_fusion import args
import numpy as np
import torch

def load_model(path, input_nc, output_nc):
    nest_model = stu_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))
    total_param = 0
    print("MODEL DETAILS:\n")
    print(nest_model)
    for param in nest_model.parameters():
        print(param.dtype)
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', nest_model._get_name(), total_param)

    bytes_per_param = 4
    total_bytes = total_param * bytes_per_param
    total_megabytes = total_bytes / (1024 * 1024)
    total_kilobytes = total_bytes / 1024

    print("Total parameters in MB:", total_megabytes)
    print("Total parameters in KB:", total_kilobytes)

    nest_model.eval()

    return nest_model


def run_demo(model, infrared_path, visible_path, output_path_root, index,mode):
    ir_img = imread(infrared_path, mode=mode);
    vi_img = imread(visible_path, mode=mode);
    ir_img = ir_img / 255.0;
    vi_img = vi_img / 255.0;
    ir_img_patches = [[ir_img]]
    vi_img_patches = [[vi_img]]


    ir_img_patches = np.stack(ir_img_patches, axis=0);
    vi_img_patches = np.stack(vi_img_patches, axis=0);
    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);

    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        model = model.cuda(args.device);
    ir_img_patches = Variable(ir_img_patches, requires_grad=False)
    vi_img_patches = Variable(vi_img_patches, requires_grad=False)

    img = torch.cat([ir_img_patches, vi_img_patches], 1);
    img = img.float()

    en = model.encoder(img)
    outs = model.decoder(en);
    out = (outs[0][0][0]).detach().cpu().numpy();
    # ########################### save fused images ##############################################
    file_name = str(index) + '.png'
    output_path = output_path_root + file_name
    imsave(output_path, out);
    return

def main():
    test_path = args.test_imgs_path
    output_path = args.save_outputs_path
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path = args.pretrained_student_path
    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        for i in range(0,args.test_nums):
            index = i + 1
            infrared_path = test_path + 'IR/' + str(index) + '.png'
            visible_path = test_path + 'VIS/'+ str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, mode)
    print('Done......')
if __name__ == '__main__':
    main()
