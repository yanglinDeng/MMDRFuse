import os
import time
import numpy as np
from tqdm import trange
import scipy.io as scio
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import torchvision.models as models
from student_net import stu_net
from teacher_net import tea_net
from args_fusion import args
from utils import gradient2,gmsd,ssim,showLossChart
import utils

def main():
    train()

def train():
    vgg_model = models.vgg19(pretrained=True)
    if (args.cuda):
        vgg_model = vgg_model.cuda(args.device);
    vggFeatures = [];
    vggFeatures.append(vgg_model.features[:4]);
    vggFeatures.append(vgg_model.features[:9]);
    vggFeatures.append(vgg_model.features[:18]);
    vggFeatures.append(vgg_model.features[:27]);
    vggFeatures.append(vgg_model.features[:36]);

    for i in range(0, 5):
        for parm in vggFeatures[i].parameters():
            parm.requires_grad = False;

    save_stumodel_dir = args.save_stumodel_dir
    save_stuloss_dir = args.save_stuloss_dir
    patchPrePath = args.Patch_path;
    PatchPaths = utils.loadPatchesPairPaths()
    batch_size = args.batch_size

    # load network model
    stu_model = stu_net(args.in_c, args.out_c)
    if (args.cuda):
        stu_model.cuda(int(args.device));
    print(stu_model)
    optimizer = Adam(filter(lambda p: p.requires_grad, stu_model.parameters()), args.lr)

    mse_loss = torch.nn.MSELoss(reduction="mean")
    if (args.cuda):
        mse_loss.cuda(int(args.device));
    teacher_path = args.pretrained_teacher_path
    teacher_model = tea_net(args.in_c, args.out_c)
    if (args.cuda):
        teacher_model.cuda(int(args.device));
    teacher_model.load_state_dict(torch.load(teacher_path))
    stu_model.conv2.load_state_dict(teacher_model.conv00.state_dict())

    tbar = trange(args.epochs)

    Loss_content = []
    Loss_memory = []
    Loss_distill = []
    Loss_total = []
    all_content_loss = 0.
    all_memory_loss = 0.
    all_distill_loss = 0.
    all_total_loss = 0.

    bs_ar = np.zeros((args.trainNumber, 1, args.HEIGHT, args.WIDTH))
    bg_ar = np.zeros((args.trainNumber, 1, args.HEIGHT, args.WIDTH))
    label_ssim = torch.zeros((args.batch_size, 1, args.HEIGHT, args.WIDTH))
    label_fsim = torch.zeros((args.batch_size, 1, args.HEIGHT, args.WIDTH))
    if (args.cuda):
        label_ssim = label_ssim.cuda(args.device)
        label_fsim = label_fsim.cuda(args.device)
    label_ssim.requires_grad = False;
    label_fsim.requires_grad = False;


    print('Start training.....')
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        patchesPaths, batches = utils.load_datasetPair(PatchPaths, batch_size);
        stu_model.train()
        count = 0

        for batch in range(batches):
            optimizer.zero_grad()
            image_paths = patchesPaths[batch * batch_size:(batch * batch_size + batch_size)]  # 取每个批次训练的图像

            # load image patches of this batch.
            image_ir = utils.get_train_images_auto(patchPrePath + "/IR", image_paths, mode="L");
            image_vi = utils.get_train_images_auto(patchPrePath + "/VIS", image_paths, mode="L");
            count += 1
            img_ir = Variable(image_ir, requires_grad=False)
            img_vi = Variable(image_vi, requires_grad=False)
            # outputs of the teacher network
            if args.cuda:
                img_ir = img_ir.cuda(args.device)
                img_vi = img_vi.cuda(args.device)
            en_teacher = teacher_model.encoder(torch.cat([img_ir, img_vi], 1));
            outputs_teacher = teacher_model.decoder(en_teacher)

            #outputs of the student network
            en_stu = stu_model.encoder(torch.cat([img_ir, img_vi], 1));
            outputs_stu = stu_model.decoder(en_stu)
            output_stu = outputs_stu[0];

            # perception loss
            img_irdup = torch.cat([img_ir, img_ir, img_ir], 1);
            img_vidup = torch.cat([img_vi, img_vi, img_vi], 1);
            img_outdup = torch.cat([output_stu, output_stu, output_stu], 1);
            perception_loss = 0.
            for j in range(5):
                g_ir = vggFeatures[j](img_irdup);
                g_vi = vggFeatures[j](img_vidup);
                g_output = vggFeatures[j](img_outdup);
                perception_loss += mse_loss(torch.max(g_ir, g_vi), g_output);
            perception_loss = perception_loss.div(5);


            # gradient loss
            grad_loss_value = 0.
            grad_loss_value += mse_loss(gradient2(output_stu), gradient2(torch.max(img_vi,img_ir)));
            grad_loss_value /= 10;

            # intensity loss
            x_in_max = torch.max(img_vi, img_ir);
            loss_intensity = F.l1_loss(x_in_max, output_stu);

            L_content = loss_intensity+ 2*perception_loss+grad_loss_value

            outputCopy = output_stu.cpu().detach().numpy();

            # dynamic refresh strategy
            ssim_cur = ssim(output_stu, img_ir, size_average=True).data.item() + ssim(output_stu, img_vi,size_average=True).data.item()
            fsim_cur = gmsd(output_stu, img_ir) + gmsd(output_stu, img_vi)
            if e == 0:
                for j, path in enumerate(image_paths):
                    bs_ar[int(path)] = outputCopy[j]
                    bg_ar[int(path)] = outputCopy[j]
                ssim_best = ssim_cur
                fsim_best = fsim_cur
            else:
                for j, path in enumerate(image_paths):
                    label_ssim[j] = torch.from_numpy(bs_ar[int(path)]).float()
                for j, path in enumerate(image_paths):
                    label_fsim[j] = torch.from_numpy(bg_ar[int(path)]).float()

                ssim_best = ssim(label_ssim, img_ir, size_average=True).data.item() + ssim(label_ssim, img_vi,
                                                                                           size_average=True).data.item()
                fsim_best = gmsd(label_fsim, img_ir) + gmsd(label_fsim, img_vi)

            print("\nssim", ssim_cur, ssim_best)
            print("fsim", fsim_cur, fsim_best)

            memory_ssimloss = 0.
            gap_ssim = 1
            if (ssim_cur > ssim_best):
                for j, path in enumerate(image_paths):
                    bs_ar[int(path)] = outputCopy[j]

            elif ssim_cur < ssim_best:
                gap_ssim = ssim_best - ssim_cur
                print("The gap of ssim between current output and history output is{}".format(gap_ssim))
                img_lastssim = torch.cat([label_ssim, label_ssim, label_ssim], 1);
                img_curssim = torch.cat([output_stu, output_stu, output_stu], 1);
                for j in range(3, 5):
                    g_last = vggFeatures[j](img_lastssim);
                    g_out = vggFeatures[j](img_curssim);
                    memory_ssimloss += mse_loss(g_last, g_out);
                memory_ssimloss = memory_ssimloss.div(2);
                memory_ssimloss += F.l1_loss(label_ssim, output_stu)

            memory_fsimloss = 0.
            gap_fsim = 1
            if fsim_cur < fsim_best:
                for j, path in enumerate(image_paths):
                    bg_ar[int(path)] = outputCopy[j]
            elif fsim_cur > fsim_best:
                gap_fsim = (fsim_cur - fsim_best);
                print("The gap of gmsd between current output and history output is{}".format(gap_fsim))
                img_lastdup = torch.cat([label_fsim, label_fsim, label_fsim], 1);
                img_curdup = torch.cat([output_stu, output_stu, output_stu], 1);
                for j in range(3):
                    g_last = vggFeatures[j](img_lastdup);
                    g_out = vggFeatures[j](img_curdup);
                    memory_fsimloss += mse_loss(g_last, g_out);
                memory_fsimloss = memory_fsimloss.div(3);
                memory_fsimloss += (mse_loss(gradient2(output_stu), gradient2(label_fsim)));

            w1 = gap_ssim * 10
            w2 = gap_fsim * 10
            L_memory = w2 * memory_fsimloss + w1 * memory_ssimloss

            stu = [en_stu[0],outputs_stu[0]]
            tea = [outputs_teacher[1],outputs_teacher[0]]
            distill_loss = utils.distillation(stu,tea)
            distill_loss = distill_loss*5

            total_loss = L_content + L_memory + distill_loss;

            total_loss.backward()
            optimizer.step()


            all_memory_loss += L_memory
            all_content_loss += L_content
            all_distill_loss += distill_loss
            all_total_loss += total_loss

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\t memory: {:.6f} \t distill: {:6f}\t total:{:6f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_content_loss / args.log_interval,
                                  all_memory_loss / args.log_interval,
                                  all_distill_loss / args.log_interval,
                                  all_total_loss / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_content.append(all_content_loss / args.log_interval);
                Loss_memory.append(all_memory_loss / args.log_interval);
                Loss_distill.append(all_distill_loss / args.log_interval);
                Loss_total.append(all_total_loss / args.log_interval);
                all_content_loss = 0.
                all_memory_loss = 0.
                all_distill_loss = 0.
                all_total_loss = 0.

            if (batch + 1) % (100) == 0:
                # save model.py
                stu_model.eval()
                stu_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(save_stumodel_dir, save_model_filename)
                torch.save(stu_model.state_dict(), save_model_path)
                # save loss data
                # Lcontent loss

                loss_data_content = torch.tensor(Loss_content).data.cpu().numpy()
                loss_filename_path = "loss_content_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(args.save_stuloss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_content})
                showLossChart(save_loss_path, save_stuloss_dir + '/content.png')

                # Ldistill loss
                loss_data_distill = torch.tensor(Loss_distill).data.cpu().numpy()
                loss_filename_path = "loss_memory_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_distill})
                showLossChart(save_loss_path, save_stuloss_dir + '/distill.png')

                #Lmemory loss
                loss_data_memory = torch.tensor(Loss_memory).data.cpu().numpy()
                loss_filename_path = "loss_memory_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_memory})
                showLossChart(save_loss_path, save_stuloss_dir + '/memory.png')

                # Ltotal loss
                loss_data_total = torch.tensor(Loss_total).data.cpu().numpy()
                loss_filename_path = "loss_memory_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                              '_') + ".mat"
                save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_total})
                showLossChart(save_loss_path, save_stuloss_dir + '/total.png')

                stu_model.train()
                if (args.cuda):
                    stu_model.cuda(int(args.device));
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

    # Lcontent loss
    loss_data_content = torch.tensor(Loss_content).data.cpu().numpy()
    loss_filename_path = "Final_loss_content_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
    save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_content})
    showLossChart(save_loss_path, save_stuloss_dir + "/content.png");
    # Ldistill loss
    loss_data_distill = torch.tensor(Loss_distill).data.cpu().numpy()
    loss_filename_path = "loss_memory_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                  '_') + ".mat"
    save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_distill})
    showLossChart(save_loss_path, save_stuloss_dir + '/distill.png')
    # Lmemory loss
    loss_data_memory = torch.tensor(Loss_memory).data.cpu().numpy()
    loss_filename_path = "loss_memory_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                  '_') + ".mat"
    save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_memory})
    showLossChart(save_loss_path, save_stuloss_dir + '/memory.png')

    # Ltotal loss
    loss_data_total = torch.tensor(Loss_total).data.cpu().numpy()
    loss_filename_path = "loss_memory_epoch_" + str(
        args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                  '_') + ".mat"
    save_loss_path = os.path.join(save_stuloss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_total})
    showLossChart(save_loss_path, save_stuloss_dir + '/total.png')

    # save model
    stu_model.eval()
    stu_model.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
    save_model_path = os.path.join(save_stuloss_dir, save_model_filename)
    torch.save(stu_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
    main()


