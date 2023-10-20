import csv
import os.path
import statistics

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import torch
import codecs
import lpips


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    # display_interval = 1 打印损失的频次
    if itr % configs.display_interval == 0:
        print('itr: ' + str(itr),
              'training L1 loss: ' + str(loss_l1), 'training L2 loss: ' + str(loss_l2))


def test(data, img_gen, configs, flag, itr):
    # gen_frm_dir = results/mau/
    res_path = configs.gen_frm_dir + '/' + str("testf")
    batch_id = 'front_'+ str(flag)  + '_' +str(itr)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    #batch_size = data.shape[0]

    #img_gen = model.testB(data, real_input_flag)
    # img_gen = 16 * 19 * 1 * 64 * 64 -> 16 * 19 * 64 * 64 * 1
    img_gen = img_gen.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
    # data = 16 * 20 * 1 * 64 * 64 -> 16 * 20 * 64 * 64 * 1 = test_ims
    # test_ims =  16 * 20 * 64 * 64 * 1
    test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
    # output_length = total_length(20) - input_length(10) = 10
    output_length = configs.total_length - configs.input_length
    # output_length = min(10,19) = 10
    output_length = min(output_length, configs.total_length - 1)
    # test_ims = 16 * 20 * 64 * 64 * 1, patch_size = 1
    # 输出是:  test_ims = 16 * 20 * 64 * 64 * 1
    test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
    # 输出是:  img_gen = 16 * 19 * 64 * 64 * 1
    img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
    # img_out = img_gen[:,-10,:] = 16 * 10 * 64 * 64 * 1
    img_out = img_gen[:, -output_length:, :]

    # res_width = 64
    res_width = configs.img_width
    # res_height = 64
    res_height = configs.img_height
    # img = (64 * 2 , 20 * 64, 1)
    interval = 4
    img = np.ones((2 * res_height,
                   configs.total_length * res_width,
                   configs.img_channel))
    img_input = np.ones((res_height,
                         configs.input_length * res_width + configs.input_length * interval,
                         configs.img_channel))
    img_ground_true = np.ones((res_height,
                               configs.pred_length * res_width + configs.pred_length * interval,
                               configs.img_channel))
    img_pred = np.ones((res_height,
                        configs.pred_length * res_width + configs.pred_length * interval,
                        configs.img_channel))
    if configs.is_training == True and configs.dataset == 'kth':
        img_input = np.ones((res_height,
                             (configs.input_length//2) * res_width + (configs.input_length//2) * interval,
                             configs.img_channel))

        img_ground_true = np.ones((res_height,
                                   (configs.pred_length//2) * res_width + (configs.pred_length//2) * interval,
                                   configs.img_channel))
        img_pred = np.ones((res_height,
                            (configs.pred_length//2) * res_width +(configs.pred_length//2)  * interval,
                            configs.img_channel))
    # name = 1.png
    name = str(batch_id) + '.png'

    img_input_name = str(batch_id) + '_input.png'
    img_ground_true_name = str(batch_id) + '_ground_true.png'
    img_pred_name = str(batch_id) + '_pred.png'

    # file_name = results/mau/1.png
    file_name = os.path.join(res_path, name)

    file_img_input_name = os.path.join(res_path, img_input_name)
    file_img_ground_true_name = os.path.join(res_path, img_ground_true_name)
    file_img_pred_name = os.path.join(res_path, img_pred_name)

    img_input_single = np.ones((res_height, res_width, configs.img_channel))
    img_ground_true_single = np.ones((res_height, res_width, configs.img_channel))
    img_pred_single = np.ones((res_height, res_width, configs.img_channel))

    # total_length = 20 | 0,1,2,3,...,17,18,19
    for i in range(configs.total_length - 1):
        # img[:res_height, i * res_width:(i + 1) * res_width, :]
        # = img[:res_height, i * res_width:(i + 1) * res_width, :]
        # = img[:64,1*64:2*64,:] = test_ims[0, 1, :]
        img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :]

        if i < configs.input_length:
            if configs.is_training == True and configs.dataset == 'kth':
                if i % 2 != 0:
                    continue
                else:
                    img_input[:res_height, ((i // 2) * res_width + (i // 2) * interval):(((i // 2) + 1) * res_width + (i // 2) * interval), :] = test_ims[0, i, :]
            else:
                img_input[:res_height, (i * res_width + i * interval):((i + 1) * res_width + i * interval),:] = test_ims[0, i, :]
            if configs.img_channel == 2:
                img_input_single[:, :, :] = test_ims[0, i, :]
                img_total_input_single = img_input_single[:, :, 0] + img_input_single[:, :, 1]
                img_input_name_single = 'batch_' + str(batch_id) + '_input_' + str(i) + '.svg'
                file_img_input_name_single_svg = os.path.join(res_path, img_input_name_single)
                plt.imsave(file_img_input_name_single_svg,img_total_input_single.reshape(img_total_input_single.shape[0],img_total_input_single.shape[1]), vmin=0, vmax=1.0)
        else:
            if configs.is_training == True and configs.dataset == 'kth':
                if (i - configs.input_length) % 2 != 0:
                    continue
                else:
                    img_ground_true[:res_height, (((i - configs.input_length) // 2) * res_width + ((i - configs.input_length) // 2) * interval):((((i - configs.input_length) // 2) + 1) * res_width + ((i - configs.input_length) // 2) * interval),:] = test_ims[0, i, :]
            else:
                img_ground_true[:res_height,((i - configs.input_length) * res_width + (i - configs.input_length) * interval):((i + 1 - configs.input_length) * res_width + (i - configs.input_length) * interval),:] = test_ims[0, i, :]
            if configs.img_channel == 2:
                img_ground_true_single[:, :, :] = test_ims[0, i, :]
                img_total_ground_true_single = img_ground_true_single[:, :, 0] + img_ground_true_single[:, :, 1]
                img_ground_true_name_single = 'batch_' + str(batch_id) + '_ground_true_' + str(i - configs.input_length) + '.svg'
                file_img_ground_true_name_single_svg = os.path.join(res_path, img_ground_true_name_single)
                plt.imsave(file_img_ground_true_name_single_svg,img_total_ground_true_single.reshape(img_total_ground_true_single.shape[0],img_total_ground_true_single.shape[1]), vmin=0,vmax=1.0)

                img_pred_single[:, :, :] = img_out[0, -output_length + (i - configs.input_length), :]
                img_total_pred_single = img_pred_single[:, :, 0] + img_pred_single[:, :, 1]

                img_total_target_pred_diff_single = img_total_ground_true_single[:, :] - img_total_pred_single[:, :]
                img_target_pred_diff_name_single = 'batch_' + str(batch_id) + '_target_pred_diff_' + str(i - configs.input_length) + '.svg'
                file_img_target_pred_diff_name_single_svg = os.path.join(res_path,img_target_pred_diff_name_single)
                plt.imsave(file_img_target_pred_diff_name_single_svg,img_total_target_pred_diff_single.reshape(img_total_target_pred_diff_single.shape[0],img_total_target_pred_diff_single.shape[1]),vmin=0, vmax=1.0)
    # total_length = 10 | 0,1,2,3,...,7,8,9
    for i in range(output_length):
        # img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:]
        # = img[64:, (10 + 1) * 64:(10 + 1 + 1) * 64,:] = img_out[0, -10 + 1, :] = img_out[0, -9, :]
        img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width,:] \
            = img_out[0, -output_length + i, :]
        if configs.is_training == True and configs.dataset == 'kth':
            if i % 2 != 0:
                continue
            else:
                img_pred[:res_height,((i // 2) * res_width + (i // 2) * interval):(((i // 2) + 1) * res_width + (i // 2) * interval),:] = img_out[0, -output_length + i, :]
        else:
            img_pred[:res_height, (i * res_width + i * interval):((i + 1) * res_width + i * interval),:] = img_out[0, -output_length + i, :]
        if configs.img_channel == 2:
            img_pred_single[:, :, :] = img_out[0, -output_length + i, :]
            img_total_pred_single = img_pred_single[:, :, 0] + img_pred_single[:, :, 1]
            img_pred_name_single = 'batch_' + str(batch_id) + '_pred_' + str(i) + '.svg'
            file_img_pred_name_single_svg = os.path.join(res_path, img_pred_name_single)
            plt.imsave(file_img_pred_name_single_svg, img_total_pred_single.reshape(img_total_pred_single.shape[0],img_total_pred_single.shape[1]),vmin=0, vmax=1.0)

    # 将小于0的变成0, 将大于1的变成1
    if configs.img_channel == 2:
        # add_image = np.zeros((2 * res_height,
        #          configs.total_length * res_width,1))
        # img = np.concatenate([img,add_image],axis=2)
        img_total = img[:, :, 0] + img[:, :, 1]
        name_svg = str(batch_id) + '.svg'

        img_total_input = img_input[:, :, 0] + img_input[:, :, 1]
        img_input_name = str(batch_id) + '_input.svg'

        img_total_ground_true = img_ground_true[:, :, 0] + img_ground_true[:, :, 1]
        img_ground_true_name = str(batch_id) + '_ground_true.svg'

        img_total_pred = img_pred[:, :, 0] + img_pred[:, :, 1]
        img_pred_name = str(batch_id) + '_pred.svg'

        # file_name = results/mau/1.png
        file_name_svg = os.path.join(res_path, name_svg)

        file_img_input_nam_svg = os.path.join(res_path, img_input_name)
        file_img_ground_true_name_svg = os.path.join(res_path, img_ground_true_name)
        file_img_pred_name_svg = os.path.join(res_path, img_pred_name)

        plt.imsave(file_name_svg, img_total.reshape(img_total.shape[0], img_total.shape[1]), vmin=0, vmax=1.0)

        plt.imsave(file_img_input_nam_svg, img_total_input.reshape(img_total_input.shape[0], img_total_input.shape[1]), vmin=0, vmax=1.0)
        plt.imsave(file_img_ground_true_name_svg, img_total_ground_true.reshape(img_total_ground_true.shape[0], img_total_ground_true.shape[1]), vmin=0, vmax=1.0)
        plt.imsave(file_img_pred_name_svg, img_total_pred.reshape(img_total_pred.shape[0], img_total_pred.shape[1]), vmin=0, vmax=1.0)
    else:
        img = np.maximum(img, 0)
        img = np.minimum(img, 1)

        img_input = np.maximum(img_input, 0)
        img_input = np.minimum(img_input, 1)

        img_ground_true = np.maximum(img_ground_true, 0)
        img_ground_true = np.minimum(img_ground_true, 1)

        img_pred = np.maximum(img_pred, 0)
        img_pred = np.minimum(img_pred, 1)

        # 写出对比图片
        cv2.imwrite(file_name, (img * 255).astype(np.uint8))
        cv2.imwrite(file_img_input_name, (img_input * 255).astype(np.uint8))
        cv2.imwrite(file_img_ground_true_name, (img_ground_true * 255).astype(np.uint8))
        cv2.imwrite(file_img_pred_name, (img_pred * 255).astype(np.uint8))
