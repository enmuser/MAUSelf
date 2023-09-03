import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.models.model_factory_B import Model_B
from core.models.model_factory_F import Model_F
import core.trainer as trainer
import core.trainer_B as trainer_B
import core.trainer_F as trainer_F
import pynvml
import cv2 as cv


pynvml.nvmlInit()
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MAU')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--is_train', type=str, default='True', required=False)
args_main = parser.parse_args()
args_main.tied = True

if args_main.is_train == 'True':
    from configs.mnist_train_configs import configs
else:
    from configs.mnist_configs import configs

parser = configs()
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.tied = True




def schedule_sampling(eta, itr, channel, batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    return eta, real_input_flag


def train_wrapper(model,model_f,model_b):
    begin = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])
    if args.pretrained_model_f:
        model_f.load(args.pretrained_model_f)
    if args.pretrained_model_b:
        model_b.load(args.pretrained_model_b)
    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    # real_input_flag = {}
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims, ims_mask, ims_back in train_input_handle:
            batch_size = ims.shape[0]
            real_input_flag_f = np.zeros(
                (batch_size,
                 args.total_length - args.input_length - 1,
                 args.img_height // args.patch_size,
                 args.img_width // args.patch_size,
                 args.patch_size ** 2 * args.img_channel))
            img_gen_f = model_f.test(ims_mask, real_input_flag_f)
            # trainer_F.test(ims_mask, img_gen_f, args, "aaaaa",itr)
            # ims_mask_ori = ims_mask
            ims_mask[:, -10:] = img_gen_f[:, -10:]
            # trainer_F.test(ims_mask_ori, ims_mask, args, "bbbbb",itr)
            real_input_flag_b = np.zeros(
                (batch_size,
                 args.total_length - args.input_length - 1,
                 args.img_height // args.patch_size,
                 args.img_width // args.patch_size,
                 args.patch_size ** 2 * args.img_channel))
            img_gen_b = model_b.test(ims_back, real_input_flag_b)
            #trainer_B.test(ims_back, img_gen_b, args, "aaaaa",itr)
            #ims_back_ori = ims_back
            ims_back[:, -10:] = img_gen_b[:, -10:]
            #trainer_B.test(ims_back_ori, ims_back, args, "bbbbb",itr)

            if itr > args.max_iterations:
                break
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size)
            if itr % args.test_interval == 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)
            trainer.train(model, ims, ims_mask, ims_back, real_input_flag, args, itr)
            # snapshot_interval = 1000 每1000次保存一次
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)
            itr += 1

            meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("GPU memory:%dM" % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)))


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':

    print('Initializing models')
    print('batch_size ', args.batch_size)
    #判断是训练还是测试
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False

    model = Model(args)
    model_f = Model_F(args)
    model_b = Model_B(args)

    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model, model_f, model_b)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)
