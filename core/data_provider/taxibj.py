# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @time: 2022/4/19 13:57
# @author: 芜情
# @description:
import os
from pathlib import Path
from typing import Tuple

import cv2
import cv2 as cv
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rembg import remove, new_session

__all__ = ["TaxiBJDataset"]

from core.utils.ImagesToVideo import img2video


class TaxiBJDataset(Dataset):


    def __init__(self, dataset: str,data_root_path):
        r"""
        数据集 0-1 归一化，按照测试集最小值 0， 最大值 1292
        """
        if dataset == "train":
            self.dataset_dir = Path(data_root_path + r"train")
        elif dataset == "test":
            self.dataset_dir = Path(data_root_path + r"test")
        else:
            raise FileNotFoundError(f"\nthe dataset {dataset} in Moving MNIST doesn't exist.\n")

        self.__len = len(list(self.dataset_dir.glob("*.npy")))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        example_path = str(self.dataset_dir.joinpath(f"example{index + 1:06d}.npy"))
        sequence = np.load(example_path)

        frames = sequence.transpose((0, 3, 2, 1))
        img_mask_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_background_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_mask_frames_0 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_background_frames_0 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_mask_frames_1 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_background_frames_1 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        rembg_session = new_session()
        for t in range(frames.shape[0]):
            img = frames[t][:, :, 0].reshape((frames.shape[1], frames.shape[2], 1))
            # name = str(index)+'_first_'+str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file0", name)
            # cv2.imwrite(file_name, img.astype(np.uint8))

            image = np.concatenate((img, img, img), axis=-1)
            mask = remove(image.astype(np.uint8), output_format='rgba', session=rembg_session)

            foreground = cv2.bitwise_and(image, image, mask=mask[..., 3])

            # name = str(index) + '_first_foreground_' + str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file0", name)
            # cv2.imwrite(file_name, foreground.astype(np.uint8))


            background = cv2.subtract(image, foreground)

            # name = str(index) + '_first_background_' + str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file0", name)
            # cv2.imwrite(file_name, background.astype(np.uint8))

            foreground = foreground[:, :, 0]
            foreground = np.expand_dims(foreground, axis=2)

            img_mask_frames_0[t] = foreground
            background = background[:, :, 0]
            background = np.expand_dims(background, axis=2)
            img_background_frames_0[t] = background


        for t in range(frames.shape[0]):
            img = frames[t][:, :, 1].reshape((frames.shape[1], frames.shape[2], 1))

            # name = str(index) + '_second_' + str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file1", name)
            # cv2.imwrite(file_name, img.astype(np.uint8))

            image = np.concatenate((img, img, img), axis=-1)
            mask = remove(image.astype(np.uint8), output_format='rgba', session=rembg_session)

            foreground = cv2.bitwise_and(image, image, mask=mask[..., 3])

            # name = str(index) + '_second_foreground_' + str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file1", name)
            # cv2.imwrite(file_name, foreground.astype(np.uint8))


            background = cv2.subtract(image, foreground)

            # name = str(index) + '_second_background_' + str(t) + '.png'
            # file_name = os.path.join("results/taxibj/video/file1", name)
            # cv2.imwrite(file_name, background.astype(np.uint8))

            foreground = foreground[:, :, 0]
            foreground = np.expand_dims(foreground, axis=2)
            img_mask_frames_1[t] = foreground
            background = background[:, :, 0]
            background = np.expand_dims(background, axis=2)
            img_background_frames_1[t] = background


        img_mask_frames[:, :, :, 0] = img_mask_frames_0.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_mask_frames[:, :, :, 1] = img_mask_frames_1.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_background_frames[:, :, :, 0] = img_background_frames_0.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_background_frames[:, :, :, 1] = img_background_frames_1.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))

        img_mask_frames = img_mask_frames.transpose((0, 3, 1, 2))
        img_background_frames = img_background_frames.transpose((0, 3, 1, 2))

        frames = frames.transpose((0, 3, 1, 2))

        frames = torch.from_numpy(frames).float() / 1292.0
        img_mask_frames = torch.from_numpy(img_mask_frames).float() / 1292.0
        img_background_frames = torch.from_numpy(img_background_frames).float() / 1292.0

       # inputs, labels = torch.split(sequence, 4)

        return frames, img_mask_frames, img_background_frames

    def __getitem_bak__(self, index: int) -> Tuple[Tensor, Tensor]:
        example_path = str(self.dataset_dir.joinpath(f"example{index + 1:06d}.npy"))
        sequence = np.load(example_path)

        frames = sequence.transpose((0, 3, 2, 1))
        img_mask_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_background_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_mask_frames_0 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_background_frames_0 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_mask_frames_1 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        img_background_frames_1 = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], 1))
        for t in range(frames.shape[0]):
            img = frames[t][:, :, 0].reshape((frames.shape[1], frames.shape[2], 1))
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/taxibj/video/file0", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/taxibj/video/file0/",
                  dst_name="/kaggle/working/MAUSelf/results/taxibj/video/file0/images0.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(
            cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/taxibj/video/file0/images0.mp4"))
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            fgMask = np.expand_dims(fgMask, axis=2)
            img_mask_frames_0[count] = fgMask
            background = backSub.getBackgroundImage()
            background_0 = background[:, :, 0]
            background_0 = np.expand_dims(background_0, axis=2)
            img_background_frames_0[count] = background_0
            count += 1

        for t in range(frames.shape[0]):
            img = frames[t][:, :, 1].reshape((frames.shape[1], frames.shape[2], 1))
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/taxibj/video/file1", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/taxibj/video/file1/",
                  dst_name="/kaggle/working/MAUSelf/results/taxibj/video/file1/images1.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(
            cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/taxibj/video/file1/images1.mp4"))
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            fgMask = np.expand_dims(fgMask, axis=2)
            img_mask_frames_1[count] = fgMask
            background = backSub.getBackgroundImage()
            background_0 = background[:, :, 0]
            background_0 = np.expand_dims(background_0, axis=2)
            img_background_frames_1[count] = background_0
            count += 1

        img_mask_frames[:, :, :, 0] = img_mask_frames_0.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_mask_frames[:, :, :, 1] = img_mask_frames_1.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_background_frames[:, :, :, 0] = img_background_frames_0.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))
        img_background_frames[:, :, :, 1] = img_background_frames_1.reshape((frames.shape[0], frames.shape[1], frames.shape[2]))

        img_mask_frames = img_mask_frames.transpose((0, 3, 1, 2))
        img_background_frames = img_background_frames.transpose((0, 3, 1, 2))

        frames = frames.transpose((0, 3, 1, 2))

        frames = torch.from_numpy(frames).float() / 1292.0
        img_mask_frames = torch.from_numpy(img_mask_frames).float() / 1292.0
        img_background_frames = torch.from_numpy(img_background_frames).float() / 1292.0

       # inputs, labels = torch.split(sequence, 4)

        return frames, img_mask_frames, img_background_frames

    def __len__(self):
        return self.__len
