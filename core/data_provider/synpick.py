"""
SynpickVP dataset. Obtaining video sequences from the SynpickVP dataset, along with
the corresponding semantic segmentation maps and gripper locations.
"""

import os
import imageio
import torch
from torchvision import transforms
import numpy as np

import cv2
import cv2 as cv

import torch.utils.data as data

from core.utils.ImagesToVideo import img2video


class SynpickDataset(data.Dataset):
    """
    Each sequence depicts a robotic suction cap gripper that moves around in a red bin filled with objects.
    Over the course of the sequence, the robot approaches 4 waypoints that are randomly chosen from the 4 corners.
    On its way, the robot is pushing around the objects in the bin.
    """

    CATEGORIES = ["master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
                  "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
                  "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
                  "large_clamp", "extra_large_clamp", "foam_brick", "gripper"]
    NUM_CLASSES = len(CATEGORIES)  # 22
    NUM_HMAP_CHANNELS = [NUM_CLASSES + 1, 1]
    STRUCT_TYPE = ["SEGMENTATION_MAPS", "KEYPOINT_BLOBS"]
    BIN_SIZE = (373, 615)
    SKIP_FIRST_N = 72            # To skip the first few frames in which gripper is not visible.
    GRIPPER_VALID_OFFSET = 0.01  # To skip sequences where the gripper_pos is near the edges of the bin.


    def __init__(self, is_training, data_root_path, num_frames, seq_step=2, img_size=(136, 240), hmap_size=(64, 112)):
        """ Dataset initializer """
        self.is_training = is_training
        if self.is_training :
            split = "train"
        else:
            split = "test"
        self.data_path = data_root_path
        self.data_dir = os.path.join(self.data_path, "SYNPICK", split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Synpick dataset does not exist in {self.data_dir}...")

        self.img_size = img_size
        self.num_frames = num_frames
        self.seq_step = seq_step

        # obtaining paths to data
        images_dir = os.path.join(self.data_dir, "rgb")
        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        # linking data into a structure
        self.valid_idx = []
        self.allow_seq_overlap = (self.is_training != "test")
        self._find_valid_sequences()
        return

    def __len__(self):
        """ """
        return len(self.valid_idx)

    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        i = self.valid_idx[i]  # only consider valid indices
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        idx = range(i, i + seq_len, self.seq_step)  # create range of indices for frame sequence
        imgs = [imageio.imread(self.image_fps[id_]) / 255. for id_ in idx]

        # preprocessing
        frames = np.stack(imgs, axis=0)

        img_mask_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_background_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        for t in range(frames.shape[0]):
            img = frames[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/synpick/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/synpick/video/file/",
                  dst_name="/kaggle/working/MAUSelf/results/synpick/video/file/images.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(
            cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/synpick/video/file/images.mp4"))
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            fgMask = np.expand_dims(fgMask, axis=2)
            img_mask_frames[count] = fgMask
            background = backSub.getBackgroundImage()
            background_0 = background[:, :, 0]
            background_0 = np.expand_dims(background_0, axis=2)
            img_background_frames[count] = background_0
            count += 1


        frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        frames = transforms.Resize(self.img_size)(frames)

        img_mask_frames = torch.Tensor(img_mask_frames).permute(0, 3, 1, 2)
        img_mask_frames = transforms.Resize(self.img_size)(img_mask_frames)

        img_background_frames = torch.Tensor(img_background_frames).permute(0, 3, 1, 2)
        img_background_frames = transforms.Resize(self.img_size)(img_background_frames)
        return frames, img_mask_frames, img_background_frames

    def _find_valid_sequences(self):
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        for idx in range(len(self.image_ids) - seq_len + 1):
            self.valid_idx.append(idx)
        assert len(self.valid_idx) > 0
        return

#
