"""
SynpickVP dataset. Obtaining video sequences from the SynpickVP dataset, along with
the corresponding semantic segmentation maps and gripper locations.
"""

import os
import imageio
import torch
from torchvision import transforms
import numpy as np
from .CONFIG import CONFIG, METRIC_SETS

import torch.utils.data as data


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

    METRICS_LEVEL_0 = METRIC_SETS["video_prediction"]
    METRICS_LEVEL_1 = METRIC_SETS["segmentation"]
    METRICS_LEVEL_2 = METRIC_SETS["single_keypoint_metric"]

    def __init__(self, split, data_root_path, num_frames, seq_step=2, img_size=(136, 240), hmap_size=(64, 112)):
        """ Dataset initializer """
        assert split in ["train", "val", "test"]
        self.data_path = data_root_path
        self.data_dir = os.path.join(self.data_path, "SYNPICK", split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Synpick dataset does not exist in {self.data_dir}...")
        self.split = split
        self.img_size = img_size
        self.num_frames = num_frames
        self.seq_step = seq_step

        # obtaining paths to data
        images_dir = os.path.join(self.data_dir, "rgb")
        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        # linking data into a structure
        self.valid_idx = []
        self.allow_seq_overlap = (self.split != "test")
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
        imgs = np.stack(imgs, axis=0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        imgs = transforms.Resize(self.img_size)(imgs)
        return imgs

    def _find_valid_sequences(self):
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        for idx in range(len(self.image_ids) - seq_len + 1):
            self.valid_idx.append(idx)
        assert len(self.valid_idx) > 0
        return

#
