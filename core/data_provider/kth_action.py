"""
KTH-Actions dataset, including frames, pose keypoints and locations
"""

import random
import os
import json
import numpy as np
import torch
import imageio
import torchfile
from torch.utils.data import Dataset

import cv2
import cv2 as cv

from core.utils.ImagesToVideo import img2video



# Helper functions
def _read_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
    return data


def _swap(L, i1, i2):
    L[i1], L[i2] = L[i2], L[i1]


class KTH(Dataset):
    """
    KTH-Actions dataset. We obtain a sequence of frames with the corresponding body-joints in
    heatmap form, as well as the location of the person as a blob
    """
    KPOINTS = [0, 2, 5, 4, 7, 9, 12, 10, 13, 1]
    NUM_KPOINTS = len(KPOINTS)
    KPT_TO_IDX = {0: 0, 2: 1, 5: 2, 4: 3, 7: 4, 9: 5, 12: 6, 10: 7, 13: 8, 1: 9}
    SWAP_PAIRS = [(2, 5), (4, 7), (9, 12), (10, 13)]
    HARD_KPTS_PER_CLASS = {
            "boxing": [4, 7],
            "handclapping": [4, 7],
            "handwaving": [4, 7],
            "walking": [9, 12, 10, 13],
            "running": [9, 12, 10, 13],
            "jogging": [9, 12, 10, 13]
    }
    CLASSES = list(HARD_KPTS_PER_CLASS.keys())

    # classes with relatively shorter sequences
    SHORT_CLASSES = ['walking', 'running', 'jogging']
    MIN_SEQ_LEN = 29  # 14, 29, 49

    NUM_HMAP_CHANNELS = [NUM_KPOINTS - 1, 1]
    STRUCT_TYPE = "KEYPOINT_BLOBS"

    train_to_val_ratio = 0.98
    first_frame_rng_seed = 1234

    def __init__(self, is_training, data_root_path, num_frames=50, num_channels=3, img_size=64, horiz_flip_aug=True):
        """ Dataset initializer"""
        data_path = data_root_path
        self.data_root = os.path.join(data_path, f"KTH_{img_size}/processed")
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"KTH-Data does not exist in {self.data_root}...")

        self.is_training = is_training
        self.n_frames = num_frames
        self.num_channels = num_channels
        self.img_size = img_size
        self.horiz_flip_aug = horiz_flip_aug and self.is_training

        dataset = "train" if self.is_training else "test"
        self.dataset = dataset
        self.data = {}
        self.keypoints = {}
        for c in self.CLASSES:
            data_fname = os.path.join(self.data_root, c, f'{dataset}_meta{img_size}x{img_size}.t7')
            kpt_fname = os.path.join(self.data_root, c, f'{dataset}_keypoints{img_size}x{img_size}.json')
            self.data[c] = torchfile.load(data_fname)
            self.keypoints[c] = _read_json(kpt_fname)

        print('current kth image size ', img_size)
        print('current kth image channel ', num_channels)
        if img_size == 128:
            self._change_file_sequences()
        self.ALL_IDX = None
        self.IDX_TO_CLS_VID_SEQ = None
        # list of valid (cls, vid_idx, seq_idx) tuples
        if self.ALL_IDX is None:
            self.IDX_TO_CLS_VID_SEQ = self._find_valid_sequences()
            self.ALL_IDX = list(range(0, len(self.IDX_TO_CLS_VID_SEQ)))
        if self.is_training :
            random.shuffle(self.ALL_IDX)
        self.idx_list = self.ALL_IDX
    def _is_valid_sequence(self, seq, cls):
        """ Exploit short sequences of specific classes by extending them with repeated last frame """
        extend_seq = (cls in self.SHORT_CLASSES and len(seq) >= self.MIN_SEQ_LEN)
        return (len(seq) >= self.n_frames or extend_seq)

    def _find_valid_sequences(self):
        """ Ensure that a sequence has the sufficient number of frames """
        idx_to_cls_vid_seq = []
        for cls, cls_data in self.data.items():
            for vid_idx, vid in enumerate(cls_data):
                vid_seq = vid[b'files']
                for seq_idx, seq in enumerate(vid_seq):
                    if self._is_valid_sequence(seq, cls):
                        idx_to_cls_vid_seq.append((cls, vid_idx, seq_idx))
        return idx_to_cls_vid_seq

    def _change_file_sequences(self):
        for cls, cls_data in self.data.items():
            for vid_idx, vid in enumerate(cls_data):
                vid_seq = vid[b'files']
                temArr = []
                for seq_idx, seq in enumerate(vid_seq):
                    curArr = []
                    for item in seq:
                        item = item.decode("utf-8").replace('64x64', '128x128').encode("utf-8")
                        curArr.append(item)
                    temArr.append(curArr)
                vid[b'files'] = temArr

    def __getitem__(self, i):
        """ Sampling sequence from the dataset """
        i = self.idx_list[i]
        cls, vid_idx, seq_idx = self.IDX_TO_CLS_VID_SEQ[i]
        vid = self.data[cls][vid_idx]
        seq = vid[b'files'][seq_idx]

        # initializing arrays for images, kpts, and blobs
        cls_kps = self.keypoints[cls]
        dname = os.path.join(self.data_root, cls, vid[b'vid'].decode('utf-8'))
        frames = np.zeros((self.n_frames, self.img_size, self.img_size, self.num_channels))


        # getting random starting idx, and corresponding data
        first_frame = 0
        if len(seq) > self.n_frames and self.is_training:
            first_frame = random.randint(0, len(seq) - self.n_frames)
        last_frame = (len(seq) - 1) if (len(seq) <= self.n_frames) else (first_frame + self.n_frames - 1)
        for i in range(first_frame, last_frame + 1):
            fname = os.path.join(dname, seq[i].decode('utf-8'))
            im = imageio.imread(fname)
            if self.num_channels == 1:
                im = im[:, :, 0][:, :, np.newaxis]
            frames[i - first_frame] = im

        for i in range(last_frame + 1, self.n_frames):
            frames[i] = frames[last_frame]

        img_mask_frames = np.ones((self.n_frames, self.img_size, self.img_size, self.num_channels))
        img_background_frames = np.ones((self.n_frames, self.img_size, self.img_size, self.num_channels))
        for t in range(self.n_frames):
            img = frames[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/kth/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/kth/video/file/", dst_name="/kaggle/working/MAUSelf/results/kth/video/file/images.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/kth/video/file/images.mp4"))
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
        # 50 * 128 * 128 * 1
        #frames = torch.Tensor(frames).permute(0, 3, 1, 2)
        #img_mask_frames = torch.Tensor(img_mask_frames).permute(0, 3, 1, 2)
        #img_background_frames = torch.Tensor(img_background_frames).permute(0, 3, 1, 2)

        frames = torch.from_numpy(frames / 255.0).contiguous().float().permute(0, 3, 1, 2)
        img_mask_frames = torch.from_numpy(img_mask_frames / 255.0).contiguous().float().permute(0, 3, 1, 2)
        img_background_frames = torch.from_numpy(img_background_frames / 255.0).contiguous().float().permute(0, 3, 1, 2)

        if self.horiz_flip_aug and (random.randint(0, 1) == 0) and self.dataset == "train":
            frames = torch.flip(frames, dims=[3])
            img_mask_frames = torch.flip(img_mask_frames, dims=[3])
            img_background_frames = torch.flip(img_background_frames, dims=[3])

        # 50 * 1 * 128 * 128
        return frames, img_mask_frames, img_background_frames

    def _horiz_flip(self, frames, hmaps):
        """ Horizontal flip augmentation """
        frames = torch.flip(frames, dims=[3])
        assert len(hmaps) == 2
        hmaps_1, hmaps_2 = hmaps
        hmaps_1 = torch.flip(hmaps_1, dims=[3])
        hmaps_2 = torch.flip(hmaps_2, dims=[3])

        # swap symmetric keypoint channels
        kpoint_order = list(range(self.NUM_HMAP_CHANNELS[0]))
        for (k1, k2) in self.SWAP_PAIRS:
            i1 = self.KPT_TO_IDX[k1]
            i2 = self.KPT_TO_IDX[k2]
            _swap(kpoint_order, i1, i2)
        hmaps_1 = hmaps_1[:, kpoint_order]
        return frames, (hmaps_1, hmaps_2)

    def __len__(self):
        """ """
        return len(self.idx_list)

    def get_heatmap_weights(self, w_easy_kpts=1.0, w_hard_kpts=1.0):
        """ Getting specific weights for different keypoints """
        weights = {}
        for cls in self.CLASSES:
            weights[cls] = [w_easy_kpts] * self.NUM_HMAP_CHANNELS[0]
            hard_kpts = self.HARD_KPTS_PER_CLASS[cls]
            for kpt in hard_kpts:
                i = self.KPT_TO_IDX[kpt]
                weights[cls][i] = w_hard_kpts
        return weights

#