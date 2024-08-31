import os
import random
from pathlib import Path

import cv2
import cv2 as cv
import numpy as np
import torch

from torch.utils.data import Dataset

from core.utils.ImagesToVideo import img2video


class KITTIDataset(Dataset):
    r"""
    Dataset class for the "raw data" regime of the "KITTI Vision Benchmark Suite", as described in
    "Vision meets Robotics: The KITTI Dataset" by Geiger et al. (http://www.cvlibs.net/publications/Geiger2013IJRR.pdf).

    Each sequence shows a short clip of 'driving around the mid-size city of Karlsruhe, in rural areas and on highways'.
    """
    NAME = "KITTI raw"
    REFERENCE = "http://www.cvlibs.net/datasets/kitti/raw_data.php"
    IS_DOWNLOADABLE = "With Registered Account"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 994  #: Minimum number of frames across all sequences (6349 in longest).
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (375, 1242, 3)
    FPS = 10  #: Frames per Second.
    AVAILABLE_CAMERAS = [f"image_{i:02d}" for i in range(4)]  #: Available cameras: [`greyscale_left`, `greyscale_right`, `color_left`, `color_right`].
    camera = "image_02"  #: Chosen camera, can be set to any of the `AVAILABLE_CAMERAS`.

    def __init__(self, is_training, data_root_path, context_frames, pred_frames, seq_step, transform=None, **dataset_kwargs):
        super(KITTIDataset, self).__init__()
        self.DEFAULT_DATA_DIR = data_root_path

        total_frames = context_frames + pred_frames
        seq_len = (total_frames - 1) * seq_step + 1
        if self.MIN_SEQ_LEN < seq_len:
            raise ValueError(f"Dataset '{self.NAME}' supports videos with up to {self.MIN_SEQ_LEN} frames, "
                             f"which is exceeded by your configuration: "
                             f"{{context frames: {context_frames}, pred frames: {pred_frames}, seq step: {seq_step}}}")
        self.total_frames = total_frames
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.frame_offsets = range(0, (total_frames) * seq_step, seq_step)

        self.value_range_min = 0.0
        self.value_range_max = 1.0

        self.train_to_test_ratio = 0.8

        # get video filepaths
        dd = Path(self.DEFAULT_DATA_DIR)
        sequence_dirs = [sub for d in dd.iterdir() for sub in d.iterdir() if dd.is_dir() and sub.is_dir()]
        if len(sequence_dirs) < 3:
            raise ValueError(f"Dataset {self.NAME}: found less than 3 sequences "
                             f"-> can't split dataset -> can't use it")
        self.is_training = is_training
        # slice accordingly
        slice_idx = max(1, int(len(sequence_dirs) * self.train_to_test_ratio))
        if self.is_training:
            sequence_dirs = sequence_dirs[:slice_idx]
            random.Random(1234).shuffle(sequence_dirs)
        else:
            sequence_dirs = sequence_dirs[slice_idx:]

        # retrieve sequence lengths and store
        self.sequences = []
        for sequence_dir in sorted(sequence_dirs):
            sequence_len = len(list(sequence_dir.rglob(f"{self.camera}/data/*.png")))
            self.sequences.append((sequence_dir, sequence_len))

        self.sequences_with_frame_index = []  # mock value, must not be used for iteration till sequence length is set
        self._set_seq_len()
        self.transform = transform

    def _set_seq_len(self):
        # Determine per video which frame indices are valid
        for sequence_path, frame_count in self.sequences:
            valid_start_idx = range(0, frame_count - self.seq_len + 1,
                                    self.seq_len + self.seq_step - 1)
            for idx in valid_start_idx:
                self.sequences_with_frame_index.append((sequence_path, idx))

    def __getitem__(self, i):
        sequence_path, start_idx = self.sequences_with_frame_index[i]
        all_img_paths = sorted(list(sequence_path.rglob(f"{self.camera}/data/*.png")))
        seq_img_paths = all_img_paths[start_idx:start_idx+self.seq_len:self.seq_step]
        #print('seq_img_paths: ',seq_img_paths)# t items of [h, w, c]
        seq_imgs = [cv2.cvtColor(cv2.imread(str(fp.resolve())), cv2.COLOR_BGR2RGB) for fp in seq_img_paths]
        frames = np.stack(seq_imgs, axis=0)  # [t, *self.DATASET_FRAME_SHAPE]

        img_mask_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_background_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        for t in range(frames.shape[0]):
            img = frames[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/kitti/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/kitti/video/file/",
                  dst_name="/kaggle/working/MAUSelf/results/kitti/video/file/images.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(
            cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/kitti/video/file/images.mp4"))
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

        frames = self.preprocess(frames)  # [t, c, h, w]
        img_mask_frames = self.preprocess(img_mask_frames)  # [t, c, h, w]
        img_background_frames = self.preprocess(img_background_frames)

        return frames, img_mask_frames, img_background_frames

    def __len__(self):
        return len(self.sequences_with_frame_index)

    @classmethod
    def download_and_prepare_dataset(cls):
        pass

    def preprocess(self, x, transform: bool = True) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            if x.dtype == np.uint16:
                x = x.astype(np.float32) / ((1 << 16) - 1)
            elif x.dtype == np.uint8:
                x = x.astype(np.float32) / ((1 << 8) - 1)
            elif x.dtype == float:
                x = x / ((1 << 8) - 1)
            else:
                raise ValueError(f"if providing numpy arrays, only dtypes "
                                 f"np.uint8, np.float and np.uint16 are supported (given: {x.dtype})")
            x = torch.from_numpy(x).float()
        elif torch.is_tensor(x):
            if x.dtype == torch.uint8:
                x = x.float() / ((1 << 8) - 1)
            elif x.dtype == torch.double:
                x = x.float()
            else:
                raise ValueError(f"if providing pytorch tensors, only dtypes "
                                 f"torch.uint8, torch.float and torch.double are supported (given: {x.dtype})")
        else:
            raise ValueError(f"expected input to be either a numpy array or a PyTorch tensor")

        # assuming shape = [..., h, w(, c)], putting channel dim at index -3
        if x.ndim < 2:
            raise ValueError(f"expected at least two dimensions for input image")
        elif x.ndim == 2:
            x = x.unsqueeze(dim=0)
        else:
            permutation = list(range(x.ndim - 3)) + [-1, -3, -2]
            x = x.permute(permutation)

        # scale
        if self.value_range_min != 0.0 or self.value_range_max != 1.0:
            x *= self.value_range_max - self.value_range_min  # [0, max_val - min_val]
            x += self.value_range_min  # [min_val, max_val]

        # crop -> resize -> augment
        if transform:
            x = self.transform(x)
        return x

