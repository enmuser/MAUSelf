import json
import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path

from core.data_provider.vp.typing import Union
import numpy as np
import cv2
import cv2 as cv

from core.utils.ImagesToVideo import img2video



class PedestrianDataset(Dataset):
    r"""
    Dataset class for the dataset "Caltech Pedestrian", as firstly encountered in
    "Pedestrian Detection: A Benchmark" by Dollár et al.
    (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/CVPR09pedestrians.pdf).

    Each sequence shows a short clip of 'driving through regular traffic in an urban environment'.
    """
    NAME = "Caltech Pedestrian"
    REFERENCE = "http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/"
    IS_DOWNLOADABLE = "Yes"
    MIN_SEQ_LEN = 568  #: Minimum number of frames across all sequences (1322 in 2nd-shortest, 2175 in longest)
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (480, 640, 3)
    FPS = 30  #: Frames per second.
    TRAIN_SETS = [f"set{i:02d}" for i in range(9)]  #: The official training sets (here: training and validation).
    TEST_SETS = [f"set{i:02d}" for i in range(9, 11)]

    def __init__(self, is_training, data_root_path, json_path, context_frames, pred_frames, seq_step, transform=None, **dataset_kwargs):
        super(PedestrianDataset, self).__init__()
        self.DEFAULT_DATA_DIR = data_root_path
        self.is_training = is_training
        self.data_dir = data_root_path
        self.data_dir_json = json_path

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

        frame_count_path = os.path.join(self.DEFAULT_DATA_DIR, "frame_counts.json")
        if not os.path.exists(frame_count_path):
            print(f"Analyzing video frame counts...")
            sequences = sorted(list(Path(self.DEFAULT_DATA_DIR).rglob("**/*.seq")))
            sequences_with_frame_counts = dict()
            for seq in tqdm(sequences):
                fp = str(seq.resolve())
                cap = cv2.VideoCapture(fp)
                # for these .seq files, cv2.CAP_PROP_FRAME_COUNT returns garbage,
                # so we have to manually read out the seq
                frames = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames += 1
                sequences_with_frame_counts[fp] = frames
            with open(str(Path(frame_count_path).resolve()), "w") as frame_count_file:
                json.dump(sequences_with_frame_counts, frame_count_file)

        # get sequence filepaths and slice accordingly
        with open(os.path.join(self.data_dir_json, "frame_counts.json"), "r") as frame_counts_file:
            sequences = json.load(frame_counts_file).items()

        if self.is_training:
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("\\")[-2] in self.TRAIN_SETS]
            random.Random(1234).shuffle(sequences)
        else :
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("\\")[-2] in self.TEST_SETS]
        self.sequences = sequences

        self.sequences_with_frame_index = []
        self._set_seq_len()
        self.transform = transform


    def _set_seq_len(self):
        # Determine per video which frame indices are valid start indices. Each resulting index marks a datapoint.
        for sequence_path, frame_count in self.sequences:
            valid_start_idx = range(0, frame_count - self.seq_len + 1,
                                    self.seq_len + self.seq_step - 1)
            for idx in valid_start_idx:
                self.sequences_with_frame_index.append((sequence_path, idx))

    def __getitem__(self, i):
        sequence_path, start_idx = self.sequences_with_frame_index[i]
        vid = read_video(sequence_path, start_index=start_idx, num_frames=self.seq_len)  # [T, h, w, c]
        frames = vid[::self.seq_step]  # [t, h, w, c]
        img_mask_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        img_background_frames = np.ones((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]))
        for t in range(frames.shape[0]):
            img = frames[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/caltech_pedestrian/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/caltech_pedestrian/video/file/",
                  dst_name="/kaggle/working/MAUSelf/results/caltech_pedestrian/video/file/images.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(
            cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/caltech_pedestrian/video/file/images.mp4"))
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
        img_background_frames = self.preprocess(img_background_frames)  # [t, c, h, w]

        return frames, img_mask_frames, img_background_frames

    def __len__(self):
        return len(self.sequences_with_frame_index)


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
def read_video(fp: Union[Path, str], img_size: (int, int) = None,
               start_index=0, num_frames=-1):
    r"""
    Reads and returns the video specified by given file path as a numpy array.

    Args:
        fp (Union[Path, str]): The filepath to read the video from.
        img_size ((int, int)): The desired frame size (height and width; frames will be reshaped to this size)
        start_index (int): Index of first frame to read.
        num_frames (int): Nmber of frames to read (default value -1 signifies that video is read to the end).

    Returns: The read video as a numpy array of shape (frames, height, width, channels).
    """
    if isinstance(fp, Path):
        fp = str(fp.resolve())
    cap = cv2.VideoCapture(fp)
    if not cap.isOpened():
        raise ValueError(f"opening MP4 file '{fp}' failed")

    collected_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    while num_frames < 0 or len(collected_frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        collected_frames.append(frame)
    cap.release()

    if img_size is not None:
        h, w = img_size
        collected_frames = [cv2.resize(frame, (w, h)) for frame in collected_frames]
    collected_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in collected_frames]
    return np.stack(collected_frames, axis=0)