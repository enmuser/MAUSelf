import json
import random
import os

import cv2
import torch
import torchvision.transforms
from tqdm import tqdm

from core.data_provider.vp import VPDataset, VPData
from pathlib import Path
#from core.data_provider.vp.defaults import SETTINGS
from core.data_provider.vp.utils import set_from_kwarg, read_video


class CaltechPedestrianDataset(VPDataset):
    r"""
    Dataset class for the dataset "Caltech Pedestrian", as firstly encountered in
    "Pedestrian Detection: A Benchmark" by Dollár et al.
    (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/CVPR09pedestrians.pdf).

    Each sequence shows a short clip of 'driving through regular traffic in an urban environment'.
    """
    NAME = "Caltech Pedestrian"
    REFERENCE = "http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/"
    IS_DOWNLOADABLE = "Yes"
    VALID_SPLITS = ["train", "val", "test"]
    MIN_SEQ_LEN = 568  #: Minimum number of frames across all sequences (1322 in 2nd-shortest, 2175 in longest)
    ACTION_SIZE = 0
    DATASET_FRAME_SHAPE = (480, 640, 3)
    FPS = 30  #: Frames per second.
    TRAIN_VAL_SETS = [f"set{i:02d}" for i in range(6)]  #: The official training sets (here: training and validation).
    TEST_SETS = [f"set{i:02d}" for i in range(6, 11)]  #: The official test sets.

    train_to_val_ratio = 0.9

    def __init__(self, split,data_root_path, json_path,**dataset_kwargs):
        super(CaltechPedestrianDataset, self).__init__(split,data_root_path,json_path, **dataset_kwargs)
        self.NON_CONFIG_VARS.extend(["sequences", "sequences_with_frame_index",
                                     "AVAILABLE_CAMERAS"])
        self.DEFAULT_DATA_DIR = data_root_path
        # set attributes
        set_from_kwarg(self, dataset_kwargs, "train_to_val_ratio")
        set_from_kwarg(self, dataset_kwargs, "train_val_seed")

        self.data_dir = data_root_path
        self.data_dir_json = json_path

        frame_count_path = os.path.join(self.DEFAULT_DATA_DIR,"frame_counts.json")
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

        if self.split == "test":
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("\\")[-2] in self.TEST_SETS]
            if len(sequences) < 1:
                raise ValueError(f"Dataset {self.NAME}: didn't find enough test sequences "
                                 f"-> can't use dataset")
        else:
            sequences = [(fp, frames) for (fp, frames) in sequences if fp.split("\\")[-2] in self.TRAIN_VAL_SETS]
            if len(sequences) < 2:
                raise ValueError(f"Dataset {self.NAME}: didn't find enough train/val sequences "
                                 f"-> can't use dataset")
            slice_idx = max(1, int(len(sequences) * self.train_to_val_ratio))
            random.Random(self.train_val_seed).shuffle(sequences)
            if self.split == "train":
                sequences = sequences[:slice_idx]
            else:
                sequences = sequences[slice_idx:]
        self.sequences = sequences

        self.sequences_with_frame_index = []  # mock value, must not be used for iteration till sequence length is set

    def _set_seq_len(self):
        # Determine per video which frame indices are valid start indices. Each resulting index marks a datapoint.
        for sequence_path, frame_count in self.sequences:
            valid_start_idx = range(0, frame_count - self.seq_len + 1,
                                    self.seq_len + self.seq_step - 1)
            for idx in valid_start_idx:
                self.sequences_with_frame_index.append((sequence_path, idx))

    def __getitem__(self, i) -> VPData:
        sequence_path, start_idx = self.sequences_with_frame_index[i]
        vid = read_video(sequence_path, start_index=start_idx, num_frames=self.seq_len)  # [T, h, w, c]
        vid = vid[::self.seq_step]  # [t, h, w, c]
        vid = self.preprocess(vid)  # [t, c, h, w]
        actions = torch.zeros((self.total_frames, 1))  # [t, a], actions should be disregarded in training logic

        data = {"frames": vid, "actions": actions, "origin": f"{sequence_path}, start frame: {start_idx}"}
        return vid

    def __len__(self):
        return len(self.sequences_with_frame_index)

    @classmethod
    def download_and_prepare_dataset(cls):
        pass

