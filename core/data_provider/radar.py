import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import imageio
class RadarDataset(data.Dataset):

    def __init__(self, split, data_root_path,num_frames=20, num_channels=3, img_size=64, horiz_flip_aug=True):
        self.data_path = data_root_path

        self.split = split
        self.n_frames = num_frames
        self.num_channels = num_channels
        self.img_size = img_size
        self.horiz_flip_aug = horiz_flip_aug and self.split != "test"

        self.dataset = "train" if self.split != "test" else "test"

        if self.dataset == "train":
            self.data_root = os.path.join(self.data_path, f"train")
            self.idx_list = list(range(0, 3150))
        else:
            self.data_root = os.path.join(self.data_path, f"test")
            self.idx_list = list(range(0, 115))

    def __getitem__(self, index):
        fileIndex = index + 1
        fileDirectoryName = os.path.join(self.data_root, f'sample_{fileIndex}')
        frames = np.zeros((self.n_frames, self.img_size, self.img_size, self.num_channels))
        img_mask_frames = np.ones((self.n_frames, self.img_size, self.img_size, self.num_channels))
        img_background_frames = np.ones((self.n_frames, self.img_size, self.img_size, self.num_channels))
        for i in range(0, 15):
            fname = os.path.join(fileDirectoryName, f'{i+1}.png')
            im = imageio.imread(fname, as_gray=True)
            if self.num_channels == 3:
               frames[i] = np.stack((im,) * 3, axis=-1)
            else:
               frames[i] = np.expand_dims(im, axis=-1)

        fileDirectoryVideoName = os.path.join(fileDirectoryName, f'sample_{fileIndex}.mp4')

        backSub = cv2.createBackgroundSubtractorMOG2()
        # backSub = cv.createBackgroundSubtractorKNN()
        capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(fileDirectoryVideoName))

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

        frames = torch.from_numpy(frames / 255.0).contiguous().float().permute(0, 3, 1, 2)
        img_mask_frames = torch.from_numpy(img_mask_frames / 255.0).contiguous().float().permute(0, 3, 1, 2)
        img_background_frames = torch.from_numpy(img_background_frames / 255.0).contiguous().float().permute(0, 3, 1, 2)
        if self.horiz_flip_aug:
            frames = torch.flip(frames, dims=[3])
            img_mask_frames = torch.flip(img_mask_frames, dims=[3])
            img_background_frames = torch.flip(img_background_frames, dims=[3])
        return frames, img_mask_frames, img_background_frames
    def __len__(self):
        return len(self.idx_list)