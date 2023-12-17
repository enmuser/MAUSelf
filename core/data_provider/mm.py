import cv2
import numpy as np
import os
import random
import torch
import torch.utils.data as data
import cv2 as cv

from core.utils.ImagesToVideo import img2video

def load_mnist(root):
    path = os.path.join(root, 'train-images.idx3-ubyte')
    with open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, is_train):
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root, False)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames = n_frames
        self.transform = transform
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            start_y, start_x = self.get_random_trajectory(self.n_frames)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames
        if self.is_train or self.num_objects[0] != 2:
            num_digits = random.choice(self.num_objects)
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:length, idx, ...]
        r = 1
        w = int(64 / r)
        img = np.ones((64, 64, 1))
        img_mask = np.ones((length, 64, 64, 1))
        img_background = np.ones((length, 64, 64, 1))
        for t in range(length):
            img = images[t]
            name = str(t) + '.png'
            file_name = os.path.join("/kaggle/working/MAUSelf/results/mau/video/file", name)
            cv2.imwrite(file_name, img.astype(np.uint8))
        img2video(image_root="/kaggle/working/MAUSelf/results/mau/video/file/", dst_name="/kaggle/working/MAUSelf/results/mau/video/file/images.mp4")
        backSub = cv.createBackgroundSubtractorMOG2()
        #backSub = cv.createBackgroundSubtractorKNN()
        capture = cv.VideoCapture(cv.samples.findFileOrKeep("/kaggle/working/MAUSelf/results/mau/video/file/images.mp4"))
        count = 0
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            fgMask = np.expand_dims(fgMask, axis=2)
            img_mask[count] = fgMask
            background = backSub.getBackgroundImage()
            background_0 = background[:, :, 0]
            background_0 = np.expand_dims(background_0, axis=2)
            img_background[count] = background_0
            count += 1
            #print("count=", count)


        # 20 * 1 * 64 * 64
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        images_mask = img_mask.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        images_background = img_background.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))
        # img = np.ones((64, 64, 1))
        # capture = cv.VideoCapture(cv.s)
        output = torch.from_numpy(images / 255.0).contiguous().float()
        output_mask = torch.from_numpy(images_mask / 255.0).contiguous().float()
        output_background = torch.from_numpy(images_background / 255.0).contiguous().float()
        # for t in range(19):
        #     net = images[t]
        #
        #     backSub = cv.createBackgroundSubtractorMOG2()
        #     fgMask = backSub.apply(net)
        #     background = backSub.getBackgroundImage()
        #     cv.imshow('Frame', net)
        #     cv.imshow('FG Background', background)
        #     cv.imshow('FG Mask', fgMask)
        return output, output_mask, output_background

    def __len__(self):
        return self.length
