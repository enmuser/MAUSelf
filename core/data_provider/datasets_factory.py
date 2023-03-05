from torchvision import transforms
from torch.utils.data import DataLoader
from core.data_provider.vp.dataset_wrapper import VPDatasetWrapper

from core.data_provider.caltech_pedestrian import CaltechPedestrianDataset
from core.data_provider.mm import MovingMNIST
from core.data_provider import CustomMovingMNIST, KTH, SynpickMoving


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size, split,
                  is_training=True,
                  is_shuffle=True):
    if is_training:
        num_workers = configs.num_workers
        root = data_train_path
    else:
        num_workers = 0
        root = data_test_path
    if configs.dataset == 'minist':
        dataset = MovingMNIST(is_train=is_training,
                              root=root,
                              n_frames=20,
                              num_objects=[2])
    elif configs.dataset == 'kth':
        # dataset = KTH(is_train=is_training,
        #                       root=root,
        #                       n_frames=20,
        #                       num_objects=[2])
        dataset = KTH(
                split=split,
                data_root_path=root,
                num_frames=30,
                num_channels=3,
                img_size=configs.img_height
            )
    elif configs.dataset == 'synpick':
        dataset = SynpickMoving(
            split=split,
            data_root_path=root,
            num_frames=20,
            img_size=[64,112]
        )
    elif configs.dataset == 'caltech_pedestrian':
        # suite = VPSuite()
        # suite.load_dataset("CP")  # load moving MNIST dataset from default location
        # datasetTotal = suite.datasets[0]
        # datasetTotal.set_seq_len(10, 10, 1)
        # train_data, val_data = datasetTotal.train_data, datasetTotal.val_data
        # if is_training:
        #     dataset = train_data
        # else:
        #     dataset = val_data
        dataset_class = CaltechPedestrianDataset(
                split=split,
                data_root_path=root
           )
        split_tmp = split
        if split == "val":
            split_tmp = "train"
        datasetTotal = VPDatasetWrapper(dataset_class,split=split_tmp,data_root_path=root)
        datasetTotal.set_seq_len(configs.input_length,configs.pred_length,1)
        if split == "train":
            dataset = datasetTotal.train_data
        elif split == "val":
            dataset = datasetTotal.val_data
        elif split == "test":
            dataset = datasetTotal.test_data
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)
