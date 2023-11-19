from torch.utils.data import DataLoader

from core.data_provider.caltech_pedestrian import PedestrianDataset
from core.data_provider.kth_action import KTH
from core.data_provider.mm import MovingMNIST
import torchvision.transforms as TF


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
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
        dataset = KTH(
            is_training=is_training,
            data_root_path=root,
            num_frames=configs.total_length,
            num_channels=configs.img_channel,
            img_size=configs.img_height
        )
    elif configs.dataset == 'caltech_pedestrian':
        dataset = PedestrianDataset(
            is_training=is_training,
            data_root_path=root,
            transform=TF.Resize(size=(128, 160)),
            json_path=configs.json_path,
            context_frames=configs.input_length,
            pred_frames=configs.pred_length,
            seq_step=1
        )
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)
