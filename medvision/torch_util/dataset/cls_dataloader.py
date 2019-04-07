from torch.utils.data import DataLoader
from .cls_dataset import ClsDataset
import medvision as mv


def build_cls_dataloader(cfg, mode, build_transform, image_loader):
    # TODO: avoid using cfg as argument
    # init dataloader parameters
    if mode == mv.ModeKey.TRAIN:
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        num_workers = cfg.TRAIN.NUM_WORKERS
        pin_memory = True
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        num_workers = cfg.TEST.NUM_WORKERS
        pin_memory = False

    # build dataset
    cls_dataset = ClsDataset(
        cfg=cfg,
        mode=mode,
        build_transform=build_transform,
        image_loader=image_loader
    )

    # build dataloader
    data_loader = DataLoader(
        cls_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return data_loader
