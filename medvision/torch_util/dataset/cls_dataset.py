from torch.utils.data.dataset import Dataset
import medvision as mv


class ClsDataset(Dataset):
    # TODO: avoid using cfg as argument
    def __init__(self, cfg, mode, build_transform, image_loader):
        self.is_train = (mode == mv.ModeKey.TRAIN)

        self.mode2dsmd = {
            mv.ModeKey.TRAIN: cfg.DATA.TRAIN_DSMD,
            mv.ModeKey.VAL: cfg.DATA.VAL_DSMD,
            mv.ModeKey.TEST: cfg.DATA.TEST_DSMD,
        }
        dsmd_path = self.mode2dsmd[mode]
        assert mv.isfile(dsmd_path) or mv.isdir(dsmd_path)

        if mv.isfile(dsmd_path):  # for a dsmd file
            self.dsmd = mv.load_dsmd(dsmd_path)
            self.filenames = list(self.dsmd.keys())
            self.filepaths = [
                mv.joinpath(cfg.DATA.IMAGE_DIR, filename)
                for filename in self.filenames
            ]
        else:  # for a directory containing test images
            self.dsmd = None
            self.filepaths = mv.listdir(dsmd_path)

        self.transform = build_transform(cfg, self.is_train)
        self.image_loader = image_loader

    def __getitem__(self, index):
        image = self.image_loader(self.filepaths[index])

        if self.transform is not None:
            image = self.transform(image)

        if self.dsmd is None:
            return image
        else:
            return image, self.dsmd[self.filenames[index]]

    def __len__(self):
        return len(self.filepaths)
