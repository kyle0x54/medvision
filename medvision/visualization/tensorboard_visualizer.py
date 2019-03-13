import time
import numpy as np
from tensorboardX import SummaryWriter
import medvision as mv


class TensorboardVisualizer:
    """ A wrapper class for tensorboardX.

    Note:
        Original tensorboardX APIs can be called via self.vis.function_name.
    """
    def __init__(self, env=None, **kwargs):
        comment = '_' + str(env) if env is not None else ''
        self.writer = SummaryWriter(comment=comment, **kwargs)
        self.log_text = ''

    def __del__(self):
        self.writer.close()

    def plot(self, name, x, y, group='data'):
        tag = group + '/' + name
        self.writer.add_scalar(tag, y, x)

    def imshow(self, name, img_tensor, global_step=None, group='image'):
        """ Accept NCHW numpy.ndarray or torch tensor as input.
        """
        imgs = mv.make_np(img_tensor)
        assert imgs.ndim == 4

        if imgs.shape[1] == 1:
            imgs = np.concatenate([imgs, imgs, imgs], 1)

        tag = group + '/' + name
        self.writer.add_images(tag, imgs, global_step)

    def log(self, info, global_step=None, tag='text/log'):
        text_string = '[{time}] {info} <br>'.format(
            time=time.strftime('%y-%m-%d %H:%M:%S'),
            info=info
        )
        self.writer.add_text(tag, text_string, global_step)

    def __getattr__(self, name):
        return getattr(self.writer, name)


if __name__ == '__main__':
    import torch
    from torchvision import datasets

    visualizer = TensorboardVisualizer('tensorboard-test')

    for n_iter in range(100):
        dummy_s1 = torch.rand(1)
        dummy_s2 = torch.rand(1)
        # data grouping by `slash`
        visualizer.plot('scalar1', n_iter, dummy_s1[0])
        visualizer.plot('scalar2', n_iter, dummy_s2[0])

        dummy_img = torch.rand(32, 1, 64, 64)  # output from network
        dummy_img_c = torch.rand(32, 3, 64, 64)  # output from network
        if n_iter % 10 == 0:
            visualizer.imshow('feature_map', dummy_img, n_iter)
            visualizer.imshow('feature_map2', dummy_img_c, n_iter)

            visualizer.log(str(n_iter), n_iter)

            # needs tensorboard 0.4RC or later
            visualizer.add_pr_curve(
                'xoxo', np.random.randint(2, size=100),
                np.random.rand(100), n_iter)

    dataset = datasets.MNIST('mnist', train=False, download=True)
    images = dataset.test_data[:100].float()
    label = dataset.test_labels[:100]

    features = images.view(100, 784)
    visualizer.add_embedding(features, metadata=label,
                             label_img=images.unsqueeze(1))
