import time
import numpy as np
import visdom


class VisdomVisualizer:
    """ A wrapper class for visdom.

    Note:
        Original visdom APIs are supported through self.vis.function.
    """
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.log_text = ''

    def plot(self, name, x, y, **kwargs):
        update = 'append' if self.vis.win_exists(name) else None
        self.vis.line(
            Y=np.array([y]), X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=update,
            **kwargs
        )

    def show_images(self, name, imgs_, **kwargs):
        self.vis.images(
            imgs_.cpu().numpy(),
            win=name,
            opts=dict(title=name),
            **kwargs
        )

    def log(self, info, win='log'):
        self.log_text += (
            '[{time}] {info} <br>'.format(
                time=time.strftime('%y-%m-%d %H:%M:%S'),
                info=info
            )
        )
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
