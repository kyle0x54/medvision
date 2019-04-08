from .hook import Hook
import medvision as mv


class LoggerHook(Hook):
    """ Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
    """
    def __init__(self, interval=10):
        self.interval = interval
        self.visualizer = None

    def log(self, runner):
        pass

    def before_run(self, runner):
        if runner.mode == mv.ModeKey.TRAIN:
            self.visualizer = mv.TensorboardVisualizer(runner.experiment)

    def before_epoch(self, runner):
        runner.average_meter.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)

        if runner.average_meter.ready:
            self.log(runner)

    def after_train_epoch(self, runner):
        if runner.average_meter.ready:
            self.log(runner)

    def after_val_epoch(self, runner):
        runner.average_meter.average()
        self.log(runner)


# class ClsLoggerHook(LoggerHook):
#     def __init__(self, interval):
#         super(ClsLoggerHook, self).__init__(interval)
#
#     def log(self, runner):
#         if runner.mode == 'train':
#             lr_str = ', '.join(
#                 ['{:.5f}'.format(lr) for lr in runner.current_lr()])
#             log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
#                 runner.epoch + 1, runner.inner_iter + 1,
#                 len(runner.data_loader), lr_str)
