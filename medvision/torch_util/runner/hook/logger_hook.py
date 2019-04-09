from .hook import Hook


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
