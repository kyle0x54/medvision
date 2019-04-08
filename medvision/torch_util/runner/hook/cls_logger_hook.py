import logging
import torch
from .logger_hook import LoggerHook
import medvision as mv


class ClsLoggerHook(LoggerHook):
    def __init__(self, interval):
        super().__init__(interval)

    def log(self, runner):
        # 1. logging
        # basic information
        if runner.mode == mv.ModeKey.TRAIN:
            lr_str = ', '.join(
                ['{:.5f}'.format(lr) for lr in runner.current_lr()])
            log_str = 'Epoch [{}][{}/{}]\tlr: {}, '.format(
                runner.epoch + 1, runner.inner_iter + 1,
                len(runner.data_loader), lr_str)
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(
                runner.mode, runner.epoch, runner.inner_iter + 1)

        # CUDA memory information
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated()
            mem_mb = int(mem / (1024 * 1024))
            mem_str = 'memory: {}GB, '.format(mem_mb)
            log_str += mem_str

        # customized intermediate results
        log_items = []
        for name, val in runner.average_meter.output.items():
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)

        logging.info(log_str)

        # 2. TensorboardX
        if runner.mode == mv.ModeKey.TRAIN:
            runner.visualizer.plot(
                'train_loss',
                runner.iter,
                runner.average_meter.output['loss']
            )
