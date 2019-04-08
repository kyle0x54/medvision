import torch
from .average_meter import AverageMeter
from .hook import Hook
import medvision as mv


class Runner:
    def __init__(self,
                 mode,
                 model,
                 batch_processor,
                 train_dataloader=None,
                 val_dataloader=None,
                 optimizer=None,
                 work_dir=None,
                 max_epochs=10000):
        """ A training helper for PyTorch.

        Args:
            model (`torch.nn.Module`): The model to be run.
            mode ('ModeKey'): running mode.
            batch_processor (callable): A callable method that process a data
                batch. The interface of this method should be
                `batch_processor(model, data, train_mode) -> dict`
            train_dataloader ('DataLoader'): train data loader.
            val_dataloader ('DataLoader'): validation data loader.
            optimizer (dict or `Optimizer`): If it is a dict, runner will
                construct an optimizer according to it.
            work_dir (str, optional): The working directory to save
                checkpoints, logs and other outputs.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(mode, mv.ModeKey)
        assert isinstance(model, torch.nn.Module)
        assert callable(batch_processor)
        assert isinstance(optimizer, (str, torch.optim.Optimizer))
        assert isinstance(work_dir, str) or work_dir is None
        assert isinstance(max_epochs, int)

        self.mode = mode
        self.epoch_runner = getattr(self, mode.value)
        self.model = model
        self.batch_processor = batch_processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = self.build_optimizer(optimizer)

        # create work_dir
        self.work_dir = mv.abspath(work_dir if work_dir is not None else '.')
        mv.mkdirs(self.work_dir)

        # init TensorboardX visualizer and dataloader
        if mode == mv.ModeKey.TRAIN:
            experiment = mv.basename(self.work_dir)
            self.visualizer = mv.TensorboardVisualizer(experiment)
            self.dataloader = self.train_dataloader
        else:
            self.visualizer = None
            self.dataloader = self.val_dataloader

        # init hooks and average meter
        self._hooks = []
        self.average_meter = AverageMeter()

        # init loop parameters
        self._epoch = 0
        self._max_epochs = max_epochs if mode == mv.ModeKey.TRAIN else 1
        self._inner_iter = 0
        self._iter = 0
        self._max_iters = 0

        # get model name from model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)

        hook.priority = mv.get_priority(priority)
        # insert the hook to a sorted list
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                break
        else:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def train(self, **kwargs):
        self.model.train()
        self._max_iters = self._max_epochs * len(self.train_dataloader)
        self.call_hook('before_train_epoch')

        for i, data_batch in enumerate(self.train_dataloader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, is_train=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.average_meter.update(
                    outputs['log_vars'],
                    outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, **kwargs):
        self.model.eval()
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(self.val_dataloader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, is_train=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.average_meter.update(
                    outputs['log_vars'],
                    outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        self._epoch += 1

    def run(self, **kwargs):
        self.call_hook('before_run')
        while self.epoch < self.max_epochs:
            self.epoch_runner(self.dataloader, **kwargs)
        self.call_hook('after_run')

    def build_optimizer(self, optimizer):
        """ Init the optimizer.

        Args:
            optimizer (dict or `Optimizer`): Either an optimizer object
                or a dict used for constructing the optimizer.

        Returns:
            (`Optimizer`): An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = mv.build_object_from_dict(
                optimizer, torch.optim, dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def current_lr(self):
        """Get current learning rates.

        Returns:
            (list): Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    @property
    def hooks(self):
        return self._hooks

    @property
    def model_name(self):
        return self._model_name

    @property
    def epoch(self):
        return self._epoch

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def iter(self):
        return self._iter

    @property
    def max_iters(self):
        return self._max_iters

    def register_logger_hooks(self):
        pass
