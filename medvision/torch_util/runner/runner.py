import torch
from .average_meter import AverageMeter
from .hook import Hook
import medvision as mv


class Runner:
    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None):
        """ A training helper for PyTorch.

        Args:
            model (:obj:`torch.nn.Module`): The model to be run.
            batch_processor (callable): A callable method that process a data
                batch. The interface of this method should be
                `batch_processor(model, data, train_mode) -> dict`
            optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
                runner will construct an optimizer according to it.
            work_dir (str, optional): The working directory to save checkpoints
                and logs.
        """
        assert isinstance(model, torch.nn.Module)
        assert callable(batch_processor)
        assert isinstance(optimizer, (str, torch.optim.Optimizer))
        assert isinstance(work_dir, str) or work_dir is None

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = self.build_optimizer(optimizer)
        self.work_dir = mv.abspath(work_dir if work_dir is not None else '.')
        self.experiment = mv.basename(self.work_dir)
        self.average_meter = AverageMeter()

        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

        self.mode = None

        # create work_dir
        mv.mkdirs(self.work_dir)

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

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = mv.ModeKey.TRAIN

        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)

        self.call_hook('before_train_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, is_train=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.average_meter.update(
                    outputs['log_vars'],
                    outputs['num_samples']
                )
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = mv.ModeKey.VAL

        self.data_loader = data_loader

        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
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
                    outputs['num_samples']
                )
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run_(self, data_loaders, workflow, **kwargs):
        while self.epoch < self.max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, mv.ModeKey):
                    epoch_runner = getattr(self, mode.value)
                elif callable(mode):  # customized epoch runner
                    epoch_runner = mode
                else:
                    raise TypeError(
                        'mode in workflow must be a ModeKey or a callable '
                        'function, not {}'.format(type(mode))
                    )
                for _ in range(epochs):
                    if (mode == mv.ModeKey.TRAIN and
                            self.epoch >= self.max_epochs):
                        return
                    epoch_runner(data_loaders[i], **kwargs)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert isinstance(workflow, (tuple, list))
        assert len(data_loaders) == len(workflow)

        self._epoch = 0
        self._max_epochs = max_epochs

        self.call_hook('before_run')
        self.run_(data_loaders, workflow, max_epochs, **kwargs)
        self.call_hook('after_run')

    def build_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

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
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    def register_logger_hooks(self, experiment='default'):
        pass
