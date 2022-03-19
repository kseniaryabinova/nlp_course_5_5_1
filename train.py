from collections import OrderedDict
import shutil

from catalyst import dl
from catalyst.contrib.nn import FocalLossMultiClass
from catalyst.runners import SupervisedRunner
from catalyst.utils import set_global_seed
from clearml import Task
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.const import INPUTS, TARGETS, LOGITS, TRAIN, VALID, LOSS, \
    PROCESSED_LOGITS, PROCESSED_TARGETS
from src.dataset import get_dataloaders
from src.model import Seq2Seq
from src.train_utils import init_weights, FocalLoss

if __name__ == '__main__':
    task = Task.init(
        project_name='test_project',
        task_name='remote_execution nlp stepik 5_5_1',
    )

    task.execute_remotely(queue_name="default")

    set_global_seed(25)
    config = Config()
    train_dataloader, valid_dataloader = get_dataloaders(config)
    batch = next(iter(train_dataloader))

    model = Seq2Seq(config)
    model.apply(init_weights)
    model = model.to(config.device)

    pad_index = 0

    runner = SupervisedRunner(
        model=model,
        input_key=INPUTS,
        output_key=LOGITS,
        target_key=TARGETS,
    )

    callbacks = [
        dl.BatchTransformCallback(
            transform=lambda x: x[1:].view(-1, x.shape[-1]),
            scope='on_batch_end',
            input_key=LOGITS,
            output_key=PROCESSED_LOGITS,
        ),
        dl.BatchTransformCallback(
            transform=lambda x: x[1:].flatten(),
            scope='on_batch_end',
            input_key=TARGETS,
            output_key=PROCESSED_TARGETS,
        ),
        dl.CriterionCallback(
            input_key=PROCESSED_LOGITS,
            target_key=PROCESSED_TARGETS,
            metric_key=LOSS,
        ),

        dl.OptimizerCallback(
            metric_key=LOSS,
            accumulation_steps=config.gradient_accumulation,
        ),
        dl.CheckpointCallback(
            logdir='checkpoints',
            loader_key=VALID,
            minimize=True,
            metric_key=LOSS,
            mode='model',
        )
    ]

    loss = FocalLoss(
        pad_index=pad_index,
        label_smoothing=config.label_smoothing,
    )
    # loss = CrossEntropyLoss(ignore_index=pad_index)
    optimizer = AdamW(lr=config.lr, params=model.parameters())
    scheduler = config.scheduler(optimizer=optimizer, **config.scheduler_kwargs)

    runner.train(
        loaders=OrderedDict({TRAIN: train_dataloader, VALID: valid_dataloader}),
        model=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        seed=config.seed,
        num_epochs=config.epochs,
        valid_metric=LOSS,
        valid_loader=VALID,
        minimize_valid_metric=True,
        verbose=True,
        check=False,
        amp=False,
    )

    shutil.copy(
        '/root/.clearml/venvs-builds/3.7/task_repository/nlp_course_5_5_1.git/checkpoints/best.pth',
        '/tmp/best.pth',
    )
