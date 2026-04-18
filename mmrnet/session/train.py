from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .wrapper import ModelWrapper
from lightning_utilities.core.rank_zero import rank_zero_info


class EpochSummaryCallback(pl.Callback):
    """Print one concise metrics line per epoch (no progress bar spam)."""

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        epoch = trainer.current_epoch

        def _fmt(key: str) -> str:
            v = m.get(key, None)
            if v is None:
                return "NA"
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        # wrapper logs: loss, acc, val_loss, val_acc (or mle variants)
        parts = [
            f"epoch={epoch}",
            f"loss={_fmt('loss')}",
            f"{pl_module.metric_name}={_fmt(pl_module.metric_name)}",
            f"val_loss={_fmt('val_loss')}",
            f"val_{pl_module.metric_name}={_fmt(f'val_{pl_module.metric_name}')}",
        ]
        rank_zero_info(" | ".join(parts))


def train(
        model,
        train_loader, val_loader,
        optimizer, 
        learning_rate, 
        weight_decay,
        plt_trainer_args, 
        save_path):
    plt_model = ModelWrapper(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=plt_trainer_args['max_epochs'],
        optimizer=optimizer)
    metric = f'val_{plt_model.metric_name}'
    if 'mle' in metric:
        mode = 'min'
    elif 'acc' in metric:
        mode = 'max'
    else:
        raise ValueError(f'Unknown metric {metric}')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric,
        mode=mode,
        filename="best",
        dirpath=save_path,
        save_last=False,
    )
    plt_trainer_args['callbacks'] = [checkpoint_callback, EpochSummaryCallback()]
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(plt_model, train_loader, val_loader)
    return plt_model.best_val_loss
