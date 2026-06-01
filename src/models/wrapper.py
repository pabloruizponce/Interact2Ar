import argparse

import lightning as L
import torch

from models.interact2ar import Interact2ArModel
from models.utils import CosineWarmupScheduler


class LitInteract2ArModel(L.LightningModule):
    """Lightning module that trains and serves the Interact2Ar model."""

    def __init__(self, opt_diffusion, opt_denoiser, opt_train, autoregressive=False):
        """
        Initializes the Lightning module with the specified options.

        Args:
            opt_diffusion: Configuration options for the diffusion process.
            opt_denoiser: Configuration options for the denoiser model.
            opt_train: Configuration options for training.
        """

        super().__init__()

        # Options for the diffusion model, denoiser, and training. Copy the
        # dictionaries so Lightning checkpoints store plain serializable values.
        self.opt_diffusion = dict(opt_diffusion)
        self.opt_denoiser = dict(opt_denoiser)
        self.opt_train = dict(opt_train)

        if not autoregressive:
            raise ValueError(
                "This public release keeps only the autoregressive Interact2Ar model."
            )

        # The main-paper checkpoint uses the top-level Interact2Ar model.
        self.model = Interact2ArModel(
            argparse.Namespace(**self.opt_diffusion),
            argparse.Namespace(**self.opt_denoiser),
        )
        self.is_autoregressive = True

        # Store the exact option dictionaries inside Lightning checkpoints.
        self.save_hyperparameters()

    def setup(self, stage: str):
        """Move device-specific modules after Lightning places the model.

        Args:
            stage: Lightning stage name, such as ``fit`` or ``test``.
        """
        # The setup hook runs once per process, after devices are assigned.
        print(f"Inside setup() on global rank {self.global_rank}: {self.device}")

        # Keep the SMPL-X body model on the same device as diffusion losses.
        self.model.diffusion_train.bm.to(self.device)

    def forward(self, batch, first_frame=None):
        """Generate motion from a dataloader or inference batch.

        Args:
            batch: Batch tuple containing captions, lengths, and optional cached text tensors.
            first_frame: Optional first frame used by the paper evaluation protocol.

        Returns:
            torch.Tensor: Generated motion sequence.
        """
        return self.model(batch, first_frame=first_frame)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): A tuple containing the input data for the training step.
                - word_embeddings (torch.Tensor): Word embeddings of shape (batch_size, T, d_model).
                - pos_one_hots (torch.Tensor): Positional embeddings of shape (batch_size, T, d_model).
                - caption (list of str): List of captions for the batch.
                - sent_len (torch.Tensor): Lengths of the sentences in the batch.
                - motion (torch.Tensor): Motion data of shape (batch_size, T, input_feats).
                - m_length (torch.Tensor): Lengths of the motions in the batch.
                - _ (any): Placeholder for additional data, not used here.
        batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The total loss for the training step.
            Logs individual losses for better monitoring.
        """
        total_loss, losses = self.model.compute_loss(batch)

        if self.global_rank == 0:
            batch_size = batch[4].shape[0]
            has_logger = self.logger is not None
            self.log(
                "train_loss",
                total_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=has_logger,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch_size,
            )

            # Log individual losses for better monitoring
            for loss_name, loss_value in losses.items():
                self.log(
                    f"train_loss/{loss_name}",
                    loss_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=has_logger,
                    sync_dist=False,
                    rank_zero_only=True,
                    batch_size=batch_size,
                )

            # Get and log the learning rate
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "learning_rate",
                lr,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=has_logger,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=batch_size,
            )

        return total_loss

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for the model."""
        opt = argparse.Namespace(**self.opt_train)

        # Set up the optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.LEARNING_RATE,
            weight_decay=opt.WEIGHT_DECAY,
        )

        # Set up the learning rate scheduler
        milestone_step = int(opt.EPOCHS * opt.LR_SCH_STEP)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[milestone_step], gamma=opt.LR_SCH_GAMA
        )

        scheduler2 = CosineWarmupScheduler(
            optimizer,
            warmup=10,
            max_iters=opt.EPOCHS,
        )

        return [optimizer], [scheduler1, scheduler2]
