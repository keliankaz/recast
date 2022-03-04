from typing import Optional, Tuple

import pytorch_lightning as pl
import torch

import eq


class TPPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def nll_loss(self, batch: eq.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[eq.data.Sequence] = None,
    ) -> eq.data.Batch:
        """
        Sample a batch of sequences from the TPP model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the time interval on which the sequence is simulated.

        Returns:
            batch: Batch of padded event sequences.
        """
        raise NotImplementedError

    def evaluate_intensity(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the intensity for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the intensity.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the intensity.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError

    def evaluate_compensator(
        self, sequence: eq.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the compensator for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the compensator.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the compensator.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.nll_loss(batch).mean()
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def test_step(self, batch, batch_idx, dataset_idx=None):
        with torch.no_grad():
            loss = self.nll_loss(batch).mean()
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )

    def configure_optimizers(self):
        if hasattr(self, "learning_rate"):
            lr = self.learning_rate
        else:
            lr = 1e-2
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
