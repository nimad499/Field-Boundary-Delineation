import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from pathlib import Path

# ToDo: Add validation and improve logging
# ToDo: Add LR Scheduler
# ToDo: Track the best model
# ToDo: Add checkpoint


class BaseTrain(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        output_dir: Path,
        device=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.device = device
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

    @abstractmethod
    def _dataloader(self, dataset, batch_size): ...

    @abstractmethod
    def _train_one_epoch(self, dataloader): ...

    def _log(self, epoch, loss):
        print(f"Epoch {epoch}: {loss}")

    def train(self, dataset, num_epoch, batch_size):
        dataloader = self._dataloader(dataset, batch_size)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "log").mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_epoch = float("inf")
        for epoch in range(1, num_epoch + 1):
            loss = self._train_one_epoch(dataloader)
            self._log(epoch, loss)

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss,
                    },
                    self.output_dir / "model" / "best_model.tar",
                )


class InstanceSegmentationTrain(BaseTrain):
    def _dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    def _train_one_epoch(self, dataloader):
        self.model = self.model.to(self.device)
        self.model.train()

        for images, targets in dataloader:
            images = [image.to(self.device) for image in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]

            loss_dict = self.model(images, targets)

            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()
