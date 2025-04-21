from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ToDo: Add validation
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
        current_epoch=0,
        log_queue=None,
        loss_callback_list=None,
        cancel_callback=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.current_epoch = current_epoch
        self.log_queue = log_queue
        self.loss_callback_list = loss_callback_list
        self.cancel_callback = cancel_callback

    @abstractmethod
    def _dataloader(self, dataset, batch_size): ...

    @abstractmethod
    def _train_one_epoch(self, dataloader): ...

    def _log(self, msg):
        if self.log_queue:
            self.log_queue.put(msg + "\n")
        else:
            print(msg)

    def train(self, dataset, num_epochs, batch_size):
        dataloader = self._dataloader(dataset, batch_size)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "log").mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.output_dir / "log")

        self.model = self.model.to(self.device)
        self.model.train()
        best_loss = float("inf")
        best_epoch = float("inf")
        for epoch in range(self.current_epoch + 1, num_epochs + self.current_epoch + 1):
            if self.cancel_callback and self.cancel_callback():
                self._log("Training cancelled.")
                break

            loss = self._train_one_epoch(dataloader)

            self.writer.add_scalar("Loss/train", loss, epoch)

            if self.loss_callback_list is not None:
                self.loss_callback_list.append(loss)

            self._log(f"Epoch {epoch}: {loss}")
            self.current_epoch = epoch

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model": self.model,
                        "optimizer": self.optimizer,
                        "loss": loss,
                    },
                    self.output_dir / "model" / "best_model.tar",
                )

        self.writer.close()


class InstanceSegmentationTrain(BaseTrain):
    def _dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    def _train_one_epoch(self, dataloader):
        running_loss = 0.0
        num_batches = len(dataloader)
        for images, targets in dataloader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / num_batches
        return avg_loss


class SemanticSegmentationTrain(BaseTrain):
    def _dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    def _train_one_epoch(self, dataloader):
        running_loss = 0.0
        num_batches = len(dataloader)
        criterion = torch.nn.CrossEntropyLoss()
        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)

            loss = criterion(outputs["out"], masks)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / num_batches
        return avg_loss
