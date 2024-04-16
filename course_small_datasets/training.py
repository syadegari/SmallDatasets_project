from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass, field
import torch


def train_epoch(epoch, model, optimizer, loss_fn, loader, print_every, device):
    model.train()
    train_loss = 0
    mse_loss, kld_loss = 0, 0
    for _, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, _mse, _kld = loss_fn(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        mse_loss += _mse
        kld_loss += _kld
        optimizer.step()
    if epoch % print_every == 0:
        print(
            f"====> Epoch: {epoch} Average training loss {train_loss / len(loader.dataset):.2f}"
        )
    return train_loss / len(loader.dataset), mse_loss / len(loader.dataset), kld_loss / len(loader.dataset)


def validation_epoch(epoch, model, loss_fn, loader, print_every, device):
    validation_loss = 0
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_fn(recon_batch, data, mu, logvar)
            validation_loss += loss.item()
    if epoch % print_every == 0:
        print(
            f"====> Epoch: {epoch} Average validation loss {validation_loss / len(loader.dataset):.2f}"
        )
    return validation_loss / len(loader.dataset)


def check_overfit(model_object, check_for_overfit, relative_threshold=0.05, patience=5):
    if not check_for_overfit or len(model_object.val_losses) < patience + 1:
        return False

    avg_recent_val_loss = sum(model_object.val_losses[-(patience + 1) : -1]) / patience
    avg_recent_train_loss = (
        sum(model_object.train_losses[-(patience + 1) : -1]) / patience
    )
    training_improving = model_object.train_losses[-1] < avg_recent_train_loss
    relative_increase_val_loss = (
        model_object.val_losses[-1] - avg_recent_val_loss
    ) / avg_recent_val_loss

    if training_improving and relative_increase_val_loss > relative_threshold:
        print(
            f"Overfitting at epoch {model_object.epoch}: Validation loss increased by {relative_increase_val_loss:.2f}, despite training loss decreasing."
        )
        return True

    return False


@dataclass
class ModelObj:
    val_losses: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    best_val_loss: float = float("inf")
    epoch: int = 0
    in_memory_checkpoint: dict = field(default_factory=dict)

    def save_best_result(self, model):
        self.update_checkpoint_in_memory(model)

    def _get_checkpoint(self, model):
        return {
            "best_val_loss": self.best_val_loss,
            "val_losses": self.val_losses,
            "train_losses": self.train_losses,
            "model_state_dict": model.state_dict(),
            "epoch": self.epoch,
        }

    def update_checkpoint_in_memory(self, model):
        self.in_memory_checkpoint = {
            "best_val_loss": self.best_val_loss,
            "val_losses": self.val_losses,
            "train_losses": self.train_losses,
            "model_state_dict": model.state_dict(),
            "epoch": self.epoch,
        }

    def save_checkpoint(self, path, model):
        checkpoint = self._get_checkpoint(model)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, model, device="cpu"):
        checkpoint = torch.load(path)
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        model.load_state_dict(checkpoint["model_weights"])
        model.to(device)

    def update(self, model, train_loss, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_best_result(model)

        self.epoch += 1
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)


def cleanup(model_object: ModelObj, save_path: str, model) -> str:
    model_object.save_checkpoint(
        f"{save_path}/last_epoch_{model_object.epoch}.pth", model
    )
    best_model_path = f"{save_path}/best_epoch_{model_object.in_memory_checkpoint['epoch']}.pth"
    torch.save(model_object.in_memory_checkpoint, best_model_path)
    return best_model_path


def train(
    epochs,
    model,
    optimizer,
    loss_fn,
    train_loader,
    test_loader,
    save_every,
    print_every,
    check_for_overfit=True,
    grace_for_overfit=500,
    patience_for_overfit=5,
    relative_val_loss_threshold_for_overfit=0.01,
    restart=False,
    save_path="",
    device="cpu",
):

    writer = SummaryWriter(save_path)

    if not restart:
        model_obj = ModelObj()
    else:
        model_obj = ModelObj()
        model_obj.load_checkpoint(save_path, model, device)

    for epoch in range(model_obj.epoch + 1, epochs + 1):
        train_loss, mse_loss, kld_loss = train_epoch(
            epoch, model, optimizer, loss_fn, train_loader, print_every, device
        )
        val_loss = validation_epoch(
            epoch, model, loss_fn, test_loader, print_every, device
        )
        model_obj.update(model, train_loss, val_loss)

        writer.add_scalars("Total Loss", {"train": train_loss, "validation": val_loss}, epoch)
        writer.add_scalars("Individual Loss Items", {"train MSE loss": mse_loss, "train KLD loss": kld_loss}, epoch)


        if epoch % save_every and save_every > 0:
            model_obj.save_checkpoint(f"{save_path}/epoch_{epoch}.pth", model)

        if epoch > grace_for_overfit:
            overfit = check_overfit(
                model_obj,
                check_for_overfit,
                patience=patience_for_overfit,
                relative_threshold=relative_val_loss_threshold_for_overfit,
            )
            if overfit:
                break

    best_model_path = cleanup(model_obj, save_path, model)
    return best_model_path
