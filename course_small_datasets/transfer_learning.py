import torch
import torch.nn as nn

from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class TravelPlanner(pl.LightningModule):

    def __init__(self, model_name="vgg16", out_features=3):
        super(TravelPlanner, self).__init__()

        self.SUPPORTED_MODELS = ["vgg16", "resnet50", "densenet121", "inception_v3"]
        self.model_name = model_name

        if model_name == "vgg16":
            self.model = models.vgg16(weights=True)
            self.freeze()
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, out_features)

        elif model_name == "resnet50":
            self.model = models.resnet50(weights=True)
            self.freeze()
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, out_features)

        elif model_name == "densenet121":
            self.model = models.densenet121(weights=True)
            self.freeze()
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, out_features)

        elif model_name == "inception_v3":
            self.model = models.inception_v3(weights=True)
            self.freeze()
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, out_features)

        else:
            raise ValueError(
                f"Model '{model_name}' is not supported. Choose from {', '.join(self.SUPPORTED_MODELS)}."
            )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def get_accuracy(self, preds, labels):
        return (preds.argmax(dim=-1) == labels).float().mean()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.get_accuracy(outputs, labels)
        self.log_dict({"loss": loss, "acc": acc})
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.get_accuracy(outputs, labels)
        self.log("val_loss", loss)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        acc = self.get_accuracy(outputs, labels)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer


def train(
    model_name,
    epochs,
    train_loader,
    validation_loader,
    test_loader,
    overfit_patience,
    device="cpu",
):
    model = TravelPlanner(model_name, out_features=3)
    checkpoint = ModelCheckpoint(
        "saved_models",
        filename=f"{model_name}_" + "{epoch:04d}_{val_loss:.2f}",
        save_top_k=1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="tboard", name="travel_planner", version="vgg16"
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=overfit_patience, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator=device,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[early_stop, checkpoint],
    )
    trainer.fit(model, train_loader, validation_loader)
    best_model_path = checkpoint.best_model_path
    model.load_state_dict(torch.load(best_model_path)["state_dict"])
    test_results = trainer.test(model, test_loader)
    return test_results, best_model_path
