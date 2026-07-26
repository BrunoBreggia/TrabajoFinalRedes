import torch
from torch import nn
from torch.optim import Optimizer
from pathlib import Path
import json
from tqdm import tqdm
from pathlib import Path

class Trainer:

    def __init__(self,
        model: torch.nn.Module,
        model_type:str,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        loss_criterion: nn.Module,
        val_dataloader: torch.utils.data.DataLoader = None,
        device: str | None = None,
        early_stopping_patience: int | None = 10):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando dispositivo: {device.upper()}")
        
        assert model_type in ["autoencoder", "classifier"], "Model type must be either autoencoder' or 'classifier'"

        self.device = device
        self.model = model.to(self.device)
        self.model_tpye = model_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.early_stopping_patience = early_stopping_patience

        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.best_validation_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def fit(self, max_epochs:int, verbose:bool=False, save_best_dir:Path|None=None):
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0

            if verbose:
                train_loader = tqdm(self.train_dataloader)
            else:
                train_loader = self.train_dataloader

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # different loss procedures depending on architectures trained
                if self.model_tpye == "autoencoder":
                    loss = self.loss_criterion(outputs, inputs)
                elif self.model_tpye == "clssifier":
                    loss = self.loss_criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.training_losses.append(avg_epoch_loss)

            print(f"Época {epoch+1}/{max_epochs} — Pérdida de entrenamiento: {avg_epoch_loss:.4f}")

            if self.val_dataloader:
                val_loss, val_acc = self.validate()
                self.validation_losses.append(val_loss)
                self.validation_accuracies.append(val_acc)

                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                    self.save_checkpoint(save_to=save_best_dir)
                    print(f"Modelo guardado en epoca {epoch+1}")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if (
                    self.early_stopping_patience is not None and
                    self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    print(f"Se detuvo el entrenamiento anticipadamente en la época {epoch+1} (sin mejoras por {self.early_stopping_patience} épocas).")
                    break

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, targets)
                total_loss += loss.item()

                predicted = outputs.argmax(dim=1)
                targets = targets.argmax(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = total_loss / len(self.val_dataloader)
        accuracy = correct / total if total > 0 else 0.0

        print(f"Pérdida de validación: {avg_val_loss:.4f} — Accuracy: {accuracy:.4f}")
        return avg_val_loss, accuracy

    def save_checkpoint(self, save_to:Path|None=None):
        self.best_model_state = self.model.state_dict().copy()
        if save_to:
            model_path = save_to/"best_model.pth"
            torch.save(self.model.state_dict(), model_path)
        
def save_model(model, save_directory:Path):
    # save configuration
    config_path = save_directory/"config.json"
    attributes = model.__dict__
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(shallow_inspect_dict, f, indent=4)

    # save model
    model_path = save_directory/"trained_model.pth"
    torch.save(model.state_dict(), model_path)
