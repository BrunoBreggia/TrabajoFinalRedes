import torch
from torch import nn
from torch.optim import Optimizer
from pathlib import Path
import json
from tqdm import tqdm
from pathlib import Path

class TrainerEncoder:

    def __init__(self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        loss_criterion: nn.Module,
        val_dataloader: torch.utils.data.DataLoader = None,
        device: str | None = None,
        early_stopping_patience: int | None = 10,
        tensorboard=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando dispositivo: {device.upper()}")

        self.device = device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.early_stopping_patience = early_stopping_patience

        self.training_losses = []
        self.validation_losses = []
        self.best_validation_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0

        self.writer = tensorboard
        ######################## LOG TO TENSORBOARD ########################
        if self.writer:
            examples = iter(self.train_dataloader)
            samples, labels = next(examples)
            self.writer.add_graph(model, samples.reshape(-1, 1, 28, 28).to(self.device))
        ####################################################################

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
                loss = self.loss_criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.training_losses.append(avg_epoch_loss)

            print(f"Época {epoch+1}/{max_epochs} — Pérdida de entrenamiento: {avg_epoch_loss:.4f}")

            ############## LOG TO TENSORBOARD ##############
            self.writer.add_scalar('avg epoch loss (train)', avg_epoch_loss, epoch)
            ################################################

            if self.val_dataloader:
                val_loss = self.validate()
                self.validation_losses.append(val_loss)

                ############## LOG TO TENSORBOARD ##############
                self.writer.add_scalar('avg epoch loss (val)', val_loss, epoch)
                ################################################

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

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, inputs)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_dataloader)

        print(f"Pérdida de validación: {avg_val_loss:.4f}")
        return avg_val_loss

    def save_checkpoint(self, save_to:Path|None=None):
        self.best_model_state = self.model.state_dict().copy()
        if save_to:
            model_path = save_to/"best_model.pth"
            torch.save(self.model.state_dict(), model_path)

class TrainerClassifier:

    def __init__(self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        loss_criterion: nn.Module,
        val_dataloader: torch.utils.data.DataLoader = None,
        device: str | None = None,
        early_stopping_patience: int | None = 10):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Usando dispositivo: {device.upper()}")
        

        self.device = device
        self.model = model.to(self.device)
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
    save_directory.mkdir(parents=True, exist_ok=True)
    # save configuration
    config_path = save_directory/"config.txt"
    attributes = repr(model.__getattr__).split("\n")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines([line + "\n" for line in attributes[1:-1]])

    # save model
    model_path = save_directory/"trained_model.pth"
    torch.save(model.state_dict(), model_path)
