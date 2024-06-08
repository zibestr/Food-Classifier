from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Trainer:
    '''
    Класс для обучения и тестирования нейронной сети
    '''
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        optimizer: Literal[
            'adam',
            'adamax',
            'adagrad',
            'sgd'
        ] = 'adam',
        learning_rate=0.001,
        batch_size: int = 64
    ):
        self.train_data = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_data = DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        self.loss_func = loss_fn
        self.model = model

        self.optimizer = torch.optim.Optimizer(model.fc.parameters(),
                                               defaults={})
        match optimizer:
            case 'adam':
                self.optimizer = torch.optim.Adam(
                    model.fc.parameters(),
                    lr=learning_rate
                )
            case 'adamax':
                self.optimizer = torch.optim.Adamax(
                    model.fc.parameters(),
                    lr=learning_rate
                )
            case 'adagrad':
                self.optimizer = torch.optim.Adagrad(
                    model.fc.parameters(),
                    lr=learning_rate
                )
            case 'sgd':
                self.optimizer = torch.optim.SGD(
                    model.fc.parameters(),
                    lr=learning_rate
                )
            case _:
                raise ValueError(f'{optimizer} is wrong optimizer algorithm')
        print(f'Using {type(self.optimizer).__name__} optimizer.')

    def __train_iter(self,
                     inputs: torch.Tensor,
                     targets: torch.Tensor) -> float:
        logits = self.model(inputs)
        loss = self.loss_func(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_loop(self,
                    verbose: bool) -> np.ndarray:
        losses = np.zeros(len(self.train_data))

        self.model.train()
        for i, (inputs, targets) in enumerate(self.train_data):
            losses[i] = self.__train_iter(inputs, targets)

            if verbose:
                if (i + 1) % 10 == 0:
                    print(f'Train loop, batch {i + 1}/' +
                          f'{len(self.train_data)}, loss: {losses[i]}')
        return losses

    def fit(self,
            max_epoch: int = 5,
            verbose: bool = False) -> tuple[np.ndarray,
                                            np.ndarray]:
        train_losses = np.zeros(shape=(max_epoch, len(self.train_data)))

        for epoch in range(1, max_epoch + 1):
            print(f'Epoch {epoch}\n--------------------------------')
            train_losses[epoch - 1] = self._train_loop(verbose)
            print(f'Epoch {epoch} loss: {train_losses[epoch - 1].mean()}')

        return train_losses

    def calculate_accuracy(self) -> float:
        self.model.eval()
        dataset_size = 0
        true_predicted = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_data):
                dataset_size += len(targets)
                predicted = self.model(inputs).argmax(dim=1)
                true_predicted += torch.sum(predicted == targets).item()
                print(f'{i + 1}/{len(self.test_data)}', true_predicted, dataset_size)
        return true_predicted / dataset_size
