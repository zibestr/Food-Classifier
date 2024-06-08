import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation, Resize,
                                    ToTensor)


class FoodDataset(Dataset):
    '''Класс датасета'''
    def __init__(self,
                 data_directory: str,
                 image_size: int,
                 normalize_means: list[float],
                 normalize_stds: list[float],
                 train: bool = True,
                 device: str = 'auto') -> None:
        if device == 'auto':
            self.device = (
                            'cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else
                            'cpu'
                          )
        else:
            self.device = device

        self.data_directory = data_directory
        with open(f'{self.data_directory}/meta/classes.txt') as file:
            self.labels = {key: value for value, key in enumerate(
                file.read().split('\n')
            )}
        if train:
            with open(f'{self.data_directory}/meta/train.txt') as file:
                self.image_files = file.read().split('\n')[:-1]
        else:
            with open(f'{self.data_directory}/meta/test.txt') as file:
                self.image_files = file.read().split('\n')[:-1]

        if train:
            self._transformer = Compose([
                RandomRotation(30),
                RandomHorizontalFlip(),
                RandomResizedCrop((image_size, image_size)),
                ToTensor()
            ])
        else:
            self._transformer = Compose([
                Resize((image_size, image_size)),
                ToTensor()
            ])
        self._normalize = Normalize(normalize_means, normalize_stds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor,
                                               torch.Tensor]:
        image_name = self.image_files[index]
        image = Image.open(f'{self.data_directory}/images/{image_name}.jpg')
        label = image_name.split('/')[0]

        target = torch.tensor(self.labels[label],
                              device=self.device,
                              dtype=torch.float32)
        inputs = self._transformer(image).to(self.device,
                                             dtype=torch.float32)
        return (self._normalize(inputs), target)

    def __len__(self) -> int:
        return len(self.image_files)
