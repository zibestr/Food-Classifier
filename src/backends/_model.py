import torch
from torch import nn
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize
)


def get_model(device: str = 'auto') -> torch.nn.Module:
    if device == 'auto':
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    model = torch.load('resources/models/model_resnet101.pth')
    return model.to(device)


class ImageClassifier:
    def __init__(
        self,
        labels: list[str],
        image_size: int,
        normalize_means: list[float],
        normalize_stds: list[float],
        device: str = 'auto'
    ) -> None:
        if device == 'auto':
            self.device = (
                            'cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else
                            'cpu'
                          )
        else:
            self.device = device

        self.labels = labels
        self.model = get_model(self.device)
        self._image_size = image_size

        self._preprocessor = Compose([
            Resize((image_size, image_size)),
            ToTensor()
        ])
        self._normalize = Normalize(normalize_means, normalize_stds)

    def _predict_one(self, image: Image.Image) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            input_tensor = self._preprocessor(image).to(
                self.device,
                dtype=torch.float32
            ) / 255
            input_tensor = self._normalize(input_tensor).reshape(
                1,
                3,
                self._image_size,
                self._image_size
            )

            logits = self.model(input_tensor)

        return {name: round(nn.functional.sigmoid(value).item() * 100, 2)
                for name, value in zip(self.labels, logits[0])}

    def predict(self, images: list[Image.Image]) -> list[dict[str, float]]:
        return [self._predict_one(image) for image in images]
