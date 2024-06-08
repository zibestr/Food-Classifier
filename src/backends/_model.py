import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_model(device: str = 'auto') -> torch.nn.Module:
    '''
    Возвращает ResNet-50 с обученным на Food-101 слоем fc
    '''
    if device == 'auto':
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.load('resources/models/resnet50_fc.pth')
    return model.to(device)


class ImageClassifier:
    '''
    Класс классификатора изображений
    '''
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
            )
            input_tensor = self._normalize(input_tensor).reshape(
                1,
                3,
                self._image_size,
                self._image_size
            )

            logits = self.model(input_tensor)

        return {name: value.item()
                for name, value in zip(self.labels, logits[0])}

    def predict(self, images: list[Image.Image]) -> list[dict[str, float]]:
        return [self._predict_one(image) for image in images]
