# Классификатор изображений еды Food-101
## Датасет
Ссылка на оригинальный датасет: https://www.kaggle.com/datasets/dansbecker/food-101/
## Обучение модели
Для разработки модели использовался Transfer Learning на основе архитектуры [DenseNet-121](https://arxiv.org/abs/1608.06993).

DenseNet-121 имеет достаточную большую точность в задач классификации изображений и небольшое количество параметров, что способствует более быстрой оптимизации параметров нейронной сети.

Итоговая метрика Accuracy: ##.##%.

[Notebook](https://colab.research.google.com/drive/1AdgtEQqZuU78c8bTpXY7qCkYK2piMfPa#scrollTo=LXStGFMPoKok) с процессом обучения.
## Итоговый продукт
Для итоговой модели разработано небольшое Web приложение на Flask с возможностью загрузки отдельного изображения или архива с изображениями для идентификации еды на изображение.

Приложение выложено на бесплатном хостинге Render.

Ссылка на приложение: https://food-classifier-zg2t.onrender.com

## Запуск приложения
Для запуска проекта необходимо выполнить команды:
```bash
pip install -r requirements.txt
python main.py
```
