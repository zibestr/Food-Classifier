# Классификатор изображений еды Food-101
## Датасет
Ссылка на оригинальный датасет: https://www.kaggle.com/datasets/dansbecker/food-101/
## Обучение модели
Для разработки модели использовался Transfer Learning на основе архитектуры [ResNet-18](https://arxiv.org/abs/1512.03385).

В основе ResNet-18 лежит концепция остаточного обучения, которая повышает скорость обучения и позволяет увеличить точность самой сети с помощью увеличения глубины.

Итоговая метрика Accuracy на тестовом датасете: 38.87% (Точность получилась не очень большая из-за недостатка ресурсов и времени на процесс обучение).

[Notebook](https://colab.research.google.com/drive/1AdgtEQqZuU78c8bTpXY7qCkYK2piMfPa#scrollTo=LXStGFMPoKok) с процессом обучения.
## Итоговый продукт
Для итоговой модели разработано небольшое Web приложение на Flask с возможностью загрузки отдельного изображения или архива с изображениями для идентификации еды на изображение.

Приложение выложено на бесплатном хостинге Render.

Ссылка на приложение: https://food-classifier-zg2t.onrender.com

## Запуск приложения
Для запуска проекта необходимо выполнить команды:
```bash
pip install -r requirements.txt
gunicorn main:app
```
