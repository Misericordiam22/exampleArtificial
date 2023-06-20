# exampleArtificial
TestArtificial_Intel
Данная модель используя библиотеки и модель huggingface обрабатывает изображения и классифицирует кошек и собак 
Код на языке python
Необходимые библиотеки(pip install):
  transformers
  torch
  pillow


Подробное описание работы кода и комментарии
import transformers as tf
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from PIL import Image

# Загрузка датасета с изображениями котов и собак
dataset = load_dataset("Bingsu/Cat_and_Dog")

# Выбор нужного изображения
image_path = 'C:/Users/sunrise/Desktop/AIProject/vit_finetuned_cats_d/test_pictures/cat.jpeg'
image = Image.open(image_path).convert("RGB")

# Определение классов и их меток
types = {"cat": 0,
         "dog": 1}

# Получение номера изображения или выбор изображения из локального хранилища
num = input('Введите номер картинки от 0 - 7999 если тренировочная выборка и 0 - 1999 если тест,'
            'или loc чтобы взять картинку из локального хранилища\n')
if num != 'loc':
    image = dataset["test"]["image"][int(num)]
    typeIm = dataset["test"]["labels"][int(num)]
else:
    typeIm = input('Введите что на картинке 0 - cat; 1 - dog\n')

# Визуализация изображения
with image as im:
    im.rotate(45).show()

# Создание модели ViT для обработки и классификации изображений
processor = ViTImageProcessor.from_pretrained("nickmuchi/vit-finetuned-cats-dogs")
model = ViTForImageClassification.from_pretrained("nickmuchi/vit-finetuned-cats-dogs")

# Подготовка изображения для входа в модель
inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# Получение предсказанного класса изображения и сравнение с реальным классом
predicted_label = logits.argmax(-1).item()
result = model.config.id2label[predicted_label]
print(result)
print(types[result] == typeIm)

start = 900
i = start
score = 0
end = 1100
needCheckAccuracy = bool(int(input('Провести тест точности 1/0?\n')))

# Проверка точности модели
if(needCheckAccuracy):
    while (i < end):
        image = dataset["test"]["image"][i]    
        typeIm = dataset["test"]["labels"][i]
        inputs = processor(image, return_tensors="pt") 
        with torch.no_grad():
            logits = model(**inputs).logits    
        predicted_label = logits.argmax(-1).item()
        result = model.config.id2label[predicted_label]
        if (types[result] == typeIm): 
            score += 1
        i += 1
        print(i)

    accuracy = score/(end - start)
    print(accuracy)
