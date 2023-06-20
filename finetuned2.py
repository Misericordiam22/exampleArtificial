import transformers as tf
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from PIL import Image

# train(image, label 8000) or test(image, labels 2000) 
dataset = load_dataset("Bingsu/Cat_and_Dog") 
image_path = 'C:/Users/sunrise/Desktop/AIProject/vit_finetuned_cats_d/test_pictures/cat.jpeg' #TODO выбор картинки
image = Image.open(image_path).convert("RGB")
    
types = {"cat": 0,
          "dog": 1}

# выбираем картинку для теста
num = input('Введите номер картинки от 0 - 7999 если тренировочная выборка и 0 - 1999 если тест, или loc чтобы взять картинку из локального хранилища\n')
if num != 'loc':
    image = dataset["test"]["image"][int(num)]
    typeIm = dataset["test"]["labels"][int(num)]
else:
    typeIm = input('Введите что на картинке 0 - cat; 1 - dog\n')

with image as im:
    im.rotate(45).show()
    
# https://huggingface.co/datasets/hf-internal-testing/cats_vs_dogs_sample

processor = ViTImageProcessor.from_pretrained("nickmuchi/vit-finetuned-cats-dogs")

model = ViTForImageClassification.from_pretrained("nickmuchi/vit-finetuned-cats-dogs")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits


# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(predicted_label)
result = model.config.id2label[predicted_label]
print(result)
print(types[result] == typeIm)

start = 900
i = start
score = 0
end = 1100
needCheckAccuracy = bool(int(input('Провести тест точности 1/0?\n'))) #TODO проверка и ввод первого и последнего изображения

# Запуск проверки точности
if(needCheckAccuracy):
    while (i < end):    #если все 2к изображений то долго
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
# model.config
