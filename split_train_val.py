import os
import numpy as np
from tqdm import tqdm

np.random.seed(2018)


file_total = '/data0/milk_jar_data'

f_train = open("./train_file.txt", "w")
f_val = open("./val_file.txt", "w")

with open('./class_file.txt') as f:
    classes = f.readlines()

CarNames = [w.strip() for w in classes]

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for car in tqdm(CarNames):

    total_images = os.listdir(os.path.join(file_total, car))
    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)
    train_images = total_images[:nbr_train]
    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(file_total, car, img)
        if img.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            f_train.writelines(source + " " + car + '\n')
        nbr_train_samples += 1

    for img in val_images:
        source = os.path.join(file_total, car, img)
        if img.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG')):
            f_val.writelines(source + " " + car + '\n')
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))

f_train.close()
f_val.close()
f.close()

