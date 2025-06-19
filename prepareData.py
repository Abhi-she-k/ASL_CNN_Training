import os
from os import listdir, rename
from os.path import isfile, join

import datetime
import random

import keras

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=6,       # Random rotation (degrees)
    height_shift_range=0.05,  # Vertical shift
    zoom_range=0.05,          # Zoom in/out
    horizontal_flip=True,    # Flip horizontally
)
datagen = ImageDataGenerator(
    rotation_range=2,             
    height_shift_range=0.03,
    width_shift_range=0.03,       
    zoom_range=0.1,              
    brightness_range=[0.8, 1.2],  
    horizontal_flip=True         
)

def process_directory_content(directory_path):

    print(f"Processing directory: {directory_path}")

    
    print("Creating training and test directories...")
    new_directory = join("D:/asl_data", "newData" + str(datetime.date.today()))
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    print("Training directory created")
    training_directory = join(new_directory ,"training")
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)

    print("Test directory created")
    test_directory = join(new_directory ,"test")
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)


    for letter in listdir(directory_path):
        
        print(f"Processing letter: {letter}")

        letter_path = join(directory_path, letter)
        
        training_directory_letter = join(training_directory, letter)
        if not os.path.exists(training_directory_letter):
            os.makedirs(training_directory_letter)
        
        test_directory_letter = join(new_directory, "test/" + letter)
        if not os.path.exists(test_directory_letter):
            os.makedirs(test_directory_letter)


        for i in range(len(listdir(letter_path))):

            image_path = join(letter_path, listdir(letter_path)[i])
            img_name = "ASL_"+letter+"_"+str(i)

            img = load_img(image_path)
            augmented_images = augment_image(img)

            img.save(join(training_directory_letter, img_name + ".jpg"))

            save_aug_imgs(augmented_images, training_directory_letter, img_name)

        test_directory_letter = join(new_directory, "test/" + letter)
        print(f"Splitting data for letter '{letter}' into test directory: {test_directory_letter}")
        test_split(training_directory_letter, test_directory_letter, test_ratio=0.2)


def augment_image(img):

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    i = 0

    for batch in datagen.flow(x, batch_size=1):
        i += 1
        if i > 3:
            break
        img_augmented = array_to_img(batch[0]) 
        yield img_augmented


def save_aug_imgs(augmented_images, new_directory_path, img_name):
    
    for i, img in enumerate(augmented_images):

        img.save(join(new_directory_path, f"{img_name}_aug_{i}.jpg"))


def test_split(training_directory_path, new_directory_path, test_ratio=0.2):

    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)    

    letter_imgs = []

    for file in listdir(training_directory_path):

        file_path = join(training_directory_path, file)

        letter_imgs.append(join(file_path))

    train_size = int(len(letter_imgs) * test_ratio)
    test_split_files = random.sample(letter_imgs, train_size)
    
    for i in range(len(test_split_files)):


        img = load_img(test_split_files[i])
        img.save(join(new_directory_path, f"{i}.jpg"))

        os.remove(join(training_directory_path,test_split_files[i]))



path = "D:/asl_data/Letters"

process_directory_content(path)










