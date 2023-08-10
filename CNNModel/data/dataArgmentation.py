import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps    

#다음 변수를 수정하여 새로 만들 이미지 갯수를 정합니다.

file_path = 'C:\\Users\\leeja\\Desktop\\CNNModel\\data\\image\\test\\word\\22\\'
file_names = os.listdir(file_path)
print(file_names)
total_origin_image_num = len(file_names)
num_augmented_images = len(file_names)
print(total_origin_image_num)
augment_cnt = 1

for i in range(0, num_augmented_images):
    change_picture_index = i
    print(change_picture_index)
    print(file_names[change_picture_index])
    file_name = file_names[change_picture_index]
    
    origin_image_path = file_path + file_name
    print(origin_image_path)
    image = Image.open(origin_image_path)
    

    #이미지 기울이기
    rotated_image = image.rotate(-5)
    rotated_image.save(file_path + 'inverted_' + str(augment_cnt) + '.jpg')
    augment_cnt += 1
    rotated_image = image.rotate(-10)
    rotated_image.save(file_path + 'inverted_' + str(augment_cnt) + '.jpg')
    augment_cnt += 1
    rotated_image = image.rotate(5)
    rotated_image.save(file_path + 'inverted_' + str(augment_cnt) + '.jpg')
    augment_cnt += 1
    rotated_image = image.rotate(10)
    rotated_image.save(file_path + 'inverted_' + str(augment_cnt) + '.jpg')
    augment_cnt += 1
        
    #노이즈 추가하기
    img = cv2.imread(origin_image_path)
    row,col,ch= img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy_array = img + gauss
    noisy_image = Image.fromarray(np.uint8(noisy_array)).convert('RGB')
    noisy_image.save(file_path + 'noiseAdded_' + str(augment_cnt) + '.jpg')
    augment_cnt += 1
    