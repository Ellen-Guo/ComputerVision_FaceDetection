# Computer Vision | Final Project
import cv2
import numpy as np
import random


# import all profile images from FaceDetection Folder
def read_img(path):
    image = []
    for i in range(40):
        folder = '/s' + str(i+1)
        directory = path + folder
        for j in range(10):
            image.append(cv2.imread(directory + '/' + str(j+1) + '.pgm', 0))
    return np.array(image)


# function that splits the data into test, training, and validation sets
def split_data(data, index, n):
    new_index = []
    for j in index:
        for i in np.arange(10):
            new_index.append(j*10 + i)
    n *= 10
    data_split = (data[new_index[0:n]], data[new_index[n:]])
    return data_split


def match_fun():
    pass


def sift_feature(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    # matcher
    return des1, des2


def main():
    images = read_img('FaceDetection')
    index = list(np.arange(40))
    random.Random(64).shuffle(index)
    N = len(index)
    train, test = split_data(images, index, int(N * 2/3))
    random.Random(128).shuffle(train) # shuffle again
    rand_train = np.random.choice(len(train), size=10, replace=False)
    d1_arr=[]
    d2_arr=[]
    # val = random.randint(0,259)
    for j in rand_train:
        for i in np.arange(40)*10:
            d1,d2 = sift_feature(train[j],images[i])
        d1_arr.append(d1)
        d2_arr.append(d2)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

