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


def main():
    images = read_img('FaceDetection')
    index = np.arange(400)
    random.Random(32).shuffle(index)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

