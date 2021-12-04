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


def match_fun(des1, des2):
    r1,c1 = des1.shape #queried
    r2,c2 = des2.shape #database image
    rate = 0
    d_all =[]
    min_darr = []
    match = []
    for i in range(r1):
        d = []
        for j in range(r2):
            distance = np.linalg.norm(des1[i]-des2[j])
            d.append(distance)
        d_all.append(d)
        min_d = min(d)
        min_darr.append(min_d)
    if (r1<r2):
        check = list(np.arange(r2))
        for i in range(r1):
            smallest_d = min(min_darr)
            threshold = np.mean(np.array(min_darr))
            if smallest_d<threshold:
                smallest_ind = min_darr.index(smallest_d)
                des_sel = d_all[smallest_ind].index(smallest_d)
                if des_sel in check:
                    match.append([smallest_ind, des_sel])
                    check.remove(des_sel)
                min_darr[smallest_ind] = np.inf
        len_match = len(match)
        rate = (len_match / r1) * 100

    if (r2<r1):
        check = list(np.arange(r1))
        for i in range(r2):
            smallest_d = min(min_darr)
            threshold = np.mean(np.array(min_darr))
            if smallest_d < threshold:
                smallest_ind = min_darr.index(smallest_d)
                des_sel = d_all[smallest_ind].index(smallest_d)
                if des_sel in check:
                    match.append([smallest_ind, des_sel])
                    check.remove(des_sel)
                min_darr[smallest_ind] = np.inf
        len_match = len(match)
        rate = (len_match / r2) * 100

    if (r1 == r2):
        if not np.count_nonzero(min_darr):
            rate = 100
        else:
            check = list(np.arange(r1))
            for i in range(r2):
                smallest_d = min(min_darr)
                threshold = np.mean(np.array(min_darr))
                if smallest_d < threshold:
                    smallest_ind = min_darr.index(smallest_d)
                    des_sel = d_all[smallest_ind].index(smallest_d)
                    if des_sel in check:
                        match.append([smallest_ind, des_sel])
                        check.remove(des_sel)
                    min_darr[smallest_ind] = np.inf
            len_match = len(match)
            rate = (len_match / r2) * 100
    return rate


def sift_feature(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    return des1, des2


def main():
    images = read_img('FaceDetection')
    index = list(np.arange(40))
    random.Random(64).shuffle(index)
    N = len(index)
    train, test = split_data(images, index, int(N * 2/3))
    random.Random(128).shuffle(train) # shuffle again
    np.random.seed(90)
    rand_train = np.random.choice(len(train), size=10, replace=False)
    # val = random.randint(0,259)
    for j in rand_train:
        fold_match = []
        for i in np.arange(40)*10:
            d1,d2 = sift_feature(images[j],images[i])
            match_rate = match_fun(d1,d2)
            max_match = []
            if (match_rate>55):
                for k in range(10):
                    fold = k+i
                    des1, des2 = sift_feature(images[j],images[fold])
                    match_rate = match_fun(des1,des2)
                    max_match.append(match_rate)
                maximum = max(max_match)
                ind_max = max_match.index(maximum)
                fold_match.append([ind_max+i,maximum])
        for i in range(len(fold_match)):
            if (fold_match[i][1] > 90):
                print('I love Anaya')
                print(fold_match[i][0])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
