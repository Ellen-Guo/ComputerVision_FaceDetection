# Computer Vision | Final Project
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


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


def check(predict, target):
    error = 0
    for k in range(len(target)):
        if target[k] != predict[k]:
            error += 1
    error = error / len(target) * 100
    return error


def display_img(img, query, predict, title):
    fig, axs = plt.subplots(len(query), 2)
    fig.suptitle(title)
    axs[0, 0].set_title('Queried Image')
    axs[0, 1].set_title('Predicted Class')
    for i in range(len(query)):
        axs[i, 0].imshow(img[query[i]], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 1].imshow(img[predict[i]*10], cmap='gray')
        axs[i, 1].axis('off')


def main():
    images = read_img('FaceDetection')
    index = list(np.arange(40))
    random.Random(64).shuffle(index)
    N = len(index)
    split = int(N * 2/3)
    train, test = split_data(images, index, split)
    train_index = index[0:split]
    test_index = index[split:]
    random.Random(128).shuffle(train_index) # shuffle again
    # randomly select the images to be queried
    sel_index = np.arange(len(train))
    count = 0
    error_arr = []
    all_train = []
    all_predicted = []
    # determine optimal database image value
    while count < 8:
        # boolean array to remove "database" images from query set
        not_first_arr = (sel_index % 10) > count
        # randomly select query images excluding images used in the database
        rand_train = np.random.choice(sel_index[not_first_arr], size=10, replace=False)
        target = []
        predict = []
        train_query = []
        for j in rand_train:
            train_id = int(j / 10)
            img_id = j % 10
            query_id = train_index[train_id] * 10 + img_id
            target.append(train_index[train_id])
            train_query.append(query_id)
            match_per = []
            # iterating through the first images of the database only
            for i in np.arange(40)*10:
                # testing if given a database of multiple images for this person; can SIFT be used to identify them
                fold_match = []
                for k in range(count + 1):
                    fold = k+i
                    des1, des2 = sift_feature(images[query_id],images[fold])
                    match_rate = match_fun(des1,des2)
                    fold_match.append(match_rate)
                match_per.append(np.average(fold_match))
            predict.append(match_per.index(max(match_per)))
        all_train.append(train_query)
        all_predicted.append(predict)
        curr_error = check(predict, target)
        count += 1
        print('Error for %s Image in Database: %s' % (count, curr_error))
        error_arr.append(curr_error)

    # Testing
    random.Random(64).shuffle(test_index) # shuffle again
    # randomly select the images to be queried
    sel_test = np.arange(len(test))
    # determine optimal database image count
    val = error_arr.index(min(error_arr)) + 1
    print('Best Image Number in Database Value: %s' %val)
    remove = (sel_test % 10) > val
    # randomly select query images excluding images used in the database
    rand_test = np.random.choice(sel_test[remove], size=10, replace=False)
    target = []
    predict = []
    test_query = []
    for j in rand_test:
        test_id = int(j / 10)
        img_id = j % 10
        query_id = train_index[test_id] * 10 + img_id
        target.append(train_index[test_id])
        test_query.append(query_id)
        match_per = []
        # iterating through the first images of the database only
        for i in np.arange(40)*10:
            # testing if given a database of multiple images for this person; can SIFT be used to identify them
            fold_match = []
            for k in range(count + 1):
                fold = k+i
                des1, des2 = sift_feature(images[query_id],images[fold])
                match_rate = match_fun(des1,des2)
                fold_match.append(match_rate)
            match_per.append(np.average(fold_match))
        predict.append(match_per.index(max(match_per)))
    test_error = check(predict, target)
    print('Test Error: ', test_error)

    for i in range(len(all_train)):
        display_img(images, all_train[i], all_predicted[i], '%s Image in Database' %(i+1))

    display_img(images, test_query, predict, 'Test Set Results')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
