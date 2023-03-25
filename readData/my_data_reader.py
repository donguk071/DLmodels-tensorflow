import glob
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split


class DataReader:
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        self.f_data_reader()

    def f_data_reader(self, rootfolder=" ", img_size = 150):

        file_path_P = glob.glob('./data/RSP_data/paper/*')
        file_path_R = glob.glob('./data/RSP_data/rock/*')
        file_path_S = glob.glob('./data/RSP_data/scissors/*')

        file_path = file_path_S + file_path_R + file_path_P

        print("size of data set : \n\n" + str(len(file_path)))

        data = []
        for i, path in enumerate(file_path):  # label data set
            img = Image.open(path)
            img = np.asarray(img)
            if i < len(file_path_P):
                img_label = 0
            if i > len(file_path_R) and i < len(file_path_S)*2:
                img_label = 1
            if i > len(file_path_P)*2 and i < len(file_path_P)*3:
                img_label = 2
            if i % 1000 == 0:
                print("labeling data set : " + str(i) + "/" + str(len(file_path)))
            data.append((img, img_label))
        print("labeling data done : " + str(len(file_path)) + "/" + str(len(file_path)))

        # optional cause we are using train_test_split func
        random.shuffle(data)

        target = [row[1] for row in data]
        data = [row[0] for row in data]

        # train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, target, test_size=0.2, shuffle=True, stratify=target, random_state=34)
        self.x_train = np.array(self.x_train)/255.0
        self.x_test = np.array(self.x_test)/255.0
        self.y_train = np.array(self.y_train) / 1.0
        self.y_test = np.array(self.y_test) / 1.0

        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.x_train.shape))
        print("Training Y Size : " + str(self.y_train.shape))
        print("Test X Size : " + str(self.x_test.shape))
        print("Test Y Size : " + str(self.y_test.shape) + '\n\n')
