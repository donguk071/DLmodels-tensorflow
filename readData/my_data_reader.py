import glob
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split

class DataReader:
    def __init__(self, rootfolder = " "):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        #self.f_data_reader(rootfolder)
    def f_data_reader(self, rootfolder, img_size=150, one_hot_encoding=False):

        print(str(rootfolder) + "/*")
        folder_path = glob.glob(str(rootfolder) + "/*")
        print("Read folder path : " + str(folder_path))

        numFiles = [0] * len(folder_path)  # numFiles 리스트 초기화
        file_path = []
        for i, path in enumerate(folder_path):
            temp_path = glob.glob(path + "/*")  # 폴더 내 모든 파일 경로 가져오기
            numFiles[i] = len(temp_path)
            file_path += temp_path

        print("size of data set : " + str(len(file_path)))
        print("size of each label : " + str(numFiles))

        data = []
        for i, path in enumerate(file_path):
            img = Image.open(path)
            if i == 0 : print("img resize " + str(img.size[1]) + " to " + str(img_size))
            img = img.resize((img_size, img_size))
            img = np.asarray(img)

            for j, folder in enumerate(folder_path):
                if i < sum(numFiles[:j + 1]):  # 파일이 속한 폴더의 인덱스 계산
                    img_label = j
                    break

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
        self.x_train = np.array(self.x_train) / 255.0
        self.x_test = np.array(self.x_test) / 255.0
        self.y_train = np.array(self.y_train) / 1.0
        self.y_test = np.array(self.y_test) / 1.0

        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.x_train.shape))
        print("Training Y Size : " + str(self.y_train.shape))
        print("Test X Size : " + str(self.x_test.shape))
        print("Test Y Size : " + str(self.y_test.shape) + '\n\n')
    
    
    def csv_data_reader(file_path = " ", one_hot_encoding=False):
        
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.x_train.shape))
        print("Training Y Size : " + str(self.y_train.shape))
        print("Test X Size : " + str(self.x_test.shape))
        print("Test Y Size : " + str(self.y_test.shape) + '\n\n')
        
    
        