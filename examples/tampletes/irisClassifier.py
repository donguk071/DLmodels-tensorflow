import numpy as np
from sklearn.model_selection import train_test_split
import csv
import random
import sys
from tensorflow import keras
from  tensorflow.keras.optimizers import Adam 
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#importing data
data = []
with open('C:/Users/drago/university/23.1 semester/DL/DLmodels-tensorflow/data/iris.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # label target data
        if len(row) != 0:
            if row [4] == "Iris-setosa":
                row [4] = 0
            if row [4] == "Iris-versicolor":
                row [4] = 1    
            if row [4] == "Iris-virginica":
                row [4] = 2
            int_list = list(map(float, row[0:4]))
            data.append((int_list, np.array(row[4])))    
               
random.shuffle(data) #optional
target = [row[1] for row in data]
data = [row[0] for row in data]


############ normalize min,max ###################
#( X- (X의 최솟값) ) / ( X의 최댓값 - X의 최솟값 )
'''
mmin = [12,12,12,12]
mmax = [0,0,0,0]
for i in range(4):
    for j in range(len(data)):
        if mmin[i] > data[j][i]:
            mmin[i] = data[j][i]
        if mmax[i] < data[j][i]:
            mmax[i] = data[j][i]

for i in range(4):
    for j in range(len(data)):
        data[j][i] = (data[j][i] - mmin[i]) / (mmax[i] - mmin[i])
'''
################# 한 줄로 대체 가능###################
'''
data = (data-np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
'''
################# 라이브러리로 대체 가능###############
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


#splite train validation data
x_train, x_test, y_train, y_test = train_test_split(
            data, target, test_size=0.2, shuffle=True, stratify=target, random_state=34)

x_train = np.array(x_train)
x_test = np.array(x_test) 
y_train = np.array(y_train) 
y_test = np.array(y_test) 


#shape of dataset 
print("\n\nData Read Done!")
print("Training X Size : " + str(x_train.shape))
print("Training Y Size : " + str(y_train.shape))
print("Test X Size : " + str(x_test.shape))
print("Test Y Size : " + str(y_test.shape) + '\n\n')


#define models
model = keras.Sequential([
    keras.layers.Dense(4),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy') 

print("*****************start training*************************\n\n", model.summary)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, epochs=50, batch_size=5,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop])

# res = model.evaluate(x_test,y_test,verbose = 0 )
# print("acc is : ", res[1]* 100)

# make results to plot
train_history = history.history["loss"]
validation_history = history.history["val_loss"]
fig = plt.figure(figsize=(8, 8))
plt.title("Loss History")
plt.xlabel("EPOCH")
plt.ylabel("LOSS Function")
plt.plot(train_history, "red")
plt.plot(validation_history, 'blue')
fig.savefig("./history/train_history.png")

train_history = history.history["accuracy"]
validation_history = history.history["val_accuracy"]
fig = plt.figure(figsize=(8, 8))
plt.title("Accuracy History")
plt.xlabel("EPOCH")
plt.ylabel("Accuracy")
plt.plot(train_history, "red")
plt.plot(validation_history, 'blue')
fig.savefig("./history/accuracy_history.png")