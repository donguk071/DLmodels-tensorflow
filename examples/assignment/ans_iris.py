import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from tensorflow import keras


import random
from matplotlib import pyplot as plt
import numpy as np


label_orig = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

print("Reading Data...")
file = open("C:/Users/drago/university/23.1 semester/DL/DLmodels-tensorflow/data/iris.csv")
data = []
for line in file:
    splt = line.split(",") #split() 함수 :  문자열 내부에 있는 공백 또는 특별한 문자를 구분해서, 리스트 아이템으로 만듦
    if len(splt) != 5:
      break
    feature_1 = float(splt[0].strip()) #strip() 함수 :  문자열 앞뒤의 공백 또는 특별한 문자 삭제
    feature_2 = float(splt[1].strip())
    feature_3 = float(splt[2].strip())
    feature_4 = float(splt[3].strip())
    label = label_orig.index(splt[4].strip())
    data.append(((feature_1, feature_2, feature_3, feature_4), label))

random.shuffle(data)

X = []
Y = []

# i=0 체크를 위한 코드               
for el in data:
    X.append(el[0])
    Y.append(el[1])
    # print(f"index: {i}, element: {el}")
    # i=i+1

X = np.asarray(X)
Y = np.asarray(Y)

X = X / np.max(X, axis=0) #각열의 최대값으로 나눠줌

train_X = X[:int(len(X)*0.8)]
train_Y = Y[:int(len(Y)*0.8)]
test_X = X[int(len(X)*0.8):]
test_Y = Y[int(len(Y)*0.8):]

print("\n\nData Read Done!")
print("Training X Size : " + str(train_X.shape))
print("Training Y Size : " + str(train_Y.shape))
print("Test X Size : " + str(test_X.shape))
print("Test Y Size : " + str(test_Y.shape) + '\n\n')


EPOCHS = 10

model = keras.Sequential([
    keras.layers.Dense(4),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(3, activation='softmax')
])


model.compile(optimizer="adam", metrics=["accuracy"],loss="sparse_categorical_crossentropy")


print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_X, train_Y, epochs=EPOCHS, batch_size=5,
                    validation_data=(test_X, test_Y),
                    callbacks=[early_stop])

train_history = history.history["loss"]
validation_history = history.history["val_loss"]
fig = plt.figure(figsize=(8, 8))
plt.title("Loss History")
plt.xlabel("EPOCH")
plt.ylabel("LOSS Function")
plt.plot(train_history, "red")
plt.plot(validation_history, 'blue')
fig.savefig("train_history.png")

train_history = history.history["accuracy"]
validation_history = history.history["val_accuracy"]
fig = plt.figure(figsize=(8, 8))
plt.title("Accuracy History")
plt.xlabel("EPOCH")
plt.ylabel("Accuracy")
plt.plot(train_history, "red")
plt.plot(validation_history, 'blue')
fig.savefig("accuracy_history.png")
