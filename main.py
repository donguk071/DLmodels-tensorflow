from tensorflow import keras
import data.data_reader as data_reader

EPOCHS = 1


dr = data_reader.DataReader()

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam', metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')


print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
early_stop2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
print(dr.train_Y.dtype)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop, early_stop2])

# monitor : 학습 조기종료를 위해 관찰하는 항목입니다. val_loss 나 val_accuracy 가 주로 사용됩니다. (default : val_loss)
# min_delta : 개선되고 있다고 판단하기 위한 최소 변화량을 나타냅니다. 만약 변화량이 min_delta 보다 적은 경우에는 개선이 없다고 판단합니다. (default = 0)
# patience : 개선이 안된다고 바로 종료시키지 않고, 개선을 위해 몇번의 에포크를 기다릴지 설정합니다. (default = 0)
# mode : 관찰항목에 대해 개선이 없다고 판단하기 위한 기준을 설정합니다. monitor에서 설정한 항목이 val_loss 이면 값이 감소되지 않을 때 종료하여야 하므로 min 을 설정하고, val_accuracy 의 경우에는 max를 설정해야 합니다. (default = auto)
# auto : monitor에 설정된 이름에 따라 자동으로 지정합니다.
# min : 관찰값이 감소하는 것을 멈출 때, 학습을 종료합니다.
# max: 관찰값이 증가하는 것을 멈출 때, 학습을 종료합니다.

data_reader.draw_graph(history)
