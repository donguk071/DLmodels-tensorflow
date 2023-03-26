import sys
sys.path.append(
    'C:\\Users\\drago\\university\\23.1 semester\\DL\\DLmodels-tensorflow')
print(sys.path)

from tensorflow import keras
from  tensorflow.keras.optimizers import Adam 
from readData import my_data_reader

dr = my_data_reader.DataReader()
dr.f_data_reader('./data/RSP_data')


model = keras.Sequential([
    # 이걸 수정해보자
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), input_shape = (150,150,3), activation ='ReLU' ),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation ='ReLU' ),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = 'ReLU'),
    keras.layers.Dense(4096, activation = 'ReLU'),
    keras.layers.Dense(1000, activation = 'ReLU'),
    #classifing 3
    keras.layers.Dense(3, activation = 'softmax')
])

model.compile(optimizer = Adam(learning_rate= 0.001), metrics=['accuracy'],loss = 'sparse_categorical_crossentropy')
print("model definded\n\n", model.summary)

print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
early_stop2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
print(dr.y_train.dtype)
history = model.fit(dr.x_train, dr.y_train, epochs=2,
                    validation_data=(dr.x_test, dr.y_test),
                    callbacks=[early_stop, early_stop2])

res = model.evaluate(dr.x_test,dr.y_test,verbose = 0 )
print("acc is : ", res[1]* 100)

