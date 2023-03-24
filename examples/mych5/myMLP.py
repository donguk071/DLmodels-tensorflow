import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# minst 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)  # tensor 모양 변환 (28x28)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
# not essential but vector로 보는것이 각각의 확률을 뽑아내기 때문에 one hot encoding 하는것이 적합하다
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# model
mlp_model = Sequential([
    Dense(units=1024, activation='tanh', input_shape=(784,),
          kernel_initializer='random_uniform', bias_initializer='zeros'),
    Dense(units=10, activation='tanh', input_shape=(1024,),
          kernel_initializer='random_uniform', bias_initializer='zeros')
])

mlp_model.compile(optimizer=Adam(learning_rate=0.001), metrics=['accuracy'],
                  loss='mean_squared_error')  # 적당한 learning rate을 찾는법은?
mlp_model.summary()
history = mlp_model.fit(x_train, y_train, batch_size=128, epochs=11,
                        validation_data=(x_test, y_test), verbose=1)
res = mlp_model.evaluate(x_test, y_test, verbose=0)
print("acc is ", res[1]*100)
