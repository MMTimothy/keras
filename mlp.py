import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD

import numpy as np

X_train = np.random.random((1000,20))
Y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
X_test = np.random.random((1000,20))
Y_test = keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
print(X_train.shape)
print(len(X_train[0]))
print (X_train)

model = Sequential()

model.add(Dense(64,activation="relu",input_dim=len(X_train[0])))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=20,batch_size=128)
score = model.evaluate(X_test,Y_test,batch_size=128)
print(score)
