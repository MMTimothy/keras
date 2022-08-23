from keras.models import Model,Sequential
from keras.layers import BatchNormalization,AveragePooling2D,MaxPooling2D,Conv2D,Activation,Dropout,Flatten,Input,Dense,concatenate

def shallownet_sequential(width,height,depth,classes):
	model = Sequential()
	inputShape = (height,width,depth)
	
	model.add(Conv2D(148,(3,3),padding="same",input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(Flatten())
	model.add(Dense(classes))
	model.add(Activation("softmax"))

	return

