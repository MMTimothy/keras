from keras.models import Sequential
from keras.layers import BatchNormalization,SeparableConv2D,MaxPooling2D,Activation,Flatten,Dropout,Dense

class FireDetectionNet:
	@staticmethod
	def build(width,height,depth,classes):	
		model = Sequential()
		inputShape = (height,width,depth)
		chanDim = -1
		

		model.add(SeparableConv2D(16,(7,7),padding="same",input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(SeparableConv2D(32,(3,3),padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(SeparableConv2D(64,(3,3),padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(SeparableConv2D(128,(3,3),padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))


		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model

