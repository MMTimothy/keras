import matplotlib
matplotlib.use("Agg")


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from library.animal_plant_detectionnet import AnimalPlantDetectionNet
from library import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys

def load_dataset(datasetPath):
	#print(datasetPath)
	imagePaths = datasetPath
	data = []
	
	for imagePath in imagePaths:
		
		image = cv2.imread(imagePath)
		try:		
			image = cv2.resize(image,(128,128))
	
			data.append(image)
		except:
			pass
	
	return np.array(data,dtype="float32")

print ("[INFO] loading data...")
kalesData = load_dataset(config.kales)
spinachData = load_dataset(config.spinach)



kalesLabels = np.zeros((kalesData.shape[0],))
spinachLabels = np.ones((spinachData.shape[0],))



data = np.vstack([kalesData,spinachData])
labels = np.hstack([kalesLabels,spinachLabels])
data /= 255

labels = to_categorical(labels,num_classes=2)
classTotals = labels.sum(axis=0)

classWeight = classTotals.max() / classTotals

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=config.TEST_SPLIT,random_state=3)

aug = ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

print("[INFO] compiling model....")
opt = SGD(config.INIT_LR ,momentum=0.9,decay=config.INIT_LR/config.NUM_EPOCHS)
model = AnimalPlantDetectionNet.build(width=128,height=128,depth=3,classes=2)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training network......")

H = model.fit_generator(aug.flow(trainX,trainY,batch_size=23),validation_data=(testX,testY),steps_per_epoch=23,epochs=50,class_weight=classWeight,verbose=1)

print ("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=config.CLASSES))

print("[INFO] serializing network to '{}'.....".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

N = np.arange(0,config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N,H.history["loss"],label="train_loss")
plt.plot(N,H.history["val_loss"],label="val_loss")
plt.plot(N,H.history["accuracy"],label="train_acc")
plt.plot(N,H.history["val_accuracy"],label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lowe left")
plt.savefig(config.TRAINING_PLOT_PATH)






