import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from library.learningratefinder import LearningRateFinder
from library.firedetectionnet import FireDetectionNet
from library import config
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import cv2
import sys


def load_dataset(datasetPath):
	imagePaths = list(paths.list_images(datasetPath))
	data = []

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image,(128,128))
		data.append(image)

	return np.array(data,dtype="float32")

ap = argparse.ArgumentParser()
ap.add_argument("-f","--lr-find",type=int,default=0,help="whether or not to find the optimal learning rate")
args = vars(ap.parse_args())

print("[INFO] loading data...")

fireData = load_dataset(config.FIRE_PATH)
nonFireData = load_dataset(config.NON_FIRE_PATH)

fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

data = np.vstack([fireData,nonFireData])
labels = np.hstack([fireLabels,nonFireLabels])

data /= 255


labels = to_categorical(labels,num_classes=2)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=config.TEST_SPLIT,random_state=43)

aug = ImageDataGenerator(rotation_range=30,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
print("[INFO] compilling model...")

opt = SGD(lr=config.INIT_LR,momentum=0.9,decay=config.INIT_LR / config.NUM_EPOCHS)

model = FireDetectionNet.build(width=128,height=128,depth=3,classes=2)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

if args["lr_find"] > 0:
	print("[INFO] finding learning rate....")
	lrf = LearningRateFinder(model)
	lrf.find(aug.flow(trainX,trainY,batch_size=config.BATCH_SIZE),1e-10,1e-1,stepsPerEpoch=np.ceil((trainX.shape[0]/ float(config.BATCH_SIZE))),epochs=20,batchSize=config.BATCH_SIZE,classWeight=classWeight)

	lrf.plot_loss()
	plt.savefig(config.LRFIND_PLOT_PATH)
	
	print("[INFO] learning rate finder complete")
	print("[INFO] examine plot and adjust learning rates before training")
	sys.exit(0)

print ("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX,trainY,batch_size=config.BATCH_SIZE),validation_data=(testX,testY),steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE,epochs=config.NUM_EPOCHS,class_weight=classWeight,verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=config.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=config.CLASSES))

print ("[INFO] sericalizing network to '{}'.....".format(config.MODEL_PATH))
model.save(config.MODEL_PATH)

N = np.arange(0,config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N,H.history["loss"],label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

		
