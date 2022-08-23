from keras.models import load_model
from library import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os
import argparse


print("[INFO] loading model...")
app = argparse.ArgumentParser()
app.add_argument("--model","-m",type=str)
app.add_argument("--image","-i",type=str,help="please provide value for --image 'image path' --model 'model path'")

args = app.parse_args()
print (type(args.image))
print (args.image)
print (args.model)
images = args.image
ml = args.model

model = load_model(ml)

print ("[INFO] predicting ...")
img = cv2.imread(images)
output = img.copy()
img = cv2.resize(img,(128,128))
img = img.astype("float32")/255.0
image = img
preds = model.predict(np.expand_dims(image,axis=0))[0]

j = np.argmax(preds)

label = config.CLASSES[j]

i = 0
text = label if label == "Non-Fire" else "Warning! Fire"
print(text)
output = imutils.resize(output,width=500)
cv2.putText(output,text,(35,50),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,250,0),5)
filename = "{}.png".format(images)
p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
cv2.imwrite(p,output)






