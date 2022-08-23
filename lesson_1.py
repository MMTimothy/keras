from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense
import numpy


seed=7
numpy.random.seed(seed)
dataset = numpy.loadtxt("pima-indians-diabetes.csv",delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,init="uniform",activation="relu"))
model.add(Dense(10,init="uniform",activation="relu"))
model.add(Dense(8,init="uniform",activation="relu"))
#model.add(Dense(6,init="uniform",activation="relu"))
##model.add(Dense(4,init="uniform",activation="relu"))
#model.add(Dense(2,init="uniform",activation="relu"))
model.add(Dense(1,init="uniform",activation="sigmoid"))

X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=3)

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch=4550,batch_size=50)

scores = model.evaluate(X_test,y_test)

print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))



forest = RandomForestClassifier(n_estimators=10,random_state=5)
forest.fit(X_train,y_train)

print (forest.score(X_test,y_test))
