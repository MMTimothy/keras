from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
import numpy as np
import keras

np.random.seed(0)

main_input = Input(shape=(100,),dtype="int32",name="main_input")

x = Embedding(output_dim=512,input_dim=10000,input_length=100)(main_input)

lstm_out=LSTM(32)(x)

auxiliary_output = Dense(1,activation="sigmoid",name="aux_output")(lstm_out)

auxiliary_input= Input(shape=(5,),name="aux_input")
x = keras.layers.concatenate([lstm_out,auxiliary_input])

x = Dense(64,activation="relu")(x)
x = Dense(64,activation="relu")(x)
x = Dense(64,activation="relu")(x)

main_output = Dense(1,activation="sigmoid",name="main_output")(x)
model = Model(inputs=[main_input,auxiliary_input],outputs=[main_output,auxiliary_output])

headline_data = np.round(np.abs(np.random.rand(12,100)*100))
additional_data = np.random.randn(12,5)
headline_labels = np.random.randn(12,1)
additional_labels = np.random.randn(12,1)
model.compile(optimizer="rmsprop",loss={"main_output":"binary_crossentropy","aux_output":"binary_crossentropy"},loss_weights={"main_output":1,"aux_output":0.2},metrics=["accuracy"])
model.fit([headline_data,additional_data],[headline_labels,additional_labels],epochs=50,batch_size=32)
