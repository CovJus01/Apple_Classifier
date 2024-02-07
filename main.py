import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as utils

#Get Dataset
X,Y = utils.importData("./apple_quality.csv")

#Normalize the features
X, means, deviations = utils.zScoreNormilization(X)

#Model Design
model = tf.keras.Sequential([
  tf.keras.Input(shape=(7,)),
  tf.keras.layers.Dense(units=25, activation='relu'),
  tf.keras.layers.Dense(units=25, activation='relu'),
  tf.keras.layers.Dense(units=25, activation='relu'),
  tf.keras.layers.Dense(units =1, activation='sigmoid')
], name = 'apple_model')

model.summary()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
  X[:3500],
  Y[:3500],
  epochs=50
)



predict = model.predict(X[3500:4000].reshape((-1,7)))
predict = (predict >= 0.5).astype(int)
count = 0
for i in range(0,500):
    if(int(predict[i, 0]) == int(Y[i+3500])):
        count += 1
    print(f"Prediction: {predict[i, 0]} Actual: {int(Y[i+3500])}")

print(f"Accuracy: {count}/500 {float(count)/5.0}%")