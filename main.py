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
    optimizer=tf.keras.optimizers.Adam(0.003),
)

model.fit(
  X[:3500],
  Y[:3500],
  epochs=100
)


#Predict and check prediction accuracy
predict = model.predict(X[3500:4000].reshape((-1,7)))
predict = (predict >= 0.5).astype(int)
correct, incorrect, accuracy = utils.checkAccuracy(predict, Y[3500:4000])
print(f"Accuracy: {correct}/500 {accuracy}%")