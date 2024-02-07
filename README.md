# Apple_Classifier
Machine learns to classify apples as good or bad quality

# Overview
This model uses tensorflow and some apple data to predict the quality based on some parameters:
- Size
- Weight
- Sweetness
- Crunchiness
- Juiciness
- Ripeness
- Acidity
These Values are all regularized using a z-score normalization. They are then passed into a neural net that consists of 4 dense layers:
Layer 1-3: 25 units with ReLU activation
Layer 4: 1 unit with sigmoid activation
This gives us a result which predicts the quality of the apple

#Result
With the dataset of 4000, 500 of the values were used to verify the models capability. With 100 Epochs the loss stabalized around 0.02
and the accuracy varied around 94-96%. The results are quite good and show the success of the learning model.
