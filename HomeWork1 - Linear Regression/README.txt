PREDICT WIND SPEED AT MAL WIND STATION

The ‘wind’ data consists of 14 features (3 date features & 11 wind speed features) and the target – wind at MAL Station. The goal is to predict the wind speed in MAL Station with the best 3 features, from amongst the 14 available using a linear model. 
In order to do so you I have implemented the Linear Regression Gradient Descent algorithm.
First, I run the model with all the features in order to find the best learning rate alpha and calculate the Training & Test error after learning with all the features.
Then, I use the alpha found in the previous step in order to build Linear Regression models each time with different 3 features. For each combination of 3 features the training error is calculated. The best 3 features are the ones that give the lowest training error. They represent the best choice of 3 out of the 14 features, based on the training data.
For these features, and the associated coefficients, the test error is calculated.
