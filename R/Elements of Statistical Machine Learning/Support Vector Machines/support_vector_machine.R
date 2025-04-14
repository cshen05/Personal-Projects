#### This file is for an illustration of support vector machine for two classes

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(e1071) #This package is for support vector machines, maximum margin classifier, support vector classifier

#### We create training data that are linearly separable

## We first generate the data
set.seed(1000)
N = 1000
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/4), rep(1,N/4))
X[1:N/8,] = X[1:N/8,] + 5
X[(N/8 + 1):N/4,] = X[(N/8 + 1):N/4,] - 2.5
experiment_data <- data.frame(X=X, Y=as.factor(Y))

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = Y)) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

## We create training and test sets
set.seed(100)
sample_size <- floor(0.75 * nrow(experiment_data))

train_index <- sample(seq_len(nrow(experiment_data)), size = sample_size)
experiment_data_train <- experiment_data[train_index,]
experiment_data_test <- experiment_data[-train_index,]

## We now use support vector machine with radial kernel to create non-linear boundary
polynomial_support_machine <- svm(Y~., data = experiment_data_train, kernel = "polynomial", degree = 2, cost = 0.5, scale = FALSE)
#plot(polynomial_support_machine, experiment_data, color.palette = terrain.colors)
my_support_machine <- svm(Y~., data = experiment_data_train, kernel = "radial", gamma = 1 , cost = 0.5, scale = FALSE) 
# cost here plays the kind of role of 1/ C, namely, when cost is small, the margin is wide and when cost is large, the margin is narrow
plot(my_support_machine, experiment_data, color.palette = terrain.colors)
plot(polynomial_support_machine, experiment_data, color.palette = terrain.colors)

# Classification accuracy
Y_pred <- predict(my_support_machine, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_test <- sum(diag(table_test))/ sum(table_test)
accuracy_test

########################################### We now use find the best gamma and cost via cross-validation
gamma_val = seq(1, 10, length = 10) # We consider 10 values of gamma from 1 to 10
cost_val = seq(0.5, 10, length = 50) # We conside 50 values of cost from 0.5 to 10
tune_err <- tune(svm, Y~., data = experiment_data_train, kernel = "radial", ranges = list(cost = cost_val , gamma = gamma_val))
# tune is used to determine the best values of gamma and cost based on cross-validation
summary(tune_err)

## Classification accuracy from best model

best_model = tune_err$best.model

## ## Classification accuracy on test set with best model
Y_pred <- predict(best_model, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)

best_accuracy_test <- sum(diag(table_test))/ sum(table_test)
best_accuracy_test

############# Performance of tree-based methods
library(tree)
library(randomForest)
####### Single classification tree
my_tree <- tree(Y~., experiment_data_train)
summary(my_tree)

## Classification accuracy in the test set
Y_pred <- predict(my_tree, newdata = experiment_data_test, type = "class")
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_tree_test <- sum(diag(table_test))/ sum(table_test)
accuracy_tree_test

######## Bagging
my_bag <- randomForest(Y~., experiment_data_train, mtry = 2, importance = TRUE) 

## Classification accuracy in the test set from bagging
Y_pred <- predict(my_bag, newdata = experiment_data_test, type = "class")
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_bag_test <- sum(diag(table_test))/ sum(table_test)
accuracy_bag_test

######## Random forest
my_forest <- randomForest(Y~., experiment_data_train, mtry = 1, importance = TRUE) 

## Classification accuracy in the test set from bagging
Y_pred <- predict(my_forest, newdata = experiment_data_test, type = "class")
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_forest_test <- sum(diag(table_test))/ sum(table_test)
accuracy_forest_test
