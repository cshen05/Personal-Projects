#### This file is for an illustration of maximal margin classifier

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(e1071) #This package is for support vector machines, maximum margin classifier, support vector classifier
library(tree)
####################################### Maximal margin classifier with separable data

################## We first generate the data
set.seed(100)
N = 1000
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/4), rep(1,N/4))
X[which(Y == 1),] <- X[which(Y==1),] + 4
experiment_data <- data.frame(X=X, Y=as.factor(Y))

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = Y)) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

#################### We create training and test sets
set.seed(100)
sample_size <- floor(0.75 * nrow(experiment_data))

train_index <- sample(seq_len(nrow(experiment_data)), size = sample_size)
experiment_data_train <- experiment_data[train_index,]
experiment_data_test <- experiment_data[-train_index,]

###################### We now use maximal margin classifier to create the hyperplane that separates the data
my_maximal <- svm(Y~., data = experiment_data_train, kernel = "linear", scale = FALSE)
# kernel = "linear" means we use maximal margin classifier
plot(my_maximal, experiment_data_train, color.palette = terrain.colors)

## Classification accuracy on training set
table(my_maximal$fitted, experiment_data_train$Y)

## Classification accuracy on test set
Y_pred <- predict(my_maximal, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_test <- sum(diag(table_test))/ sum(table_test)
accuracy_test

###################### Performance from tree method
library(tree) #This library is used for classification and regression trees

## Single classification tree
my_tree <- tree(Y~., experiment_data_train)
summary(my_tree)

plot(my_tree)
text(my_tree, pretty = 0)

## Classification accuracy on test set from single tree
experiment_pred <- predict(my_tree, newdata = experiment_data_test, type = "class")
table_experiment_test <- table(experiment_pred, experiment_data_test$Y)

accuracy_experiment_test <- sum(diag(table_experiment_test))/ sum(table_experiment_test)
accuracy_experiment_test

################################################ Maximal margin classifier with poorly separable data
set.seed(100)
N = 1000
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/4), rep(1,N/4))
X[which(Y == 1),] <- X[which(Y==1),] + 1.5
experiment_data <- data.frame(X=X, Y=as.factor(Y))

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = Y)) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

################### We create training/ test sets
set.seed(100)
sample_size <- floor(0.75 * nrow(experiment_data))

train_index <- sample(seq_len(nrow(experiment_data)), size = sample_size)
experiment_data_train <- experiment_data[train_index,]
experiment_data_test <- experiment_data[-train_index,]

###################### We now use maximal margin classifier to create the hyperplane that separates the data
my_maximal <- svm(Y~., data = experiment_data_train, kernel = "linear", scale = FALSE)
plot(my_maximal, experiment_data_train, color.palette = terrain.colors)

## Classification accuracy on training set
table(my_maximal$fitted, experiment_data_train$Y)

## Classification accuracy on test set
Y_pred <- predict(my_maximal, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_test <- sum(diag(table_test))/ sum(table_test)
accuracy_test
