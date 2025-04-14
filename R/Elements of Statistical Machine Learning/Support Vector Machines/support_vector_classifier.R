#### This file is for an illustration of maximal margin classifier

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(e1071) #This package is for support vector machines, maximum margin classifier, support vector classifier
library(tree)
#### We create training data that are linearly separable

## We first generate the data
set.seed(1000)
N = 1000
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/4), rep(1,N/4))
X[which(Y == 1),] <- X[which(Y==1),] + 1.5
experiment_data <- data.frame(X=X, Y=as.factor(Y))

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = Y)) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

## We create training and test sets
set.seed(100)
sample_size <- floor(0.75 * nrow(experiment_data))

train_index <- sample(seq_len(nrow(experiment_data)), size = sample_size)
experiment_data_train <- experiment_data[train_index,]
experiment_data_test <- experiment_data[-train_index,]

## We now use support vector classifier to create a hyperplane that separates the data
my_support_classifier <- svm(Y~., data = experiment_data_train, kernel = "linear", cost = 0.1, scale = FALSE) 
# cost = 1/C, namely, its plays the kind of role of 1/ C, namely, 
# when cost is small, the margin is wide and when cost is large, the margin is narrow
plot(my_support_classifier, experiment_data_train, color.palette = terrain.colors)

# Classification accuracy
Y_pred <- predict(my_support_classifier, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)
table_test

accuracy_test <- sum(diag(table_test))/ sum(table_test)
accuracy_test

########################################### Choose the best tuning parameter via cross-validation
cost_val = seq(0.001, 100, length = 200)
tune_err = tune(svm, Y~., data = experiment_data_train, kernel = "linear", ranges = list (cost = cost_val))

## We choose the best model with best tuning parameter
best_model = tune_err$best.model
best_model
## ## Classification accuracy on test set with best model
Y_pred <- predict(best_model, newdata = experiment_data_test)
table_test <- table(Y_pred, experiment_data_test$Y)

best_accuracy_test <- sum(diag(table_test))/ sum(table_test)
best_accuracy_test

############################# Performance from tree method
library(tree) #This library is used for classification and regression trees

## Single classification tree
my_tree <- tree(Y~., experiment_data_train)
summary(my_tree)

plot(my_tree)
text(my_tree, pretty = 0)

############ Classification accuracy in the test set
Y_pred <- predict(my_tree, newdata = experiment_data_test, type = "class")
table_test <- table(Y_pred, experiment_data_test$Y)

accuracy_tree_test <- sum(diag(table_test))/ sum(table_test)
accuracy_tree_test


