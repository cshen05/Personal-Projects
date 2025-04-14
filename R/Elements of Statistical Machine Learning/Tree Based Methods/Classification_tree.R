#### This file is for an illustration of classification tree

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(tree) #This library is used for classification and regression trees

################################################################################
############## We first try with the "Heart_disease.csv"
heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

### Remove missing data from data
heartdisease <- na.omit(heartdisease)

################################################################################ We first only consider the predictors totChol, diaBP, glucose
new_heartdisease <- heartdisease[,c(10, 12, 15, 16)]

############## We create training and test sets
sample_size <- floor(0.75 * nrow(new_heartdisease))

train_index <- sample(seq_len(nrow(new_heartdisease)), size = sample_size)
heartdisease_train <- new_heartdisease[train_index,]
heartdisease_test <- new_heartdisease[-train_index,]

heartdisease_train$Heart_disease <- as.factor(heartdisease_train$Heart_disease)

############# Classification tree
my_tree <- tree(Heart_disease~., heartdisease_train)
### Tree can be used for either regression or classification tree
### R will automatically detect whether this is classication or regression task
summary(my_tree)

############ Plot the classification tree
plot(my_tree)
text(my_tree, pretty = 0)

############ Classification accuracy in the test set
heart_pred <- predict(my_tree, newdata = heartdisease_test, type = "class")
table_heart <- table(heart_pred, heartdisease_test$Heart_disease)

accuracy_heart_test <- sum(diag(table_heart))/ sum(table_heart)
accuracy_heart_test

############################## We now determine whether pruning the classification tree improves the error
cv_heart <- cv.tree(my_tree,  K = 10) #10-fold cross validation
### cv.tree = cross validation on either regression/ classification tree
### K = 10 means we have K = 10 folds in K-fold cross validation
### We usually use K = 5 or K = 10 in K-fold cross validation because it
### gives a good trade-off between bias and variance (and also computation)
### my_tree is the name of the original tree
plot(cv_heart$size, cv_heart$dev, type = 'b', xlab = 'Tree size', ylab = 'Cross Validation error') 

################################################################################ We now consider all the predictors
set.seed(100)
sample_size <- floor(0.75 * nrow(heartdisease))

train_index <- sample(seq_len(nrow(heartdisease)), size = sample_size)
heartdisease_train <- heartdisease[train_index,]
heartdisease_test <- heartdisease[-train_index,]

heartdisease_train$Heart_disease <- as.factor(heartdisease_train$Heart_disease)

############# Classification tree
my_tree <- tree(Heart_disease~., heartdisease_train)
### Tree can be used for either regression or classification tree
### R will automatically detect whether this is classication or regression task
summary(my_tree)

############ Plot the classification tree
plot(my_tree)
text(my_tree, pretty = 0)

############ Classification accuracy in the test set
heart_pred <- predict(my_tree, newdata = heartdisease_test, type = "class")
table_heart <- table(heart_pred, heartdisease_test$Heart_disease)

accuracy_heart_test <- sum(diag(table_heart))/ sum(table_heart)
accuracy_heart_test

############ We now determine whether pruning the classification tree improves the error
set.seed(10)
cv_heart <- cv.tree(my_tree,  K = 10) #10-fold cross validation
### cv.tree = cross validation on either regression/ classification tree
### K = 10 means we have K = 10 folds in K-fold cross validation
### We usually use K = 5 or K = 10 in K-fold cross validation because it
### gives a good trade-off between bias and variance (and also computation)
### my_tree is the name of the original tree
plot(cv_heart$size, cv_heart$dev, type = 'b', xlab = 'Tree size', ylab = 'Cross Validation error') 

###################################################### Constructing pruning tree with the best size from the K-fold cross validation
prune_cv_estate <- prune.tree(my_tree, best = 2)
summary(prune_cv_estate)
plot(prune_cv_estate)
text(prune_cv_estate)


