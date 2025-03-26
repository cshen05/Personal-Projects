#### This file is for an illustration of Problem 9 in Section 8.4

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(tree) #This library is used for classification and regression trees
library(ISLR)

### We first load the OJ dataset
data(OJ)
head(OJ)

### Part (a)
set.seed(100)
sample_size <- 800

train_index <- sample(seq_len(nrow(OJ)), size = sample_size)
OJ_train <- OJ[train_index,]
OJ_test <- OJ[-train_index,]

### Part (b)
my_tree <- tree(Purchase~., data = OJ_train)
summary(my_tree)

### Parts (c) and (d)
plot(my_tree)
text(my_tree, pretty = 0)

### Part (e)
Purchase_pred <- predict(my_tree, newdata = OJ_test, type = "class")
table_Purchase <- table(Purchase_pred, OJ_test$Purchase)

accuracy_Purchase_test <- sum(diag(table_Purchase))/ sum(table_Purchase)
accuracy_Purchase_test

### Parts (f), (g), (h) 
cv_OJ <- cv.tree(my_tree,  K = 10) #10-fold cross validation
plot(cv_OJ$size, cv_OJ$dev, type = 'b', xlab = 'Tree size', ylab = 'Cross Validation error') 

### Part (i)
prune_cv_OJ <- prune.tree(my_tree, best = 5)
summary(prune_cv_OJ)
plot(prune_cv_OJ)
text(prune_cv_OJ)

### Part (j)
Prune_Purchase_pred <- predict(prune_cv_OJ, newdata = OJ_test, type = "class")
Prune_table_Purchase <- table(Prune_Purchase_pred, OJ_test$Purchase)

accuracy_Prune_Purchase_test <- sum(diag(Prune_table_Purchase))/ sum(Prune_table_Purchase)
accuracy_Prune_Purchase_test

############ Bagging, random forest, boosting

## Bagging
my_bag <- randomForest(Purchase~., data = OJ_train, mtry = 17, importance = TRUE) 

## Test error of bagging
Purchase_pred <- predict(my_bag, newdata = OJ_test, type = "class")
table_Purchase <- table(Purchase_pred, OJ_test$Purchase)

accuracy_Purchase_test <- sum(diag(table_Purchase))/ sum(table_Purchase)
accuracy_Purchase_test

## Random forest
my_forest <- randomForest(Purchase~., data = OJ_train, mtry = 6, importance = TRUE) 

## Test error of random forest
Purchase_pred <- predict(my_forest, newdata = OJ_test, type = "class")
table_Purchase <- table(Purchase_pred, OJ_test$Purchase)

accuracy_Purchase_test <- sum(diag(table_Purchase))/ sum(table_Purchase)
accuracy_Purchase_test

## Boosting
## Convert Purchase into {0,1} format before using
my_boosting <- gbm(Purchase~., data = OJ_train, distribution = "bernoulli", n.trees= 5000, interaction.depth = 3, shrinkage = 0.001)

## Test error of boosting
Purchase_pred <- predict(my_boosting, newdata = OJ_test, type = "class")
table_Purchase <- table(Purchase_pred, OJ_test$Purchase)

accuracy_Purchase_test <- sum(diag(table_Purchase))/ sum(table_Purchase)
accuracy_Purchase_test