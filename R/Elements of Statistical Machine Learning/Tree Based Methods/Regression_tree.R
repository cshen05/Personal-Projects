#### This file is for an illustration of regression tree

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(tree) #This library is used for classification and regression trees

################################################################################
############## We first try with the "Real_estate.csv"
realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

################################################################################ We first consider the predictors house age and distance to station
new_realestate <- realestate[,c(3, 4, 8)]

############## We create training and test sets
set.seed(100)
sample_size <- floor(0.75 * nrow(new_realestate))

train_index <- sample(seq_len(nrow(new_realestate)), size = sample_size)
realestate_train <- new_realestate[train_index,]
realestate_test <- new_realestate[-train_index,]

############# Regression tree
my_tree <- tree(price~., data = realestate_train)
summary(my_tree)

############ Plot the regression tree
plot(my_tree)
text(my_tree, pretty = 0)

############ Prediction with the original regression tree
price_pred <- predict(my_tree, newdata = realestate_test)
test_error <- RMSE(price_pred, realestate_test$price)
test_error
############ We now determine whether pruning the regression tree improves the RMSE
set.seed(10)
cv_estate <- cv.tree(my_tree,  K = 10) #10-fold cross validation
plot(cv_estate$size, cv_estate$dev, type = 'b', xlab = 'Tree size', ylab = 'Cross validation error') #cv_estate$dev means that we take the deviances of different tree size

###################################################### Constructing pruning tree with the best size from the K-fold cross validation
prune_cv_estate <- prune.tree(my_tree, best = 5)
summary(prune_cv_estate)
plot(prune_cv_estate)
text(prune_cv_estate)

################################################################################ We now consider all the predictors
set.seed(100)
sample_size <- floor(0.75 * nrow(realestate))

train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
realestate_train <- realestate[train_index,]
realestate_test <- realestate[-train_index,]

############# Regression tree
my_tree <- tree(price~., data = realestate_train)
summary(my_tree)

############ Plot the regression tree
plot(my_tree)
text(my_tree, pretty = 0)

############ Prediction with the original regression tree
price_pred <- predict(my_tree, newdata = realestate_test)
test_error <- RMSE(price_pred, realestate_test$price)

############ We now determine whether pruning the regression tree improves the RMSE
set.seed(10)
cv_estate <- cv.tree(my_tree,  K = 10) #10-fold cross validation
plot(cv_estate$size, cv_estate$dev, type = 'b', xlab = 'Tree size', ylab = 'Cross validation error') #cv_estate$dev means that we take the deviances of different tree size

###################################################### Constructing pruning tree with the best size from the K-fold cross validation
prune_cv_estate <- prune.tree(my_tree, best = 5)
summary(prune_cv_estate)
plot(prune_cv_estate)
text(prune_cv_estate)

################################################################################ Comparing to linear models
my_lm <- lm(price~., data = realestate_train)
summary(my_lm)

price_pred <- predict(my_lm, newdata = realestate_test)
test_error <- RMSE(price_pred, realestate_test$price)

tree_price_pred <- predict(prune_cv_estate, newdata = realestate_test)
tree_test_error <- RMSE(tree_price_pred, realestate_test$price)

















