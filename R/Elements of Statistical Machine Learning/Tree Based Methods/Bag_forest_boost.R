#### This file is for an illustration of bagging/ random forests/ boosting

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(tree) #This library is used for classification and regression trees
library(randomForest) #This library is used for bagging and random forests
library(gbm) #This library is used for boosting

################################################################################
############## We first try with the "agriculture_worldbank.csv"
realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

############################################ Create training and test set
set.seed(100)
sample_size <- floor(0.75 * nrow(realestate))

train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
realestate_train <- realestate[train_index,]
realestate_test <- realestate[-train_index,]

################################################################################ Bagging with all predictors
ncol_realestate <- ncol(realestate)
# ncol_realestate computes the number of columns in the data

############################# We first consider bagging with 100 trees
my_bag <- randomForest(price~., data = realestate_train, mtry = ncol_realestate - 1, ntree = 100, importance = TRUE) 
#mtry indicates how many predictors we consider each time we split the tree. 
# Here, for bagging, we consider all the possible 7 predictors
# ntree indicates the number of trees to grow
#importance indicates whether we would like to assess the importance of the predictors

#### We look at the test error
price_pred <- predict(my_bag, newdata = realestate_test)
bag_test_error <- RMSE(price_pred, realestate_test$price)
bag_test_error
############################# We now determine the best number of trees that gives the best error
sequence_tree <- seq(100, 1000, by = 100)
# We create the number of trees in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
bag_test_error <- rep(0, length(sequence_tree))
bag_test_error
# We create a sequence of test errors with 0 values for each number of trees
for (i in 1:length(sequence_tree))
{
  my_bag <- randomForest(price~., data = realestate_train, mtry = ncol_realestate - 1, ntree = sequence_tree[i], importance = TRUE)
  price_pred <- predict(my_bag, newdata = realestate_test)
  bag_test_error[i] <- RMSE(price_pred, realestate_test$price)
}

### Plot the test errors
bag_test_error_data_frame <- as.data.frame(bag_test_error)
ggplot(data = bag_test_error_data_frame, aes(x = seq(100,1000, by = 100), y = bag_test_error)) + geom_line() + xlab("The number of trees") + ylab("Test error")

################# Compare bagging to single regression tree
single_tree <- tree(price~., data = realestate_train)
summary(single_tree)

price_pred <- predict(single_tree, newdata = realestate_test)
single_tree_test_error <- RMSE(price_pred, realestate_test$price)
single_tree_test_error

################## We look at the importance of predictors
importance(my_bag)

################## We plot the importance of predictors
varImpPlot(my_bag)


################################################################################ Random Forest

#################### We first build random forest where we only split 2 predictors each time
my_forest <- randomForest(price~., data = realestate_train, mtry = 2, ntree = 700, importance = TRUE) 
#mtry indicates how many predictors we consider each time we split the tree.
#importance indicates whether we would like to assess the importance of the predictors

#### We look at the test error
price_pred <- predict(my_forest, newdata = realestate_test)
forest_test_error <- RMSE(price_pred, realestate_test$price)
forest_test_error

#################### We now consider the best number of predictors each time we split the trees
sequence_predictors <- seq(1, ncol_realestate - 1, by = 1)
# We create the number of predictors in {1, 2, ..., 7}
forest_test_error <- rep(0, length(sequence_predictors))
# We create a sequence of test errors with 0 values for each number of trees
for (i in 1:length(sequence_predictors))
{
  my_forest <- randomForest(price~., data = realestate_train, mtry = sequence_predictors[i], ntree = 700, importance = TRUE)
  price_pred <- predict(my_forest, newdata = realestate_test)
  forest_test_error[i] <- RMSE(price_pred, realestate_test$price)
}

### Plot the test errors
forest_test_error_data_frame <- as.data.frame(forest_test_error)
head(forest_test_error_data_frame)
ggplot(data = forest_test_error_data_frame, aes(x = seq(1,7, by = 1), y = forest_test_error)) + geom_line() + xlab("The number of predictors") + ylab("Test error")

######################## We look at the importance of predictors
importance(my_forest)

####################### We plot the importance of predictors
varImpPlot(my_forest)

################################################################################ Boosting
my_boost <- gbm(price~., data = realestate_train, distribution = "gaussian", n.trees= 10000, interaction.depth = 3, shrinkage = 0.001)
# distribution indicates the option for regression or classification. 
# distribution = "gaussian" is for regression and distribution = "bernoulli" is for binary classification
# n.trees indicates the number of trees (it is B in the slides)
# interaction.depth indicates the maximum depth of tree (it is d in the slides)
# shrinkage indicates the value of regularized parameter lambda
summary(my_boost)

##### Test error
price_pred <- predict(my_boost, newdata = realestate_test)
boost_test_error <- RMSE(price_pred, realestate_test$price)
boost_test_error
#################### We now consider the best number of trees when the shrinkage is 0.001
sequence_tree_boosting <- seq(1000, 20000, by = 1000)
# We create the number of predictors in {1, 2, ..., 7}
boosting_test_error <- rep(0, length(sequence_tree_boosting))
# We create a sequence of test errors with 0 values for each number of trees
for (i in 1:length(sequence_tree_boosting))
{
  my_boost <- gbm(price~., data = realestate_train, distribution = "gaussian", n.trees= sequence_tree_boosting[i], interaction.depth = 3, shrinkage = 0.0005)
  price_pred <- predict(my_boost, newdata = realestate_test)
  boosting_test_error[i] <- RMSE(price_pred, realestate_test$price)
}

### Plot the test errors
boosting_test_error_data_frame <- as.data.frame(boosting_test_error)
min(boosting_test_error_data_frame)
ggplot(data = boosting_test_error_data_frame, aes(x = seq(1000,20000, by = 1000), y = boosting_test_error)) + geom_line() + xlab("The number of trees") + ylab("Test error")

