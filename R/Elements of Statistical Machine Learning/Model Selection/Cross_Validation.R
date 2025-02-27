#### This file is for an illustration of cross-validation with "Real_estate.csv" data

library(ggplot2)
library(tidyr)
library(dplyr)
library(FNN) # For K-nearest neighbor regression
library(MLmetrics) #For RMSE calculation

################################################################################
############## We first try with the "Real_estate.csv"

realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

#################################################### Comparison between LOOCV and validation set approach when Y = real estate price and X = house age

################### Validation set approach for 10 training/ test sets

set.seed(1000)
N = 10 # number of training / test sets
M = 150 # the number of possible values of K
K <- seq(3, M+2)
rmse_out_val <- matrix(0, nrow = M, ncol = N)

### 75% of the sample size for the training set
sample_size <- floor(0.75 * nrow(realestate))

for (i in 1:N)
{
  train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
  realestate_train <- realestate[train_index,]
  realestate_test <- realestate[-train_index,]

  age_train <- data.frame(realestate_train$house_age)
  age_test <- data.frame(realestate_test$house_age)
  price_train <- data.frame(realestate_train$price)
  price_test <- as.vector(realestate_test$price)

  for (j in 1:M)
  {
    knnfit = knn.reg(age_train, age_test, price_train, k = K[j])
    rmse_out_val[j,i] = RMSE(knnfit$pred, price_test)
  }
}

### Plot the RMSE for 10 training/ test sets at different K
new_rmse_out_val <- as.data.frame(rmse_out_val)
plot <- ggplot(data = new_rmse_out_val, aes(x = K))
for (i in 1:N)
{
  plot <- plot + geom_line( aes_string( y = new_rmse_out_val[,i]), col = i+2)
}
plot + ylab('rmse_out')

#### Take the average of error rates over N training/ test sets
average_rmse_out_val <- rowMeans(rmse_out_val) #We take the average of each row in the matrix rmse_out_val
average_rmse_out_val <- as.data.frame(average_rmse_out_val)
ggplot(average_rmse_out_val, aes(x = K, y = average_rmse_out_val)) + geom_line(col = "red")

################### LOOCV

#### We first need to remove missing data from the dataset

## We first create a data frame with only column real estate price and house age
myvars <- c("house_age", "price")
realestate_sub <- realestate[myvars]
realestate_sub <- realestate_sub[complete.cases(realestate_sub), ] #Remove missing data

n = nrow(realestate_sub) #Number of data in the real estate dataset, amount of training/ test sets we have in LOOCV
error_rate <- rep(0, M) #We create a vector of error rates with size M for the M values of K
for (i in 1:M)
{
  for (j in 1:n)
  {
    age_train <- realestate_sub$house_age[-j]
    age_test <- realestate_sub$house_age[j]
    price_train <- realestate_sub$price[-j]
    price_test <- realestate_sub$price[j]
    knnfit = knn.reg(age_train, age_test, price_train, k = K[i])
    error_rate[i] <- error_rate[i] + RMSE(knnfit$pred, price_test)
  }
  error_rate[i] <- error_rate[i]/ n
}

## Plot the RMSE values
error_rate <-as.data.frame(error_rate)
ggplot(error_rate, aes(x = K, y = error_rate)) + geom_line()

#################################################### Performance of K-fold cross validation approach when Y = real estate price and X = house age

### We divide data into K folds
K = 10 
N = nrow(realestate_sub)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 10 #The number of possible degress of polynomial regression
fold_error_rate <- matrix(0, nrow = M, ncol = K)

for (i in 1:M)
{
  for (j in 1:K)
  {
    train_index = which(fold_index != j)
    realestate_sub_train = realestate_sub[train_index,]
    realestate_sub_test = realestate_sub[-train_index,]
    
    polyfit <- lm(price~poly(house_age,i), data = realestate_sub_train)
    pricetest_pred <- predict(polyfit, data = realestate_sub_test)
    fold_error_rate[i,j] <- RMSE(pricetest_pred, realestate_sub_test$price)
  }
}

### Plot the RMSE for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(1:M)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Degree of Polynomial") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(1:M), y = mean_fold_error_rate)) + geom_line() + xlab("Degree of Polynomial") + ylab("Error rate")


