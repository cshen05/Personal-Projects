#### This file is for an illustration of shrinkage methods with "Real_estate.csv" data

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics)

# Install packges for ridge regression and the Lasso
#install.packages("glmnet")
library(glmnet)

#install.packages("seriation")
library(seriation)

################################################################################
############## We first try with the "Real_estate.csv"

realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

### We remove column "No", "latitude", and "longtitude"
realestate <- realestate[,c(-1, -6, -7)]


############################################################ Ridge regression

## We first change data into appropriate forms
response_val <- realestate$price
predictors_val <- model.matrix(price~.,realestate)[,-1]

######### We first try ridge regression when lambda  = 1
my_ridge <- glmnet(predictors_val, response_val, alpha = 0, lambda = 1) 
#alpha = 0 means we perform ridge regression
coef(my_ridge)

######### We now try ridge regression when lambda  = 100
my_ridge <- glmnet(predictors_val, response_val, alpha = 0, lambda = 100) #alpha = 0 means we perform ridge regression
coef(my_ridge)

######## We consider 100 values of lambda from 0 to 1000
lambda_val <- seq(1000,0, length = 100)
my_ridge <- glmnet(predictors_val, response_val, alpha = 0, lambda = lambda_val) #alpha = 0 means we perform ridge regression

## Output the coefficients of ridge regression
coef(my_ridge)

## Permute the columns of matrix
coef_my_ridge <- as.matrix(coef(my_ridge))
order_seq <- seq(100, 1, length = 100)
coef_my_ridge <- permute(coef_my_ridge, ser_permutation(NA, order_seq))

## Plot the coefficients of ridge regression versus the values of lambda
coef_my_ridge <- t(coef_my_ridge)
coef_my_ridge <- as.data.frame(coef_my_ridge)
plot <- ggplot(data = coef_my_ridge, aes(x = seq(0,1000, length = 100)))
for (i in 1:4)
{
  plot <- plot + geom_line( aes_string(y = as.vector(coef_my_ridge[,i+1])), col = i+2)
}
plot + xlab('lambda') + ylab('estimated coefficients')

###################################################### We use cross-validation approaches to choose "best" lambda


################################### Validation set approach for 10 training/ test sets

############# We create training sets whose sizes are much larger than the number of predictors

set.seed(1000)
N = 10 # number of training / test sets
M = 100 # number of values of lambda
lambda_val <- seq(0,1000, length = M) # sequence of values of lambda
rmse_out_val <- matrix(0, nrow = M, ncol = N)

### 75% of the sample size for the training set
sample_size <- floor(0.75 * nrow(realestate))

for (i in 1:N)
{
  train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
  realestate_train <- realestate[train_index,]
  realestate_test <- realestate[-train_index,]
  
  response_train_val <- realestate_train$price
  predictors_train_val <- model.matrix(price~.,realestate_train)[,-1]
  response_test_val <- realestate_test$price
  predictors_test_val <- model.matrix(price~.,realestate_test)[,-1]
  
  for (j in 1:M)
  {
    my_ridge <- glmnet(predictors_train_val, response_train_val, alpha = 0, lambda = lambda_val[j])
    my_ride_pred <- predict(my_ridge, s = lambda_val[j], predictors_test_val)
    rmse_out_val[j,i] = RMSE(my_ride_pred, response_test_val)
  }
}

### Plot the RMSE for 10 training/ test sets at different lambda
new_rmse_out_val <- as.data.frame(rmse_out_val)
plot <- ggplot(data = new_rmse_out_val, aes(x = lambda_val))
for (i in 1:N)
{
  plot <- plot + geom_line( aes_string( y = new_rmse_out_val[,i]), col = i+2)
}
plot + ylab('rmse_out')

#### Take the average of error rates over N training/ test sets
average_rmse_out_val <- rowMeans(rmse_out_val) #We take the average of each row in the matrix rmse_out_val
average_rmse_out_val <- as.data.frame(average_rmse_out_val)
ggplot(average_rmse_out_val, aes(x = lambda_val, y = average_rmse_out_val)) + geom_line(col = "red")

################################ We create small training data sets whose sizes are slightly larger than the number of predictors

set.seed(1000)
N = 10 # number of training / test sets
M = 100 # number of values of lambda
lambda_val <- seq(0,1000, length = M) # sequence of values of lambda
rmse_out_val <- matrix(0, nrow = M, ncol = N)

sample_size <- floor(0.015 * nrow(realestate))

for (i in 1:N)
{
  train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
  realestate_train <- realestate[train_index,]
  realestate_test <- realestate[-train_index,]
  
  response_train_val <- realestate_train$price
  predictors_train_val <- model.matrix(price~.,realestate_train)[,-1]
  response_test_val <- realestate_test$price
  predictors_test_val <- model.matrix(price~.,realestate_test)[,-1]
  
  for (j in 1:M)
  {
    my_ridge <- glmnet(predictors_train_val, response_train_val, alpha = 0, lambda = lambda_val[j])
    my_ride_pred <- predict(my_ridge, s = lambda_val[j], predictors_test_val)
    rmse_out_val[j,i] = RMSE(my_ride_pred, response_test_val)
  }
}

### Plot the RMSE for 10 training/ test sets at different lambda
new_rmse_out_val <- as.data.frame(rmse_out_val)
plot <- ggplot(data = new_rmse_out_val, aes(x = lambda_val))
for (i in 1:N)
{
  plot <- plot + geom_line( aes_string( y = new_rmse_out_val[,i]), col = i+2)
}
plot + ylab('rmse_out')

#### Take the average of error rates over N training/ test sets
average_rmse_out_val <- rowMeans(rmse_out_val) #We take the average of each row in the matrix rmse_out_val
average_rmse_out_val <- as.data.frame(average_rmse_out_val)
ggplot(average_rmse_out_val, aes(x = lambda_val, y = average_rmse_out_val)) + geom_line(col = "red")


############################################################### The Lasso

######### We first try the lasso when lambda  = 1
my_lasso <- glmnet(predictors_val, response_val, alpha = 1, lambda = 1) #alpha = 1 means we perform the lasso
coef(my_lasso)

######### We then try the lasso when lambda  = 2
my_lasso <- glmnet(predictors_val, response_val, alpha = 1, lambda = 6) #alpha = 1 means we perform the lasso
coef(my_lasso)

######## We consider 100 values of lambda from 0 to 10
lambda_val <- seq(10,0, length = 100)
my_lasso <- glmnet(predictors_val, response_val, alpha = 1, lambda = lambda_val) #alpha = 1 means we perform the lasso

## Output the coefficients of the lasso
coef(my_lasso)

## Permute the columns of matrix
coef_my_lasso <- as.matrix(coef(my_lasso))
order_seq <- seq(100, 1, length = 100)
coef_my_lasso <- permute(coef_my_lasso, ser_permutation(NA, order_seq))

## Plot the coefficients of ridge regression versus the values of lambda
coef_my_lasso <- t(coef_my_lasso)
coef_my_lasso <- as.data.frame(coef_my_lasso)
plot <- ggplot(data = coef_my_lasso, aes(x = seq(0,10, length = 100)))
for (i in 1:4)
{
  plot <- plot + geom_line( aes_string(y = as.vector(coef_my_lasso[,i+1])), col = i+2)
}
plot + xlab('lambda') + ylab('estimated coefficients')

###################################################### 
###### We use cross-validation approaches to choose "best" lambda


################################### Validation set approach for 10 training/ test sets

################ When the size of training sets is much larger than the number of predictors
set.seed(1000)
N = 10 # number of training / test sets
M = 100 # number of values of lambda
lambda_val <- seq(0,10, length = M) # sequence of values of lambda
rmse_out_val <- matrix(0, nrow = M, ncol = N)

### 75% of the sample size for the training set
### Sample size = 310
sample_size <- floor(0.75 * nrow(realestate))
#$sample_size <- 3

for (i in 1:N)
{
  train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
  realestate_train <- realestate[train_index,]
  realestate_test <- realestate[-train_index,]
  
  response_train_val <- realestate_train$price
  predictors_train_val <- model.matrix(price~.,realestate_train)[,-1]
  response_test_val <- realestate_test$price
  predictors_test_val <- model.matrix(price~.,realestate_test)[,-1]
  
  for (j in 1:M)
  {
    my_lasso <- glmnet(predictors_train_val, response_train_val, alpha = 1, lambda = lambda_val[j])
    my_lasso_pred <- predict(my_lasso, s = lambda_val[j], predictors_test_val)
    rmse_out_val[j,i] = RMSE(my_lasso_pred, response_test_val)
  }
}

### Plot the RMSE for 10 training/ test sets at different lambda
new_rmse_out_val <- as.data.frame(rmse_out_val)
plot <- ggplot(data = new_rmse_out_val, aes(x = lambda_val))
for (i in 1:N)
{
  plot <- plot + geom_line( aes_string( y = new_rmse_out_val[,i]), col = i+2)
}
plot + ylab('rmse_out')

#### Take the average of error rates over N training/ test sets
average_rmse_out_val <- rowMeans(rmse_out_val) #We take the average of each row in the matrix rmse_out_val
average_rmse_out_val <- as.data.frame(average_rmse_out_val)
ggplot(average_rmse_out_val, aes(x = lambda_val, y = average_rmse_out_val)) + geom_line(col = "red")

################################## When the size of training sets is smaller than the number of predictors
set.seed(1000)
N = 10 # number of training / test sets
M = 100 # number of values of lambda
lambda_val <- seq(0,10, length = M) # sequence of values of lambda
rmse_out_val <- matrix(0, nrow = M, ncol = N)

### 75% of the sample size for the training set
#sample_size <- floor(0.75 * nrow(realestate))
sample_size <- 3

for (i in 1:N)
{
  train_index <- sample(seq_len(nrow(realestate)), size = sample_size)
  realestate_train <- realestate[train_index,]
  realestate_test <- realestate[-train_index,]
  
  response_train_val <- realestate_train$price
  predictors_train_val <- model.matrix(price~.,realestate_train)[,-1]
  response_test_val <- realestate_test$price
  predictors_test_val <- model.matrix(price~.,realestate_test)[,-1]
  
  for (j in 1:M)
  {
    my_lasso <- glmnet(predictors_train_val, response_train_val, alpha = 1, lambda = lambda_val[j])
    my_lasso_pred <- predict(my_lasso, s = lambda_val[j], predictors_test_val)
    rmse_out_val[j,i] = RMSE(my_lasso_pred, response_test_val)
  }
}

### Plot the RMSE for 10 training/ test sets at different lambda
new_rmse_out_val <- as.data.frame(rmse_out_val)
plot <- ggplot(data = new_rmse_out_val, aes(x = lambda_val))
for (i in 1:N)
{
  plot <- plot + geom_line( aes_string( y = new_rmse_out_val[,i]), col = i+2)
}
plot + ylab('rmse_out')

#### Take the average of error rates over N training/ test sets
average_rmse_out_val <- rowMeans(rmse_out_val) #We take the average of each row in the matrix rmse_out_val
average_rmse_out_val <- as.data.frame(average_rmse_out_val)
ggplot(average_rmse_out_val, aes(x = lambda_val, y = average_rmse_out_val)) + geom_line(col = "red")

