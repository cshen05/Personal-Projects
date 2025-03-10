#### This file is for an illustration of regression splines

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(splines) #For splines regression

######################################################### Splines for regression

########## We look at the range of hp
range(mtcars$hp)

######################## We manually choose the knots at 100, 200, 300 

### We consider first order polynomial regression at each region. We choose the knots at 100, 200, 300
my_spline <- lm(mpg~bs(hp, knots = c(100,200,300), degree = 1), data = mtcars)
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, knots = c(100,200,300), degree = 1), se = FALSE, colour="red")

### We consider second order polynomial regression at each region. We choose the cut points at 100, 200, 300
my_spline <- lm(mpg~bs(hp, knots = c(100,200,300), degree = 2), data = mtcars)
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, knots = c(100,200,300), degree = 2), se = FALSE, colour="red")

### We consider cubic order polynomial regression at each region. We choose the cut points at 100, 200, 300
my_spline <- lm(mpg~bs(hp, knots = c(100,200,300), degree = 3), data = mtcars)
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, knots = c(100,200,300), degree = 3), se = FALSE, colour="red")

################ We use K-fold cross validation to choose the best order spline
K = 5
N = nrow(mtcars)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
set.seed(1000)
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 5 #The maximum degree
fold_error_rate <- matrix(0, nrow = M, ncol = K)

for (i in 1:M)
{
  for (j in 1:K)
  {
    train_index = which(fold_index != j)
    mtcars_train = mtcars[train_index,]
    mtcars_test = mtcars[-train_index,]
    
    my_spline <- lm(mpg~bs(hp, knots = c(100,200,300), degree = i), data = mtcars_train)
    mpg_pred <- predict(my_spline, data = mtcars_test)
    fold_error_rate[i,j] <- RMSE(mpg_pred, mtcars_test$mpg)
  }
}

### Plot the RMSE for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(1,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Degree of Polynomial in Spline") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(1,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Degree of Polynomial in Spline") + ylab("Error rate")


######################################## We now use quantile to choose the knots. 
########### These knots are usually chosen at the 25-th, 50-th, and 75-th quantile of the data

################# With M = 3 knots

#### Linear model
my_spline <- lm(mpg~bs(hp, df = 4, degree = 1), data = mtcars)  #For linear model, with three knots we have df = 4 degree of freedoms in R
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, df = 4, degree = 1), se = FALSE, colour="red")

#### Quadratic model
my_spline <- lm(mpg~bs(hp, df = 5, degree = 2), data = mtcars)  #For quadratic model, with three knots we have 5 degree of freedoms in R
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, df = 5, degree = 2), se = FALSE, colour="red")

#### Cubic model
my_spline <- lm(mpg~bs(hp, df = 6, degree = 3), data = mtcars)  #For cubic model, with three knots we have 6 degree of freedoms
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, df = 6, degree = 3), se = FALSE, colour="red")

################ We use K-fold cross validation to choose the best order spline
K = 10
N = nrow(mtcars)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
set.seed(1000)
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 5 #The maximum degree of polynomial
fold_error_rate <- matrix(0, nrow = M, ncol = K)

for (i in 1:M)
{
  for (j in 1:K)
  {
    train_index = which(fold_index != j)
    mtcars_train = mtcars[train_index,]
    mtcars_test = mtcars[-train_index,]
    
    my_spline <- lm(mpg~bs(hp, df = 3 + i, degree = i), data = mtcars_train)
    mpg_pred <- predict(my_spline, data = mtcars_test)
    fold_error_rate[i,j] <- RMSE(mpg_pred, mtcars_test$mpg)
  }
}

### Plot the RMSE for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(1,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Degree of Polynomial in Spline") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(1,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Degree of Polynomial in Spline") + ylab("Error rate")

##################### We use K-fold cross validation to choose the best number of knots for linear spline
K = 10
N = nrow(mtcars)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
set.seed(1000)
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 10 #The maximum number of knots
fold_error_rate <- matrix(0, nrow = M, ncol = K)

for (i in 1:M)
{
  for (j in 1:K)
  {
    train_index = which(fold_index != j)
    mtcars_train = mtcars[train_index,]
    mtcars_test = mtcars[-train_index,]
    
    my_spline <- lm(mpg~bs(hp, df = 1 + i, degree = 1), data = mtcars_train)
    mpg_pred <- predict(my_spline, data = mtcars_test)
    fold_error_rate[i,j] <- RMSE(mpg_pred, mtcars_test$mpg)
  }
}

### Plot the RMSE for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(1,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Number of Knots") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
best_index <- which.min(mean_fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(1,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Number of Knots") + ylab("Error rate")

##### Plot the spline with best number of knots
my_spline <- lm(mpg~bs(hp, df = 1 + best_index, degree = 1), data = mtcars)  #For linear model, with three knots we have df = 4 degree of freedoms in R
summary(my_spline)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~bs(x, df = 1 + best_index, degree = 1), se = FALSE, colour="red")

