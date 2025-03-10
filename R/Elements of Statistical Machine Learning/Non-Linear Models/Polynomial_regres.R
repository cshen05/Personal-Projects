#### This file is for an illustration of polynomial regressions

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation

#### Polynomial regression for mtcars data;  Y = mpg (miles per galon), X = hp (horse power)

## Degree 2
my_lm <- lm(mpg~poly(hp, 2), data = mtcars)
summary(my_lm)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,2), se = FALSE, colour="red")

## Degree 3
my_lm <- lm(mpg~poly(hp, 3), data = mtcars)
summary(my_lm)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,3), se = FALSE, colour="red")

## Degree 4
my_lm <- lm(mpg~poly(hp, 4), data = mtcars)
summary(my_lm)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,4), se = FALSE, colour="red")

## Degree 5
my_lm <- lm(mpg~poly(hp, 5), data = mtcars)
summary(my_lm)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,5), se = FALSE, colour="red")

## K-fold cross validation to choose the best polynomial regression with degree at most 5
K = 5
N = nrow(mtcars)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 5 #The maximum degree
fold_error_rate <- matrix(0, nrow = M - 1, ncol = K)

for (i in 1:(M - 1))
{
  for (j in 1:K)
  {
    train_index = which(fold_index != j)
    mtcars_train = mtcars[train_index,]
    mtcars_test = mtcars[-train_index,]
    
    polyfit <- lm(mpg~poly(hp,i), data = mtcars_train)
    mpg_pred <- predict(polyfit, mtcars_test)
    fold_error_rate[i,j] <- RMSE(mpg_pred, mtcars_test$mpg)
  }
}

### Plot the RMSE for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(2,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Degree of Polynomial") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(2,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Degree of Polynomial") + ylab("Error rate")




############################## Polynomial logistic regression with the heart disease dataset

setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

################################################################################
############## We first try with the "Heart_disease.csv"
heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

############# Remove missing values
heartdisease <- na.omit(heartdisease)


############# Polynomial Logistic regression for heart disease data;  Y = Heart_disease, X = diaBP 

#### Second degree
my_glm <- glm(Heart_disease~poly(diaBP,2), data = heartdisease, family = "binomial")
summary(my_glm)

logis_plot <- ggplot(heartdisease, aes(x=diaBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', formula= y~poly(x,2), se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

#### Third degree
my_glm <- glm(Heart_disease~poly(diaBP,3), data = heartdisease, family = "binomial")
summary(my_glm)

logis_plot <- ggplot(heartdisease, aes(x=diaBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', formula= y~poly(x,3), se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

#### Fourth degree
my_glm <- glm(Heart_disease~poly(diaBP,4), data = heartdisease, family = "binomial")
summary(my_glm)

logis_plot <- ggplot(heartdisease, aes(x=diaBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', formula= y~poly(x,4), se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

#### Fifth degree
my_glm <- glm(Heart_disease~poly(diaBP,5), data = heartdisease, family = "binomial")
summary(my_glm)

logis_plot <- ggplot(heartdisease, aes(x=diaBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', formula= y~poly(x,5), se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

################################## K-fold cross validation to choose the best polynomial regression with degree at most 5
K = 10
N = nrow(heartdisease)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
set.seed(1000)
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 5 #The maximum degree
fold_error_rate <- matrix(0, nrow = M, ncol = K)

for (i in 1:M) # degree of polynomials
{
  for (j in 1:K) # folds of K-fold
  {
    train_index = which(fold_index != j)
    heartdisease_train = heartdisease[train_index,]
    heartdisease_test = heartdisease[-train_index,]
    
    polyfit <- glm(Heart_disease~poly(diaBP,i), data = heartdisease_train, family = "binomial")
    heartdisease_pred <- predict(polyfit, heartdisease_test)
    yhat_predict_test <- ifelse(heartdisease_pred > 0.5, 1, 0)
    table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
    fold_error_rate[i,j] <- sum(diag(table_heart_test))/ sum(table_heart_test)
  }
}

### Plot the classification accuracy for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(1,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Degree of Polynomial") + ylab("Classification accuracy")

## Plot the average classification accuracy
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(1,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Degree of Polynomial") + ylab("Classification accuracy")
