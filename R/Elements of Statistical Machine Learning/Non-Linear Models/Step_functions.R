#### This file is for an illustration of step function

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation


########################### Step functions for regression

######## We make certain changes in mtcars
mtcars <- mtcars[-31,]

######## We use R to automatically create the cut points for X = hp (horse power)
M = 7 # number of cut points
table(cut(mtcars$hp, M))

####### We use these cut points for regression
M = 7
my_step <- lm(mpg~cut(hp, M), data = mtcars)
summary(my_step)

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~cut(x,M), se = FALSE, colour="red")

####### We use K-fold cross validation to determine the best number of cut points
K = 10
N = nrow(mtcars)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 7 #The maximum number of cut points
fold_error_rate <- matrix(0, nrow = M - 1, ncol = K)

for (i in 1:(M - 1))
{
  for (j in 1:K)
  {
    mtcars$cutpoints <- cut(mtcars$hp,i + 1)
    train_index = which(fold_index != j)
    mtcars_train = mtcars[train_index,]
    mtcars_test = mtcars[-train_index,]
    
    step_function_fit <- lm(mpg~cutpoints, data = mtcars_train)
    mpg_pred <- predict(step_function_fit, mtcars_test)
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
plot + xlab("Number of cut points") + ylab("Error rate")

## Plot the average RMSE values
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(2,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Number of cut points") + ylab("Error rate")






########################################################### Step functions for classification

setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")
heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

############# Remove missing values
heartdisease <- na.omit(heartdisease)

############ We use M = 3 cut points for logistic regression
M = 10
my_glm <- glm(Heart_disease~cut(diaBP,M), data = heartdisease, family = "binomial")
summary(my_glm)

logis_plot <- ggplot(heartdisease, aes(x=diaBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', formula= y~cut(x,M), se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

############ K-fold cross validation to determine the best number of cutpoints
K = 10
N = nrow(heartdisease)
fold_index = rep_len(1:K, N) #We repeat the sequence (1,2...,K) until reach size N
set.seed(1000)
fold_index = sample(fold_index, replace = FALSE) #We randomly permute the elements of fold_index

M = 20 #The maximum number of cut points
fold_error_rate <- matrix(0, nrow = M - 1, ncol = K)

for (i in 1:(M - 1))
{
  for (j in 1:K)
  {
    heartdisease$cutpoints <- cut(heartdisease$diaBP,i + 1)
    train_index = which(fold_index != j)
    heartdisease_train = heartdisease[train_index,]
    heartdisease_test = heartdisease[-train_index,]
    
    polyfit <- glm(Heart_disease~cutpoints, data = heartdisease_train, family = "binomial")
    heartdisease_pred <- predict(polyfit, heartdisease_test)
    yhat_predict_test <- ifelse(heartdisease_pred > 0.5, 1, 0)
    table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
    fold_error_rate[i,j] <- sum(diag(table_heart_test))/ sum(table_heart_test)
  }
}


### Plot the classification accuracy for K folds
new_fold_error_rate <- as.data.frame(fold_error_rate)
plot <- ggplot(data = new_fold_error_rate, aes(x = seq(2,M, by = 1)))
for (i in 1:K)
{
  plot <- plot + geom_line( aes_string( y = new_fold_error_rate[,i]), col = i+2)
}
plot + xlab("Number of cut points") + ylab("Classification accuracy")

## Plot the average classification accuracy
mean_fold_error_rate <- rowMeans(fold_error_rate)
mean_fold_error_rate <- as.data.frame(mean_fold_error_rate)
ggplot(mean_fold_error_rate, aes(x = seq(2,M, by = 1), y = mean_fold_error_rate)) + geom_line() + xlab("Number of cut points") + ylab("Classification accuracy")

