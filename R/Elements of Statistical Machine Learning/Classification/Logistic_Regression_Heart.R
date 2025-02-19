#### This file is for an illustration of logistic regression with "Heart_disease.csv" data

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)

# Set the working directory (you need to choose different directory in your computer)
setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

################################################################################
############## We first try with the "Heart_disease.csv"
heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

############## Create training and test sets

### 75% of the sample size
sample_size <- floor(0.75 * nrow(heartdisease))

## We create the seed to make our partition reproducible
set.seed(1000)
train_index <- sample(seq_len(nrow(heartdisease)), size = sample_size)
heartdisease_train <- heartdisease[train_index,]
heartdisease_test <- heartdisease[-train_index,]


################################################################# Simple linear probability model

##################### Y = heart disease, X = systolic blood pressure (sysBP)
my_lm <- lm(Heart_disease~sysBP, data = heartdisease_train)
summary(my_lm)

## Visualization of the simple linear probability model
ggplot(heartdisease_train, aes(x=sysBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

##### Compute classification accuracy with training data from simple probability model
predicted_lm_train <- predict(my_lm, heartdisease_train)
yhat_predict_train <- ifelse(predicted_lm_train > 0.5, 1, 0) # We compare the predicted values of heart disease to 0.5 and output 0 if they less then 0.5 and 1 if they are larger than 0.5

## Create a table
table_heart_train <- table(y = heartdisease_train$Heart_disease, yhat = yhat_predict_train)
table_heart_train

## Classification accuracy from training data
accuracy_heart_train <- sum(diag(table_heart_train))/ sum(table_heart_train)
accuracy_heart_train

#### Compute classification accuracy with test data
predicted_lm_test <- predict(my_lm, heartdisease_test)
yhat_predict_test <- ifelse(predicted_lm_test > 0.5, 1, 0)

table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
table_heart_test

## Classification accuracy from test data
accuracy_heart_test <- sum(diag(table_heart_test))/ sum(table_heart_test)
accuracy_heart_test

################################################################# Simple logistic regression model

###################### Y = heart disease, X = systolic blood pressure (sysBP)

### Glm = generalized linear model, "family = binomial" means that we consider logistic regression with binary values of Y
my_glm <- glm(Heart_disease~sysBP, data = heartdisease_train, family = "binomial")
summary(my_glm)

# Visualization within the range of data
logis_plot <- ggplot(heartdisease_train, aes(x=sysBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

# Visualization outside the range of data
logis_plot <- ggplot(heartdisease_train, aes(x=sysBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial")) + xlim(0, 350)

################### Compute the classification accuracy from training data
predicted_glm_train <- predict(my_glm, heartdisease_train, type = "response")
yhat_predict_train <- ifelse(predicted_glm_train > 0.5, 1, 0) # We compare the predicted values of heart disease to 0.5 and output 0 if they less then 0.5 and 1 if they are larger than 0.5

## Create a table
table_heart_train <- table(y = heartdisease_train$Heart_disease, yhat = yhat_predict_train)
table_heart_train

## Classification accuracy
accuracy_heart_train <- sum(diag(table_heart_train))/ sum(table_heart_train)
accuracy_heart_train

#################### Compute the classification accuracy from test data
predicted_glm_test <- predict(my_glm, heartdisease_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
table_heart_test

## Classification accuracy from test data
accuracy_heart_test <- sum(diag(table_heart_test))/ sum(table_heart_test)
accuracy_heart_test

###################### Y = heart disease, X = total Cholesterol (totChol)
my_glm <- glm(Heart_disease~totChol, data = heartdisease_train, family = "binomial")
summary(my_glm)

# Visualization within the range of data
logis_plot <- ggplot(heartdisease_train, aes(x=totChol, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

# Visualization outside the range of data
logis_plot <- ggplot(heartdisease_train, aes(x=totChol, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', se = FALSE, colour="red", fullrange=TRUE, method.args = list(family= "binomial")) + xlim(0, 350)

################### Compute the classification accuracy from training data
predicted_glm_train <- predict(my_glm, heartdisease_train, type = "response")
yhat_predict_train <- ifelse(predicted_glm_train > 0.5, 1, 0) # We compare the predicted values of heart disease to 0.5 and output 0 if they less then 0.5 and 1 if they are larger than 0.5

## Create a table
table_heart_train <- table(y = heartdisease_train$Heart_disease, yhat = yhat_predict_train)
table_heart_train

## Classification accuracy
accuracy_heart_train <- sum(diag(table_heart_train))/ sum(table_heart_train)
accuracy_heart_train

#################### Compute the classification accuracy from test data
predicted_glm_test <- predict(my_glm, heartdisease_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
table_heart_test

## Classification accuracy from test data
accuracy_heart_test <- sum(diag(table_heart_test))/ sum(table_heart_test)
accuracy_heart_test

###################### Confidence intervals 
my_glm <- glm(Heart_disease~sysBP, data = heartdisease_train, family = "binomial")
confint(my_glm)

## Visualization of confidence intervals with logistic regression
logis_plot <- ggplot(heartdisease_train, aes(x=sysBP, y=Heart_disease)) + geom_point(size=2, color = "blue", shape=19) 
logis_plot + geom_smooth(method='glm', se = TRUE, colour="red", fullrange=TRUE, method.args = list(family= "binomial"))

############### Remove totChol, BMI, and heartRate from the predictors
my_glm <- glm(Heart_disease~ age + cigsPerDay  + sysBP + glucose, data = heartdisease_train, family = "binomial")
summary(my_glm)

#################### Compute the classification accuracy from test data
predicted_glm_test <- predict(my_glm, heartdisease_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
table_heart_test

## Classification accuracy from training data
accuracy_heart_test <- sum(diag(table_heart_test))/ sum(table_heart_test)
accuracy_heart_test

my_glm <- glm(Heart_disease~ age + cigsPerDay + totChol + sysBP + BMI + heartRate + glucose, data = heartdisease_train, family = "binomial")
summary(my_glm)

#################### Compute the classification accuracy from test data
predicted_glm_test <- predict(my_glm, heartdisease_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_heart_test <- table(y = heartdisease_test$Heart_disease, yhat = yhat_predict_test)
table_heart_test

## Classification accuracy from training data
accuracy_heart_test <- sum(diag(table_heart_test))/ sum(table_heart_test)
accuracy_heart_test

############################################################# Multiple logistic regression model with qualitative predictors
my_glm <- glm(Heart_disease~ age + diabetes, data = heartdisease_train, family = "binomial")
summary(my_glm)





















