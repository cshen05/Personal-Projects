#### This file is for an illustration of generalized additive models (GAMs)

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics) #For RMSE calculation
library(gam) #For GAMs

############################# Regression GAMs

######## We first consider the setting when both functions of hp and wt are linear
my_gam <- gam(mpg~ hp + wt, data = mtcars)
summary(my_gam)

par(mfrow = c(1,2))
plot(my_gam, se = TRUE) #This creates the plot for model my_gam

####### We then consider the setting when the function of hp is quadratic and the function of wt is linear
my_gam_1 <- gam(mpg~poly(hp, 2) + wt, data = mtcars)
summary(my_gam_1)

par(mfrow = c(1,2))
plot(my_gam_1, se = TRUE)  #This creates the plot for model my_gam_1

#### To determine which model is better, we can use Anova in R
### Comparing my_gam versus my_gam_1
anova(my_gam, my_gam_1)

####### We then consider the setting when the function of hp is linear and the function of wt is quadratic

my_gam_2 <- gam(mpg~ poly(hp, 2) + poly(wt,2), data = mtcars)
summary(my_gam_2)

par(mfrow = c(1,2))
plot(my_gam_2, se = TRUE)  #This creates the plot for model my_gam_2

### Comparing my_gam versus my_gam_2
anova(my_gam_1, my_gam_2)

######################### We then consider the setting when both the functions of hp and wt are splines
my_gam_3 <- gam(mpg~bs(hp, 5, degree = 2) + bs(wt, 5, degree = 2), data = mtcars)
summary(my_gam_3)

## We plot the functions for hp and wt
par(mfrow = c(1,2))
plot(my_gam_3, se = TRUE)

anova(my_gam_2,my_gam_3)

#### Prediction error from the best model
mymodel_pred <- predict(my_gam_2, mtcars)
error <- RMSE(mymodel_pred, mtcars$mpg)

############################################# Classification GAMs with the heart disease dataset

setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

############# Remove missing values
heartdisease <- na.omit(heartdisease)

############################## GAMs with Y = heart disease, X = (diaBP, glucose)

############### First, we consider standard linear logistic regression
my_gam <- gam(Heart_disease~diaBP + glucose, family = binomial, data = heartdisease)
summary(my_gam)

## Then, we consider the quadratic function for diaBP
my_gam_1 <- gam(Heart_disease~poly(diaBP, 2) + glucose, family = binomial, data = heartdisease)
summary(my_gam_1)

par(mfrow = c(1,2))
plot(my_gam_1, se = TRUE)

## Finally, we consider the quadratic functions for both diaBP and glucose
my_gam_2 <- gam(Heart_disease~ poly(diaBP, 2) + poly(glucose,2), family = binomial, data = heartdisease)
summary(my_gam_2)

anova(my_gam, my_gam_1, my_gam_2)

######## Compute classification accuracy
heartdisease_pred <- predict(my_gam_1, heartdisease)
yhat_predict <- ifelse(heartdisease_pred > 0.5, 1, 0)
table_heart <- table(y = heartdisease$Heart_disease, yhat = yhat_predict)
accuracy <- sum(diag(table_heart))/ sum(table_heart)