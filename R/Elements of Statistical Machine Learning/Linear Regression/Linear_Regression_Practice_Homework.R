#### This file is for an illustration of linear regression with "Real_estate.csv" data

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)
library(ISLR2)

### Load the Boston data. The Boston data is built-in data in the library "ISLR2".
Boston
head(Boston)

############ Part (a)

### Crim versus Zn
ggplot(Boston, aes(x= zn, y=crim)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

# Coefficients from linear regression fit
Zn_model <- lm(crim~medv, data=Boston)
summary(Zn_model)

### Crim versus Age
ggplot(Boston, aes(x= age, y=crim)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

# Coefficients from linear regression fit
lm(crim~age, data=Boston)

############ Part (b) --- Crim versus all predictors


# Coefficients from linear regression fit
crim_model <- lm(crim~., data=Boston)
summary(crim_model)

############ Part (c) --- Redisual plots


#### Crim_versus_Zn

# Linear model
Crim_versus_Zn <- lm(crim~zn, data=Boston)
plot(Crim_versus_Zn)

# Cubic model
Crim_versus_Zn <- lm(crim~poly(zn,3), data=Boston)
plot(Crim_versus_Zn)

