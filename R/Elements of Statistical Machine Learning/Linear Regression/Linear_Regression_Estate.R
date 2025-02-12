#### This file is for an illustration of linear regression with "Real_estate.csv" data

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)

################################################################################
############## We first try with the "agriculture_worldbank.csv"
realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

################################################################################ Simple linear regression

############## Linear regression fit with Y = real estate price, X = house age
ggplot(realestate, aes(x=house_age, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

# Coefficients from linear regression fit
lm(price~house_age, data=realestate) 

############## Linear regression fit with Y = real estate price, X = distance to station
ggplot(realestate, aes(x=dist_station, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

# Coefficients from linear regression fit
lm(price~dist_station, data=realestate) 

############# Linear regression fit with Y = real estate price, X = house age
ggplot(realestate, aes(x=number_stores, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, se = FALSE, colour="red")

# Coefficients from linear regression fit
lm(price~number_stores, data=realestate) 

################################################### Confidence intervals


######### Y = real estate price, X = house age
ggplot(realestate, aes(x=house_age, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, colour="red")

## Take a look at confidence intervals
my_lm <- lm(price~house_age, data=realestate) 
confint(my_lm) # This operator is used to construct the confidence interval for the coefficients in linear model

######### Y = real estate price, X = distance to station
ggplot(realestate, aes(x=dist_station, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, colour="red")

## Take a look at confidence intervals
my_lm <- lm(price~dist_station, data=realestate) 
confint(my_lm)

######### Y = real estate price, X = number of stores
ggplot(realestate, aes(x=number_stores, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~x, colour="red")

## Take a look at confidence intervals
my_lm <- lm(price~number_stores, data=realestate) 
confint(my_lm)


################################################### Hypothesis testing

######### Y = real estate price, X = house age

## Hypothesis testing
my_lm <- lm(price~house_age, data=realestate) 
summary(my_lm) # Summary of the linear model (including hypothesis testing)

######### Y = real estate price, X = distance to station

## Hypothesis testing
my_lm <- lm(price~dist_station, data=realestate) 
summary(my_lm) # Summary of the linear model (including hypothesis testing)

######### Y = real estate price, X = number of stores

## Hypothesis testing
my_lm <- lm(price~number_stores, data=realestate) 
summary(my_lm) # Summary of the linear model (including hypothesis testing)


################################################################################ Multiple linear regression

######### Y = real estate price, X = (house age, distance to station, number of stores)

lm(price~house_age + dist_station + number_stores, data=realestate) 

######### Confidence intervals for the cofficients

my_lm <- lm(price~house_age + dist_station + number_stores, data=realestate)
confint(my_lm)

######### Hypothesis testing
my_lm <- lm(price~house_age + dist_station + number_stores, data=realestate)
summary(my_lm)

################################################################################ Interactions of predictors

###### Y = price, X = (house age, distance to station)

my_lm <- lm(price~house_age + dist_station + house_age * dist_station, data=realestate)
summary(my_lm)

##### Y = price, X = (house age, number of stores)

my_lm <- lm(price~house_age + number_stores + house_age * number_stores, data=realestate)
summary(my_lm)

##### Y = price, X = (distance to station, number of stores)

my_lm <- lm(price~dist_station + number_stores + dist_station * number_stores, data=realestate)
summary(my_lm)

################################################################################ Non-linear relationships

###### Y = price, X = house age

## Quadratic form

my_lm <- lm(price~poly(house_age,2), data=realestate)
summary(my_lm)

# Visualization for quadratic form
ggplot(realestate, aes(x=house_age, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,2), se = FALSE, colour="red")

## Cubic form

my_lm <- lm(price~poly(house_age,3), data=realestate)
summary(my_lm)

# Visualization for cubic form
ggplot(realestate, aes(x=house_age, y=price)) + geom_point(size=2, color = "blue", shape=19) + geom_smooth(method='lm', formula= y~poly(x,3), se = FALSE, colour="red")

################################################################################ Residual plots

############### Y = real estate price, X = house age

## Linear model
my_lm <- lm(price~., data=realestate)
plot(my_lm)

## Quadratic model
my_lm <- lm(price~poly(house_age, 2), data=realestate)
plot(my_lm)

############### Y = real estate price, X = distance to station

## Linear model
my_lm <- lm(price~dist_station, data=realestate)
plot(my_lm)

## Quadratic model
my_lm <- lm(price~poly(dist_station, 2), data=realestate)
plot(my_lm)

## Cubic model
my_lm <- lm(price~poly(dist_station, 3), data=realestate)
plot(my_lm)

############### Y = real estate price, X = number of stores

## Linear model
my_lm <- lm(price~number_stores, data=realestate)
plot(my_lm)

## Quadratic model
my_lm <- lm(price~poly(number_stores,2), data=realestate)
plot(my_lm)

## Cubic model
my_lm <- lm(price~poly(number_stores,3), data=realestate)
plot(my_lm)



