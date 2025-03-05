#### This file is for an illustration of subset selection with "Real_estate.csv" data

library(ggplot2)
library(tidyr)
library(dplyr)

#Install library that is used for subset selection in R
#install.packages("leaps")
library(leaps) 

realestate = read.csv("Real_estate.csv", head = TRUE, check.names=FALSE)

realestate <- data.frame(realestate)
colnames(realestate) <- c("No","transact_date", "house_age", "dist_station", "number_stores", "latitude", "longitude", "price")
head(realestate, 6)

### We remove column "No"
realestate <- realestate[,-1]

############################################################# Best subset selection 

### We use "regsubsets" operator to perform best subset selection. It chooses the best set of variables for each model size
my_regsubset <- regsubsets(price~., realestate)
summary(my_regsubset)

### Take a look at the values of BIC, Adjusted R^2, AIC, C_p
my_summary <- summary(my_regsubset)
names(my_summary)

### Create plots with these values

# With C_p - smallest
ggplot(as.data.frame(my_summary$cp), aes(x = seq(1:6), y = my_summary$cp)) + geom_line(color = "red") + geom_point(size = 1) + xlab("Number of predictors") + ylab("C_p")

# With BIC - smallest
ggplot(as.data.frame(my_summary$bic), aes(x = seq(1:6), y = my_summary$bic)) + geom_line(color = "blue") + geom_point(size = 1) + xlab("Number of predictors") + ylab("BIC")

# With adjusted R^2 - largest
ggplot(as.data.frame(my_summary$adjr2), aes(x = seq(1:6), y = my_summary$adjr2)) + geom_line(color = "pink") + geom_point(size = 1) + xlab("Number of predictors") + ylab("Adjusted R2")

############################################################# Forward and backward stepwise selection

############################################## Forward stepwise selection
forward_selec <- regsubsets(price~., realestate, method = "forward")
summary(forward_selec)

my_summary <- summary(forward_selec)

### Create plots with these values

# With C_p - smallest
ggplot(as.data.frame(my_summary$cp), aes(x = seq(1:6), y = my_summary$cp)) + geom_line(color = "red") + geom_point(size = 1) + xlab("Number of predictors") + ylab("C_p")

# With BIC - smallest
ggplot(as.data.frame(my_summary$bic), aes(x = seq(1:6), y = my_summary$bic)) + geom_line(color = "blue") + geom_point(size = 1) + xlab("Number of predictors") + ylab("BIC")

# With adjusted R^2 - largest
ggplot(as.data.frame(my_summary$adjr2), aes(x = seq(1:6), y = my_summary$adjr2)) + geom_line(color = "pink") + geom_point(size = 1) + xlab("Number of predictors") + ylab("Adjusted R2")


############################################## Backward stepwise selection
backward_selec <- regsubsets(price~., realestate, method = "backward")
summary(backward_selec)

my_summary <- summary(backward_selec)

### Create plots with these values

# With C_p - smallest
ggplot(as.data.frame(my_summary$cp), aes(x = seq(1:6), y = my_summary$cp)) + geom_line(color = "red") + geom_point(size = 1) + xlab("Number of predictors") + ylab("C_p")

# With BIC - smallest
ggplot(as.data.frame(my_summary$bic), aes(x = seq(1:6), y = my_summary$bic)) + geom_line(color = "blue") + geom_point(size = 1) + xlab("Number of predictors") + ylab("BIC")

# With adjusted R^2 - largest
ggplot(as.data.frame(my_summary$adjr2), aes(x = seq(1:6), y = my_summary$adjr2)) + geom_line(color = "pink") + geom_point(size = 1) + xlab("Number of predictors") + ylab("Adjusted R2")










