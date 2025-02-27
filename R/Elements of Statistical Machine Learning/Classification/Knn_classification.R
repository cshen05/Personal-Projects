#### This file is for an illustration of K-nearest neighbor with "Heart_disease.csv" data

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)
library(FNN)


# Set the working directory (you need to choose different directory in your computer)
setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

################################################################################
############## We first try with the "Heart_disease.csv"
heartdisease = read.csv("Heart_disease.csv", head = TRUE, check.names=FALSE)

############## Change the column names for easier implementation
heartdisease <- data.frame(heartdisease)
colnames(heartdisease)[16] <- c("Heart_disease") # Change the column name of Y
head(heartdisease, 6)

### Remove missing data from data
heartdisease <- na.omit(heartdisease)

############## Create training and test sets

### 75% of the sample size
sample_size <- floor(0.75 * nrow(heartdisease))

## We create the seed to make our partition reproducible
set.seed(1000)
train_index <- sample(seq_len(nrow(heartdisease)), size = sample_size)

## We create separate training and test sets for predictors and labels
heartlabel_train <- heartdisease[train_index,16]
heartlabel_test <- heartdisease[-train_index,16]

heartdisease_train <- heartdisease[train_index,-16]
heartdisease_test <- heartdisease[-train_index,-16]

## We perform K-nearest neighbor

# We first try with K = 1
my_knn <- knn(heartdisease_train, heartdisease_test, heartlabel_train, k = 1)
table_knn <- table(my_knn, heartlabel_test)

accuracy_heart_test <- sum(diag(table_knn))/ sum(table_knn)
accuracy_heart_test

# We plot the accuracy for different values of K
M = 150 # the number of possible values of K
K <- seq(1, M)

accuracy_heart_test <- rep(0,M)

for (i in 1:M)
{
  my_knn <- knn(heartdisease_train, heartdisease_test, heartlabel_train, k = K[i])
  table_knn <- table(my_knn, heartlabel_test)
  
  accuracy_heart_test[i] <- sum(diag(table_knn))/ sum(table_knn)
}
max(accuracy_heart_test)

accuracy_heart_test <- as.data.frame(accuracy_heart_test)
ggplot(accuracy_heart_test, aes(x = K, y = accuracy_heart_test)) + geom_line(col = "red")


