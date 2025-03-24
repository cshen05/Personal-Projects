library(ggplot2)
library(dplyr)
library(caret)


auto_data <- read.csv("Auto.csv", stringsAsFactors = FALSE)

# Data cleaning
str(auto_data)
auto_data$horsepower <- as.numeric(gsub("\\?", NA, auto_data$horsepower))
auto_data <- na.omit(auto_data)

# a):
median_mpg <- median(auto_data$mpg)
auto_data$mpg01 <- ifelse(auto_data$mpg > median_mpg, 1, 0)

# b):
summary(auto_data)

# Scatterplots for continuous variables vs. mpg01
pairs(auto_data[, c("mpg", "displacement", "horsepower", "weight", "acceleration")], 
      col = auto_data$mpg01 + 1)

# Boxplots to compare distributions
boxplot(auto_data$displacement ~ auto_data$mpg01, main = "Displacement vs mpg01", xlab = "mpg01", ylab = "Displacement")
boxplot(auto_data$horsepower ~ auto_data$mpg01, main = "Horsepower vs mpg01", xlab = "mpg01", ylab = "Horsepower")
boxplot(auto_data$weight ~ auto_data$mpg01, main = "Weight vs mpg01", xlab = "mpg01", ylab = "Weight")

# c):
set.seed(42)
train_index <- createDataPartition(auto_data$mpg01, p = 0.7, list = FALSE)
train_data <- auto_data[train_index, ]
test_data <- auto_data[-train_index, ]

# Verify the split
table(train_data$mpg01)
table(test_data$mpg01)

library(tidyr)
library(FNN)

# f): Logistic Regression
# Train model
logistic_model <- glm(mpg01 ~ weight + horsepower + displacement, 
                      data = train_data, 
                      family = binomial)

# Predict on test data
logistic_probs <- predict(logistic_model, test_data, type = "response")
logistic_pred <- ifelse(logistic_probs > 0.5, 1, 0)

# Test error
logistic_test_error <- mean(logistic_pred != test_data$mpg01)
print(paste("Logistic Regression Test Error:", logistic_test_error))

# h): KNN Classification
# Standardize the predictor variables

# Test errors for different K
k_values <- c(1, 3, 5, 7, 9, 11, 15)
knn_test_errors <- numeric(length(k_values))

train_target <- as.factor(train_data$mpg01)
test_target <- as.factor(test_data$mpg01)

for (i in 1:length(k_values)) {
  k <- k_values[i]
  knn_pred <- knn(train_data, test_data, train_target, k = k)
  knn_test_errors[i] <- mean(knn_pred != test_target)
  print(paste("K =", k, "Test Error:", knn_test_errors[i]))
}

# Best K
best_k <- k_values[which.min(knn_test_errors)]
print(paste("Best K:", best_k, "with Test Error:", min(knn_test_errors)))
