#### This file is for an in-class exercise with "spam_train.csv" (for training) and "spam_test.csv" (for testing)

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)

################################################################################
############## We first try with the "Heart_disease.csv"
spamemail_train = read.csv("spam_train.csv", head = TRUE, check.names = FALSE)
spamemail_test = read.csv("spam_test.csv", head = TRUE, check.names = FALSE)

############## Change the column names for easier implementation
spamemail_train <- data.frame(spamemail_train)
head(spamemail_train, 6)

############################################# Logistic regression with all of the predictors
my_glm <- glm(y~ . , data = spamemail_train, family = "binomial")
summary(my_glm)

############################################# Classification accuracy on the test data
predicted_glm_test <- predict(my_glm, spamemail_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_spam_test <- table(y = spamemail_test$y, yhat = yhat_predict_test)
table_spam_test

## Classification accuracy from training data
accuracy_spam_test <- sum(diag(table_spam_test))/ sum(table_spam_test)
accuracy_spam_test

############################################# Logistic regression without "word.freq.meeting", "word.freq.re", and "char.freq.semicolon"
my_glm <- glm(y~ word.freq.remove + char.freq.exclamation + word.freq.order + capital.run.length.average + word.freq.free + word.freq.edu, data = spamemail_train, family = "binomial")
summary(my_glm)
############################################# Classification accuracy on the test data
predicted_glm_test <- predict(my_glm, spamemail_test, type = "response")
yhat_predict_test <- ifelse(predicted_glm_test > 0.5, 1, 0)

table_spam_test <- table(y = spamemail_test$y, yhat = yhat_predict_test)
table_spam_test

## Classification accuracy from training data
accuracy_spam_test <- sum(diag(table_spam_test))/ sum(table_spam_test)
accuracy_spam_test