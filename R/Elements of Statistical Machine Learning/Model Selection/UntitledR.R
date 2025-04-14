library(ISLR)

weekly <- read.csv("Weekly.csv")
head(weekly)
weekly$Direction <- as.factor(weekly$Direction)

# a)
model_all <- glm(Direction ~ Lag1 + Lag2, data=weekly, family=binomial)
summary(model_all)

# b)
train_data_b <- weekly[-1, ]
test_data_b <- weekly[1, , drop=FALSE]
model_b <- glm(Direction ~ Lag1 + Lag2, data=train_data_b, family=binomial)
summary(model_b)

# c)
pred_prob_c <- predict(model_b, newdata=test_data_b, type="response")
pred_class_c <- ifelse(pred_prob_c > 0.5, "Up", "Down")

actual_class_c <- weekly$Direction[1]
correct_c <- pred_class_c == actual_class_c
correct_c

# d)
n <- nrow(weekly)
errors <- rep(NA, n)

for (i in 1:n) {
  train_data <- weekly[-i, ]
  test_data <- weekly[i, , drop=FALSE]
  
  model <- glm(Direction ~ Lag1 + Lag2, data=train_data, family=binomial)
  pred_prob <- predict(model, newdata=test_data, type="response")
  pred_class <- ifelse(pred_prob > 0.5, "Up", "Down")
  
  actual_class <- weekly$Direction[i]
  errors[i] <- ifelse(pred_class != actual_class, 1, 0)
}

# e)
loocv_error <- mean(errors)
loocv_error

###################################################################################################################
###################################################################################################################

set.seed(1)

n <- 1000
p <- 20

# a)
X <- matrix(rnorm(n * p), nrow=n, ncol=p)
beta <- c(1, 1.5, 0, 0, 2, rep(0, p-5))
epsilon <- rnorm(n)

Y <- X %*% beta + epsilon

colnames(X) <- paste0("X", 1:p)
data <- data.frame(Y, X)

# b)
train_indices <- sample(1:n, 100)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# c)
library(leaps)

regfit_full <- regsubsets(Y ~ ., data=train_data, nvmax=p)
summary_regfit <- summary(regfit_full)

train_mat <- model.matrix(Y ~ ., data=train_data)
test_mat <- model.matrix(Y ~ ., data=test_data)

train_mse <- rep(NA, p)
test_mse <- rep(NA, p)

for (i in 1:p) {
  coefi <- coef(regfit_full, id=i)
  
  train_pred <- train_mat[, names(coefi)] %*% coefi
  test_pred <- test_mat[, names(coefi)] %*% coefi
  
  train_mse[i] <- mean((train_data$Y - train_pred)^2)
  test_mse[i] <- mean((test_data$Y - test_pred)^2)
}

# d)
plot(1:p, test_mse, type="b", pch=19, 
     xlab="Model Size", ylab="Test MSE", main="Test MSE vs Model Size")

# e)
which.min(test_mse)
min(test_mse)

# f)
best_model_coefs <- coef(regfit_full, id=3)
best_model_coefs

# g)
rmse_coef <- rep(NA, p)

for (i in 1:p) {
  est_beta <- rep(0, p)
  
  coefi <- coef(regfit_full, id=i)
  coef_names <- names(coefi)[-1]
  
  for (name in coef_names) {
    j <- as.numeric(sub("X", "", name))
    est_beta[j] <- coefi[name]
  }
  
  rmse_coef[i] <- sqrt(sum((est_beta - beta)^2))
}

plot(1:p, rmse_coef, type="b", pch=19, 
     xlab="Model Size", ylab="RMSE of Coefficients", main="Coefficient Estimation Error vs Model Size")








