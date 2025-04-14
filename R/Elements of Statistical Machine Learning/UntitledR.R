library(ggplot2)

# Load data
boston <- read.csv("Boston.csv")

#1)
fit_cubic <- lm(nox ~ poly(dis, 3), data = boston)
summary(fit_cubic)

ggplot(boston, aes(x = dis, y = nox)) +
  geom_point(alpha = 0.5) +
  stat_smooth(method = "lm", formula = y ~ poly(x, 3), color = "blue") +
  ggtitle("Cubic Polynomial Fit of NOX ~ DIS")

rss_values <- numeric(10)

for (d in 1:10) {
  fit <- lm(nox ~ poly(dis, d), data = boston)
  rss_values[d] <- sum(residuals(fit)^2)
}

plot(1:10, rss_values, type = "b", xlab = "Degree of Polynomial", ylab = "RSS",
     main = "RSS for Polynomial Fits of Various Degrees")

rss_df <- data.frame(
  Degree = 1:10,
  RSS = round(rss_values, 4)
)

print(rss_df)

library(boot)

cv_errors <- numeric(10)
for (d in 1:10) {
  fit <- glm(nox ~ poly(dis, d), data = boston)
  cv_errors[d] <- cv.glm(boston, fit, K = 10)$delta[1]
}

# Plot CV Error
plot(1:10, cv_errors, type = "b", xlab = "Degree", ylab = "CV Error",
     main = "10-fold CV Error for Polynomial Degrees")

library(splines)

# Regression spline with 4 df
fit_spline_4df <- lm(nox ~ bs(dis, df = 4), data = boston)
summary(fit_spline_4df)

# Plot
dis_range <- seq(min(boston$dis), max(boston$dis), length.out = 100)
pred_df <- data.frame(dis = dis_range)
pred_df$nox <- predict(fit_spline_4df, newdata = pred_df)

ggplot(boston, aes(x = dis, y = nox)) +
  geom_point(alpha = 0.5) +
  geom_line(data = pred_df, aes(x = dis, y = nox), color = "red") +
  ggtitle("Regression Spline with 4 Degrees of Freedom")

rss_spline <- numeric(10)

for (df in 3:10) {
  fit <- lm(nox ~ bs(dis, df = df), data = boston)
  rss_spline[df] <- sum(residuals(fit)^2)
}

# Plot RSS for splines
plot(3:10, rss_spline[3:10], type = "b", xlab = "Degrees of Freedom", ylab = "RSS",
     main = "RSS for Regression Splines")

rss_splines <- data.frame(
  Degree = 1:10,
  RSS = round(rss_spline, 4)
)

print(rss_splines)

cv_spline <- numeric(10)

for (df in 3:10) {
  fit <- glm(nox ~ bs(dis, df = df), data = boston)
  cv_spline[df] <- cv.glm(boston, fit, K = 10)$delta[1]
}

# Plot CV Error
plot(3:10, cv_spline[3:10], type = "b", xlab = "Degrees of Freedom", ylab = "CV Error",
     main = "10-fold CV Error for Regression Splines")

###########################################################################################################
#3)

library(ISLR2)
library(tree)
library(caret)

# Load data
carseats <- read.csv("Carseats.csv")

# Set seed and split
set.seed(42)
train_index <- createDataPartition(carseats$Sales, p = 0.7, list = FALSE)
train_data <- carseats[train_index, ]
test_data <- carseats[-train_index, ]

# Fit regression tree
tree_model <- tree(Sales ~ ., data = train_data)
summary(tree_model)

# Plot tree
plot(tree_model)
text(tree_model, pretty = 0)

# Predict on test set and calculate MSE
pred_tree <- predict(tree_model, newdata = test_data)
mse_tree <- mean((pred_tree - test_data$Sales)^2)
print(mse_tree)

# Cross-validation
cv_model <- cv.tree(tree_model)
plot(cv_model$size, cv_model$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

# Prune the tree (choose size with lowest deviance)
pruned_tree <- prune.tree(tree_model, best = which.min(cv_model$dev))
plot(pruned_tree)
text(pruned_tree, pretty = 0)

# Predict with pruned tree
pred_pruned <- predict(pruned_tree, newdata = test_data)
mse_pruned <- mean((pred_pruned - test_data$Sales)^2)
print(mse_pruned)

library(randomForest)

# Bagging: set mtry = total number of predictors
set.seed(42)
bag_model <- randomForest(Sales ~ ., data = train_data, mtry = ncol(train_data) - 1, importance = TRUE)

# Test MSE
pred_bag <- predict(bag_model, newdata = test_data)
mse_bag <- mean((pred_bag - test_data$Sales)^2)
print(mse_bag)

# Variable importance
importance(bag_model)
varImpPlot(bag_model)

# Random forest: mtry = sqrt(p) ~ 3 or 4
set.seed(42)
rf_model <- randomForest(Sales ~ ., data = train_data, mtry = 4, importance = TRUE)

# Test MSE
pred_rf <- predict(rf_model, newdata = test_data)
mse_rf <- mean((pred_rf - test_data$Sales)^2)
print(mse_rf)

# Importance
importance(rf_model)
varImpPlot(rf_model)

library(BART)

# Prepare X and y
x_train <- train_data[, names(train_data) != "Sales"]
y_train <- train_data$Sales
x_test <- test_data[, names(test_data) != "Sales"]

# Convert factors to numeric for BART
x_train <- data.frame(lapply(x_train, function(x) if(is.factor(x)) as.numeric(as.factor(x)) else x))
x_test <- data.frame(lapply(x_test, function(x) if(is.factor(x)) as.numeric(as.factor(x)) else x))

# Fit BART model
set.seed(42)# Convert factors and characters to numeric
x_train <- data.frame(lapply(train_data[, names(train_data) != "Sales"], function(x) {
  if (is.character(x) || is.factor(x)) as.numeric(as.factor(x)) else x
}))

x_test <- data.frame(lapply(test_data[, names(test_data) != "Sales"], function(x) {
  if (is.character(x) || is.factor(x)) as.numeric(as.factor(x)) else x
}))

y_train <- train_data$Sales

bart_model <- wbart(x.train = x_train, y.train = y_train, x.test = x_test)

# Test predictions and MSE
pred_bart <- bart_model$yhat.test.mean
mse_bart <- mean((pred_bart - test_data$Sales)^2)
print(mse_bart)

###############################################################################################
#7)
caravan <- read.csv("Caravan.csv")

train_data <- caravan[1:1000, ]
test_data <- caravan[-(1:1000), ]

library(gbm)

# Convert response to binary (1 = Yes, 0 = No)
train_data$Purchase <- ifelse(train_data$Purchase == "Yes", 1, 0)
test_data$Purchase <- ifelse(test_data$Purchase == "Yes", 1, 0)

# Fit boosting model
set.seed(42)
boost_model <- gbm(Purchase ~ ., data = train_data, distribution = "bernoulli",
                   n.trees = 1000, shrinkage = 0.01, interaction.depth = 4)

# Variable importance
summary(boost_model)

# Predict probabilities on test set
boost_probs <- predict(boost_model, newdata = test_data, n.trees = 1000, type = "response")

# Predict 'Yes' if prob > 0.20
boost_pred <- ifelse(boost_probs > 0.2, 1, 0)

# Confusion matrix
table(Predicted = boost_pred, Actual = test_data$Purchase)
# Fraction of predicted "Yes" that are actually "Yes"
true_positives <- sum(boost_pred == 1 & test_data$Purchase == 1)
total_predicted_yes <- sum(boost_pred == 1)
fraction_correct <- true_positives / total_predicted_yes
print(fraction_correct)

# Fit logistic regression on training set
glm_model <- glm(Purchase ~ ., data = train_data, family = binomial)

# Predict probabilities on test set
glm_probs <- predict(glm_model, newdata = test_data, type = "response")

# Predict "Yes" if probability > 0.2
glm_pred <- ifelse(glm_probs > 0.2, 1, 0)

# Confusion matrix
table(Predicted = glm_pred, Actual = test_data$Purchase)

# Evaluate performance
true_pos_glm <- sum(glm_pred == 1 & test_data$Purchase == 1)
total_pred_glm <- sum(glm_pred == 1)
fraction_glm <- true_pos_glm / total_pred_glm
print(fraction_glm)
library(class)

# Standardize predictors
train_X <- scale(train_data[, -ncol(train_data)])
test_X <- scale(test_data[, -ncol(test_data)],
                center = attr(train_X, "scaled:center"),
                scale = attr(train_X, "scaled:scale"))

# Convert Purchase to factor for KNN
train_Y <- as.factor(ifelse(train_data$Purchase == 1, "Yes", "No"))
test_Y <- as.factor(ifelse(test_data$Purchase == 1, "Yes", "No"))

# Remove rows with missing values from both train and test
train_complete <- complete.cases(train_X)
test_complete <- complete.cases(test_X)

train_X <- train_X[train_complete, ]
train_Y <- train_Y[train_complete]

test_X <- test_X[test_complete, ]
test_Y <- test_Y[test_complete]


# Run KNN (try k = 5)
set.seed(42)
knn_pred <- knn(train = train_X, test = test_X, cl = train_Y, k = 5, prob = TRUE)

# Get predicted probabilities from attributes
knn_probs <- attr(knn_pred, "prob")
knn_probs <- ifelse(knn_pred == "Yes", knn_probs, 1 - knn_probs)

# Apply 20% threshold manually
knn_binary_pred <- ifelse(knn_probs > 0.2, 1, 0)

# Confusion matrix
table(Predicted = knn_binary_pred, Actual = test_data$Purchase)

# Evaluate performance
true_pos_knn <- sum(knn_binary_pred == 1 & test_data$Purchase == 1)
total_pred_knn <- sum(knn_binary_pred == 1)
fraction_knn <- true_pos_knn / total_pred_knn
print(fraction_knn)

