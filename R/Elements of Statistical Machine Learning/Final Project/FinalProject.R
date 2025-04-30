library(dplyr)
library(tidyr)
library(caret)
library(randomForest)
library(ggplot2)
library(rpart)
library(xgboost)

df <- read.csv("TelcoCustomerChurn.csv", stringsAsFactors = FALSE)

# -------- CLEANING --------
# Drop customerID
df <- dplyr::select(df, -customerID)

# Convert empty strings to NA and drop incomplete rows
df[df == ""] <- NA
df$TotalCharges <- as.numeric(df$TotalCharges)
df <- na.omit(df)

# Convert 'Churn' to binary target variable
df$Churn <- ifelse(df$Churn == "Yes", 1, 0)

# Convert categorical variables to factors
cat_vars <- c(
  "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
  "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
  "Contract", "PaperlessBilling", "PaymentMethod"
)
df[cat_vars] <- lapply(df[cat_vars], as.factor)

# -------- SPLIT DATA  --------
set.seed(42)
train_index <- createDataPartition(df$Churn, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]

# -------- ENCODING  --------
dummy_model <- dummyVars(Churn~ ., data = train, fullRank = TRUE)

train_encoded <- predict(dummy_model, newdata = train) %>% as.data.frame()
test_encoded  <- predict(dummy_model, newdata = test) %>% as.data.frame()

train_encoded$Churn <- train$Churn
test_encoded$Churn  <- test$Churn

# -------- SCALE NUMERICAL COLUMNS --------
num_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
train_encoded[num_cols] <- scale(train_encoded[num_cols])
test_encoded[num_cols]  <- scale(test_encoded[num_cols])

################################################################################## Q1
# remove tenure and total charges as they do not cause churn
train_encoded_filtered <- train_encoded %>% dplyr::select(-tenure, -TotalCharges, -`Contract.One year`, -`Contract.Two year`, -MonthlyCharges)
test_encoded_filtered  <- test_encoded %>% dplyr::select(-tenure, -TotalCharges, -`Contract.One year`, -`Contract.Two year`, -MonthlyCharges)

x_train <- train_encoded_filtered %>% dplyr::select(-Churn)
y_train <- as.factor(train_encoded_filtered$Churn)
x_test  <- test_encoded_filtered %>% dplyr::select(-Churn)
y_test  <- as.factor(test_encoded_filtered$Churn)

set.seed(42)
mtry_values <- seq(1, ncol(x_train), by = 2)
test_errors_rf <- numeric(length(mtry_values))

for (i in seq_along(mtry_values)) {
  rf_model <- randomForest(
    x = x_train,
    y = y_train,
    mtry = mtry_values[i],
    ntree = 300
  )
  
  rf_preds <- predict(rf_model, newdata = x_test)
  test_errors_rf[i] <- mean(rf_preds != y_test)
}

# Plot mtry tuning curve
error_rf_df <- data.frame(mtry = mtry_values, ErrorRate = test_errors_rf)

ggplot(error_rf_df, aes(x = mtry, y = ErrorRate)) +
  geom_line(color = "blue", size = 1.2) +
  labs(
    title = "Random Forest: Test Error vs mtry",
    x = "mtry (predictors per split)",
    y = "Test Error Rate"
  ) +
  theme_minimal()

# Best Random Forest model
best_mtry <- mtry_values[which.min(test_errors_rf)]

final_rf <- randomForest(
  x = x_train,
  y = y_train,
  mtry = best_mtry,
  ntree = 500,
  importance = TRUE
)

rf_final_preds <- predict(final_rf, newdata = x_test)
rf_final_error <- mean(rf_final_preds != y_test)

# =======================================
# ---- Decision Tree Model ----
# =======================================
# Tune over cp (complexity parameter)
set.seed(42)
cp_values <- seq(0.001, 0.05, by = 0.005)
test_errors_tree <- numeric(length(cp_values))

for (i in seq_along(cp_values)) {
  tree_model <- rpart(Churn ~ ., data = train_encoded_filtered, method = "class", cp = cp_values[i])
  tree_preds <- predict(tree_model, newdata = test_encoded_filtered, type = "class")
  test_errors_tree[i] <- mean(tree_preds != y_test)
}

# Plot cp tuning curve
error_tree_df <- data.frame(cp = cp_values, ErrorRate = test_errors_tree)

ggplot(error_tree_df, aes(x = cp, y = ErrorRate)) +
  geom_line(color = "darkgreen", size = 1.2) +
  labs(
    title = "Decision Tree: Test Error vs cp",
    x = "cp (complexity parameter)",
    y = "Test Error Rate"
  ) +
  theme_minimal()

# Best Decision Tree model
best_cp <- cp_values[which.min(test_errors_tree)]

final_tree <- rpart(Churn ~ ., data = train_encoded_filtered, method = "class", cp = best_cp)

tree_final_preds <- predict(final_tree, newdata = test_encoded_filtered, type = "class")
tree_final_error <- mean(tree_final_preds != y_test)

# =======================================
# ---- XGBoost Model ----
# =======================================
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = as.numeric(as.character(y_train)))
dtest  <- xgb.DMatrix(data = as.matrix(x_test), label = as.numeric(as.character(y_test)))

nrounds_values <- seq(50, 300, by = 50)
test_errors_xgb <- numeric(length(nrounds_values))

for (i in seq_along(nrounds_values)) {
  xgb_model <- xgboost(
    data = dtrain,
    objective = "binary:logistic",
    nrounds = nrounds_values[i],
    max_depth = 6,
    eta = 0.1,
    verbose = 0
  )
  xgb_preds_prob <- predict(xgb_model, newdata = dtest)
  xgb_preds <- ifelse(xgb_preds_prob > 0.5, 1, 0)
  test_errors_xgb[i] <- mean(xgb_preds != as.numeric(as.character(y_test)))
}

# Plot nrounds tuning curve
error_xgb_df <- data.frame(nrounds = nrounds_values, ErrorRate = test_errors_xgb)

ggplot(error_xgb_df, aes(x = nrounds, y = ErrorRate)) +
  geom_line(color = "red", size = 1.2) +
  labs(
    title = "XGBoost: Test Error vs nrounds",
    x = "nrounds (number of boosting rounds)",
    y = "Test Error Rate"
  ) +
  theme_minimal()

# Best XGBoost model
best_nrounds <- nrounds_values[which.min(test_errors_xgb)]

final_xgb <- xgboost(
  data = dtrain,
  objective = "binary:logistic",
  nrounds = best_nrounds,
  max_depth = 6,
  eta = 0.1,
  verbose = 0
)

xgb_preds_prob <- predict(final_xgb, newdata = dtest)
xgb_preds <- ifelse(xgb_preds_prob > 0.5, 1, 0)
xgb_final_error <- mean(xgb_preds != as.numeric(as.character(y_test)))

# =======================================
# ---- Final Model Comparison ----
# =======================================

comparison_table <- data.frame(
  Model = c("Random Forest", "Decision Tree", "XGBoost"),
  Test_Error_Rate = c(rf_final_error, tree_final_error, xgb_final_error)
)

print(comparison_table)

# ---- Plot all models side by side ----
ggplot(comparison_table, aes(x = Model, y = Test_Error_Rate, fill = Model)) +
  geom_col() +
  labs(
    title = "Final Model Comparison: Test Error Rates",
    x = "Model",
    y = "Test Error Rate"
  ) +
  theme_minimal() +
  geom_text(aes(label = round(Test_Error_Rate, 3)), vjust = -0.5)

# =======================================
# ---- Pick Best Model and Show Feature Importance ----
# =======================================

best_model_index <- which.min(comparison_table$Test_Error_Rate)
best_model_name <- comparison_table$Model[best_model_index]

cat("Best Model:", best_model_name, "\n")

if (best_model_name == "Random Forest") {
  varImpPlot(final_rf)
  
} else if (best_model_name == "Decision Tree") {
  importance_vals <- final_tree$variable.importance
  importance_df <- data.frame(Feature = names(importance_vals), Importance = importance_vals)
  importance_df <- importance_df[order(-importance_df$Importance), ]
  print(importance_df)
  
} else if (best_model_name == "XGBoost") {
  xgb.importance(feature_names = colnames(x_train), model = final_xgb) %>%
    xgb.plot.importance(top_n = 10)
}
################################################################################## Q2
library(rpart)
library(rpart.plot)
library(janitor)

# Convert relevant behavioral vars to factors
behavior_vars <- c(
  "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
  "StreamingTV", "StreamingMovies", "MultipleLines", "InternetService",
  "PaperlessBilling", "PaymentMethod"
)
df[behavior_vars] <- lapply(df[behavior_vars], as.factor)

# -------- Build a decision tree with behavioral features --------
df_behavior <- df[, c(behavior_vars, "Churn")]

# Convert Churn to factor
df_behavior$Churn <- as.factor(df_behavior$Churn)

# Set up 10-fold cross-validation
set.seed(42)
cv_control <- trainControl(method = "cv", number = 10)

# Train tree using caret with CV
cv_tree <- train(
  Churn ~ .,
  data = df_behavior,
  method = "rpart",
  trControl = cv_control,
  tuneLength = 10
)

# Print results
print(cv_tree)

# Extract cross-validated error rate
best_accuracy <- max(cv_tree$results$Accuracy)
cv_error <- 1 - best_accuracy
print(paste("Cross-validated error rate:", round(cv_error, 4)))


# Plot the tree
rpart.plot(
  cv_tree$finalModel,
  type = 2,
  extra = 104,
  box.palette = "RdBu",
  fallen.leaves = TRUE,
  main = "Churn Risk Based on Behavioral Features"
)

log_model <- glm(Churn ~ OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport +
                   StreamingTV + StreamingMovies + MultipleLines + InternetService +
                   PaperlessBilling + PaymentMethod,
                 data = df, family = "binomial")

summary(log_model)

################################################################################## Q3
# -------- Filter to churners --------
churners <- df %>% filter(Churn == 1)

# -------- Create revenue tiers using TotalCharges --------
churners <- churners %>%
  mutate(ValueTier = case_when(
    TotalCharges >= quantile(TotalCharges, 0.75) ~ "High",
    TotalCharges >= quantile(TotalCharges, 0.25) ~ "Medium",
    TRUE ~ "Low"
  )) %>%
  mutate(ValueTier = factor(ValueTier, levels = c("Low", "Medium", "High")))

# -------- Explore numeric feature differences --------
# Average tenure and charges by value tier
aggregate(cbind(tenure, MonthlyCharges, TotalCharges) ~ ValueTier, data = churners, mean)

# -------- Explore categorical patterns --------
# Contract types by tier
table(churners$ValueTier, churners$Contract)

# Payment methods by tier
table(churners$ValueTier, churners$PaymentMethod)

# Tech support usage by tier
table(churners$ValueTier, churners$TechSupport)

# -------- Visualizations --------

# Boxplot of Monthly Charges
ggplot(churners, aes(x = ValueTier, y = MonthlyCharges, fill = ValueTier)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Monthly Charges by Churner Value Tier", y = "Monthly Charges", x = "Value Tier")

# Boxplot of Tenure
ggplot(churners, aes(x = ValueTier, y = tenure, fill = ValueTier)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Tenure by Churner Value Tier", y = "Tenure (Months)", x = "Value Tier")

# Stacked bar: Contract type distribution by tier
ggplot(churners, aes(x = ValueTier, fill = Contract)) +
  geom_bar(position = "fill") +
  labs(title = "Contract Type Distribution by Value Tier", y = "Proportion", x = "Value Tier") +
  theme_minimal()

# Stacked bar: Payment method distribution
ggplot(churners, aes(x = ValueTier, fill = PaymentMethod)) +
  geom_bar(position = "fill") +
  labs(title = "Payment Method Distribution by Value Tier", y = "Proportion", x = "Value Tier") +
  theme_minimal()