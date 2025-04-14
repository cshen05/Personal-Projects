library(dplyr)
library(tidyr)
library(caret)
library(randomForest)
library(ggplot2)

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
# -------- HYPERPARAMETER TUNING OVER MTRY --------
mtry_values <- seq(1, ncol(train_encoded) - 1, by = 5)
test_errors <- numeric(length(mtry_values))

for (i in seq_along(mtry_values)) {
  rf <- randomForest(x = train_encoded[, setdiff(names(train_encoded), "Churn")],
                     y = as.factor(train_encoded$Churn),
                     mtry = mtry_values[i], ntree = 300)
  
  preds <- predict(rf, newdata = test_encoded)
  test_errors[i] <- mean(preds != test_encoded$Churn)  # classification error rate
}

# -------- PLOT TEST ERROR VS MTRY --------
error_df <- data.frame(mtry = mtry_values, ErrorRate = test_errors)

ggplot(error_df, aes(x = mtry, y = ErrorRate)) +
  geom_line(color = "blue", size = 1.2) +
  labs(
    title = "Random Forest Classification Error vs mtry",
    x = "mtry (predictors per split)",
    y = "Test Error Rate"
  ) +
  theme_minimal()

# -------- FINAL MODEL WITH BEST MTRY --------
best_mtry <- mtry_values[which.min(test_errors)]

final_rf <- randomForest(x = train_encoded[, setdiff(names(train_encoded), "Churn")],
                         y = as.factor(train_encoded$Churn),
                         mtry = best_mtry, ntree = 500, importance = TRUE)
# Predict on test set
final_preds <- predict(final_rf, newdata = test_encoded[, setdiff(names(test_encoded), "Churn")])

# Final error rate
final_error <- mean(final_preds != test_encoded$Churn)
print(paste("Final Test Error Rate:", round(final_error, 4)))

# -------- PLOT VARIABLE IMPORTANCE --------
varImpPlot(final_rf)

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
# -------- Step 1: Filter to churners --------
churners <- df %>% filter(Churn == 1)

# -------- Step 2: Create revenue tiers using TotalCharges --------
churners <- churners %>%
  mutate(ValueTier = case_when(
    TotalCharges >= quantile(TotalCharges, 0.75) ~ "High",
    TotalCharges >= quantile(TotalCharges, 0.25) ~ "Medium",
    TRUE ~ "Low"
  )) %>%
  mutate(ValueTier = factor(ValueTier, levels = c("Low", "Medium", "High")))

# -------- Step 3: Explore numeric feature differences --------
# Average tenure and charges by value tier
aggregate(cbind(tenure, MonthlyCharges, TotalCharges) ~ ValueTier, data = churners, mean)

# -------- Step 4: Explore categorical patterns --------
# Contract types by tier
table(churners$ValueTier, churners$Contract)

# Payment methods by tier
table(churners$ValueTier, churners$PaymentMethod)

# Tech support usage by tier
table(churners$ValueTier, churners$TechSupport)

# -------- Step 5: Visualizations --------

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