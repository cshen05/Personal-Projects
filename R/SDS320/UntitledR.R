set.seed(1)
x1 <- runif(100)
x2 <- 0.5 * x1 + rnorm(100) / 10
y <- -2 + 2 * x1 + 0.3 * x2 + rnorm(100)

correlation <- cor(x1, x2)
print(paste("Correlation between x1 and x2:", round(correlation, 3)))

plot(x1, x2, main = paste("Scatterplot of x1 vs x2 (Correlation = ", round(correlation, 3), ")"),
     xlab = "x1", ylab = "x2", pch = 19, col = "blue")
grid()

model <- lm(y ~ x1 + x2)
summary(model)

model_x1 <- lm(y ~ x1)
summary(model_x1)

model_x2 <- lm(y ~ x2)
summary(model_x2)

x1 <- c(x1, 0.1)
x2 <- c(x2, 0.8)
y <- c(y, 6)

model <- lm(y ~ x1 + x2)
summary(model)

model_x1 <- lm(y ~ x1)
summary(model_x1)

model_x2 <- lm(y ~ x2)
summary(model_x2)

#11
set.seed(2)
x <- rnorm(100)
y <- 2*x+rnorm(100)

model <- lm(y~x+0)
summary(model)

model <- lm(x~y+0)
summary(model)

model <- lm(y ~ x + 0)
summary(model)

n <- length(x)
sum_x2 <- sum(x^2)
sum_y2 <- sum(y^2)
sum_xy <- sum(x * y)

t_stat_formula <- (sqrt(n - 1) * sum_xy) / sqrt((sum_x2 * sum_y2) - (sum_xy^2))

t_stat_model <- summary(model)$coefficients[1, 3]

cat("t-statistic from regression model:", t_stat_model, "\n")
cat("t-statistic from formula:", t_stat_formula, "\n")

model_y_on_x <- lm(y ~ x)
summary(model_y_on_x)

t_stat_y_on_x <- summary(model_y_on_x)$coefficients[2, 3]

model_x_on_y <- lm(x ~ y)
summary(model_x_on_y)

t_stat_x_on_y <- summary(model_x_on_y)$coefficients[2, 3]

cat("t-statistic for regression of y onto x:", t_stat_y_on_x, "\n")
cat("t-statistic for regression of x onto y:", t_stat_x_on_y, "\n")

