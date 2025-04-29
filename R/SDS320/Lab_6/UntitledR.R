data <- read.csv("GLMLab_bikeshare.csv")


glm_model <- lm(ridership ~ holiday + windspeed, data = data)

# Confirm linearity of numeric predictors
plot(data$windspeed, data$ridership, xlab = "Windspeed (mph)", ylab = "Ridership",
     main = "Windspeed and Ridership", pch = 20)

# Confirm normality of residuals
hist(glm_model$residuals, main = "Model Residuals", xlab = "Residual",
     col = "light grey", right = F)

plot(glm_model$fitted.values, glm_model$residuals, xlab = "Fitted Values",
     ylab = "Residuals", main = "Residual Plot", pch = 20)
abline(h = 0, col = "red")

summary(glm_model)
summary(glm_model)$adj.r.squared

################################### MULTIVARIATE
library(ggplot2)

data$yhat <- glm_model$fitted.values

ggplot(data, aes(x = windspeed, y = ridership, col = holiday, shape = holiday)) +
  geom_point() + xlab("Windspeed (mph)") + ylab("Ridership") +
  labs(col = "Holiday", shape = "Holiday") + ggtitle("Ridership by Holiday, Windspeed") +
  theme_classic() + scale_color_manual(values = c("blue", "orange")) +
  geom_line(aes(y = yhat))