library(car)

cs65692 <- Robey
head(cs65692)

summary(cs65692$tfr)
table(cs65692$region)

anova_model <- lm(tfr ~ region, data = cs65692)
summary(anova_model)

hist(residuals(anova_model), 
     main = "Histogram of Residuals", 
     xlab = "Residuals", 
     col = "lightblue", 
     breaks = 10)

leveneTest(tfr ~ region, data = cs65692)

library(emmeans)
emmeans(anova_model, pairwise~region)
