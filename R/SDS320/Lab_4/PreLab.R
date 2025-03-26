library(dplyr)
library(car)
library(emmeans)

data <- read.csv("ANOVALab_insurance.csv")

boxplot(costs~smoker, data=data, 
        xlab='Smoking Status', 
        ylab='Cost', 
        main='Cost of Medical Bill by Smoking Status', 
        col=c('orchid','pale green','salmon'))


leveneTest(costs~smoker,data=data)

my_anova <- lm(costs~smoker,data=data)
Anova(my_anova)

summary(my_anova)$r.squared

emmeans(my_anova, pairwise~smoker)