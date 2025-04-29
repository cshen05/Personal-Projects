#
#SDS 320E - Multiple Regression Part 2
#

#Import data
med <- read.csv("MedicalData.csv",header=TRUE)


#Categorical predictor
mymodel3 <- lm(BP ~ Diabetic, data=med)
summary(mymodel3)

#Compare to independent t-test
t.test(med$BP[med$Diabetic=='yes'],med$BP[med$Diabetic=='no'], var.eq=TRUE)


#Categorical and numeric predictors
mymodel4 <- lm(BP ~ Diabetic + BMI, data=med)
summary(mymodel4)


#Multivariate plot
library(ggplot2)

med$yhat <- mymodel4$fitted.values

ggplot(med, aes(x = BMI, y = BP, col = Diabetic, shape = Diabetic)) + geom_point() + xlab("BMI") + ylab("Diastolic Blood Pressure (mmHg)") + labs(col = "Diabetic Status", shape = "Diabetic Status") + ggtitle("Blood Pressure by BMI and Diabetic Status") + theme_classic() + scale_color_manual(values = c("blue", "red")) + geom_line(aes(y = yhat))


#Assumptions
myresid <- mymodel4$residuals
hist(myresid)

myfitted <- mymodel4$fitted.values
plot(myfitted, myresid, pch=16)
abline(h=0, col='red')
