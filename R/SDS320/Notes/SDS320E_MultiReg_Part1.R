#SDS 320E - Multiple Regression Part 1

#Import data from file
med <- read.csv("MedicalData.csv",header=TRUE)

#Upload csv into R then load into the session
med <- MedicalData

#Simple regression (one numeric predictor)
mymodel1 <- lm(BP ~ BMI, data=med)
summary(mymodel1)

mymodel2 <- lm(BP ~ Glucose, data=med)
summary(mymodel2)

#Adding another numeric predictor
mymodel3 <- lm(BP ~ BMI + Glucose, data=med)
summary(mymodel3)
