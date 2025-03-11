library(car)
data <- read.csv("TTestLab_acupuncture.csv")
leveneTest(severity_diff ~ group, data = data)
t.test(severity_diff ~ group, data = data, var.equal = TRUE)
