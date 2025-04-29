#SDS 320E - One-Sample t Test

#Bosworth sleep data:
sleepdata <- c(16.6, 17.3, 21.3, 18.4, 20.2, 18, 18.5, 19.3, 18.7, 18.3)
mean(sleepdata)
sd(sleepdata)

#Make histogram to visualize distribution
hist(sleepdata, main='Bosworth Sleep Times', 
     xlab='Daily Hours Slept')

#Conduct the test:
t.test(sleepdata, mu=17)


