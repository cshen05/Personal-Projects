library(car)

cs65692 <- Davis
head(cs65692)

cs65692$height_diff <- cs65692$height - cs65692$repht
head(cs65692$height_diff)

outlier <- which.max(abs(cs65692$height_diff))
cs65692 <- cs65692[-outlier, ]

hist(cs65692$height_diff,
     main="Histogram of Height Differences (After Removing Outlier)",
     xlab="Actual Height - Self-reported Height (cm)",
     col="lightblue",
     border="black")
abline(v=mean(cs65692$height_diff, na.rm=TRUE), col="red", lwd=2)

shapiro.test(cs65692$height_diff)
t.test(cs65692$height_diff, 
       mu=0,
       alternative="two.sided")