data('trees')
cs65692 <- trees

most_frequent_height <- as.numeric(names(sort(table(cs65692$Height), decreasing = TRUE)[1]))
most_frequent_height

mean_girth <- mean(cs65692$Girth)
std_dev_girth <- sd(cs65692$Girth)

mean_girth
std_dev_girth