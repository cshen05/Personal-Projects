### Question 8
college <- read.csv("College.csv")

rownames(college) <- college[,1]
college <- college[,-1]
View(college)

summary(college)

pairs(college[,2:11])
boxplot(Outstate~Private, college)

Elite <- rep("No", nrow(college))
Elite[college$Top10perc > 50] <- "Yes"
Elite <- as.factor(Elite)
college <- data.frame(college, Elite)

summary(college)
boxplot(Outstate~Elite, college)

par(mfrow = c(2, 2))

hist(college$Outstate, breaks = 10, main = "Outstate Tuition (10 bins)", col = "lightblue", xlab = "Outstate Tuition")
hist(college$Outstate, breaks = 20, main = "Outstate Tuition  (20 bins)", col = "orange", xlab = "Outstate Tuition")
hist(college$Outstate, breaks = 30, main = "Outstate Tuition  (30 bins)", col = "lightgreen", xlab = "Outstate Tuition")
hist(college$Outstate, breaks = 40, main = "Outstate Tuition (40 bins)", col = "pink", xlab = "Outstate Tuition")

library(dplyr)
college %>% 
  summarize(
    across(where(is.numeric), list(
      mean = mean,
      sd = sd,
      min = min, 
      max = max
    ), na.rm=TRUE)
  )


### Question 9
auto <- read.csv("Auto.csv", stringsAsFactors = FALSE) 
auto$horsepower <- as.numeric(auto$horsepower)

# quantitative: mpg, displacement, horsepower, weight, acceleration
# qualitative: cylinders, origin, name, year
quant <- c("mpg", "displacement", "horsepower", "horsepower", "weight", "acceleration")
for (x in quant) {
  range <- diff(range(auto[[x]], na.rm=TRUE))
  cat(paste(x,":", range(auto[[x]], na.rm=TRUE), "range:", range, "\n"))
}

for (x in quant) {
  mean <- round(mean(auto[[x]], na.rm=TRUE), 2)
  cat(paste(x,":", mean, "\n"))
}

for (x in quant) {
  sd <- round(sd(auto[[x]], na.rm=TRUE), 2)
  cat(paste(x,":", sd, "\n"))
}

auto_subset <- auto[-(10:85),]
for (x in quant) {
  range <- diff(range(auto_subset[[x]], na.rm=TRUE))
  mean <- round(mean(auto_subset[[x]], na.rm=TRUE), 2)
  sd <- round(sd(auto_subset[[x]], na.rm=TRUE), 2)
  cat(paste(x, " - Range:", range, 
            "Mean:", mean, 
            "SD:", sd, "\n"))
}

library(ggplot2)
# Remove rows with missing values
auto_data <- na.omit(auto)

# MPG vs Weight
ggplot(auto_data, aes(x = weight, y = mpg)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "MPG vs Weight", x = "Weight", y = "MPG") +
  theme_minimal()

# MPG vs Horsepower
ggplot(auto_data, aes(x = horsepower, y = mpg)) +
  geom_point(alpha = 0.5, color = "red") +
  labs(title = "MPG vs Horsepower", x = "Horsepower", y = "MPG") +
  theme_minimal()

# Displacement vs Horsepower
ggplot(auto_data, aes(x = displacement, y = horsepower)) +
  geom_point(alpha = 0.5, color = "green") +
  labs(title = "Horsepower vs Displacement", x = "Displacement", y = "Horsepower") +
  theme_minimal()

# Acceleration vs Weight
ggplot(auto_data, aes(x = weight, y = acceleration)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Acceleration vs Weight", x = "Weight", y = "Acceleration") +
  theme_minimal()

### Question 10
boston <- read.csv("boston.csv")
library(ISLR2)
Boston
?Boston

# Crime rate vs. Distance to employment centers
ggplot(boston, aes(x = dis, y = crim)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Crime Rate vs. Distance to Employment Centers",
       x = "Distance to Employment Centers (dis)",
       y = "Per Capita Crime Rate (crim)") +
  theme_minimal()

# Crime rate vs. Tax Rate
ggplot(boston, aes(x = tax, y = crim)) +
  geom_point(alpha = 0.5, color = "red") +
  labs(title = "Crime Rate vs. Tax Rate",
       x = "Property Tax Rate (tax)",
       y = "Per Capita Crime Rate (crim)") +
  theme_minimal()

correlation_crim <- cor(boston)[, "crim"]
sorted_correlation <- sort(correlation_crim, decreasing = TRUE)
print("Correlation of predictors with crime rate (crim):")
print(sorted_correlation)

find_outliers <- function(column) {
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR_value <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  return(list(lower_bound = lower_bound, upper_bound = upper_bound))
}

crim_outliers <- find_outliers(boston$crim)
high_crime_tracts <- boston %>% filter(crim > crim_outliers$upper_bound)

tax_outliers <- find_outliers(boston$tax)
high_tax_tracts <- boston %>% filter(tax > tax_outliers$upper_bound)

ptratio_outliers <- find_outliers(boston$ptratio)
high_ptratio_tracts <- boston %>% filter(ptratio > ptratio_outliers$upper_bound)

crim_range <- range(boston$crim)
tax_range <- range(boston$tax)
ptratio_range <- range(boston$ptratio)

cat("Number of census tracts with high crime rates:", nrow(high_crime_tracts), "\n")
cat("Crime Rates Range:", crim_range,"\n")
cat("Number of census tracts with high tax rates:", nrow(high_tax_tracts), "\n")
cat("Tax Range:", tax_range,"\n")
cat("Number of census tracts with high pupil-teacher ratios:", nrow(high_ptratio_tracts), "\n")
cat("Pupil-Teacher Range:", ptratio_range,"\n")

cat("Crime Rate Outlier Threshold:", crim_outliers$upper_bound, "\n")
cat("Tax Rate Outlier Threshold:", tax_outliers$upper_bound, "\n")
cat("Pupil-Teacher Ratio Outlier Threshold:", ptratio_outliers$upper_bound, "\n")

charles_river_count <- sum(boston$chas == 1)
cat("Number of census tracts bounding the Charles River:", charles_river_count, "\n")

median_ptratio <- median(boston$ptratio)
cat("Median pupil-teacher ratio:", median_ptratio, "\n")