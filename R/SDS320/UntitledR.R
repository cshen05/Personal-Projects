library(car)
library(ggplot2)
library(dplyr)

cs65692 <- Robey
cs65692

mean_tfr <- mean(Robey$tfr, na.rm = TRUE)
median_tfr <- median(Robey$tfr, na.rm = TRUE)
sd_tfr <- sd(Robey$tfr, na.rm = TRUE)
iqr_tfr <- IQR(Robey$tfr, na.rm=TRUE)

summary_stats <- data.frame(
  Statistic = c("Mean", "Median", "Standard Deviation", "IQR"),
  Value = c(mean_tfr, median_tfr, sd_tfr, iqr_tfr)
)
print(summary_stats)

ggplot(Robey, aes(x = tfr)) +
  geom_histogram(binwidth = 0.5, fill = "blue", color = "black", alpha = 0.7) +
  geom_density(alpha = 0.2, fill = "red") +
  labs(title = "Distribution of Total Fertility Rate (TFR)",
       x = "Total Fertility Rate",
       y = "Frequency") +
  theme_minimal()

ggplot(Robey, aes(x = region, y = tfr, fill = region)) +
  geom_boxplot() +
  labs(title = "Comparison of Total Fertility Rate Across Regions",
       x = "World Region",
       y = "Total Fertility Rate (TFR)") +
  theme_minimal()

stat_tfr_by_region <- Robey %>%
  group_by(region) %>%
  summarise(mean_tfr = mean(tfr, na.rm = TRUE),
            median_tfr = median(tfr, na.rm=TRUE),
            sd_tfr = sd(tfr, na.rm=TRUE),
            iqr_tfr = IQR(tfr, na.rm=TRUE),
            min_tfr = min(tfr, na.rm=TRUE),
            max_tfr = max(tfr, na.rm=TRUE))

print(stat_tfr_by_region)