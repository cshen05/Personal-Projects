library(ggplot2)
library(dplyr)

df <- read.csv("matches.csv", stringsAsFactors = FALSE)

# Clean and select relevant columns
df_clean <- na.omit(df[, c("xg", "poss", "venue")])
colSums(is.na(df_clean))

# Univariate plot: xG
ggplot(df_clean, aes(x = xg)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  geom_density(aes(y = ..count.. * 5), color = "red") +
  labs(title = "Distribution of Expected Goals (xG)", x = "xG", y = "Count") +
  theme_minimal()
median(df_clean$xg)
IQR(df_clean$xg)
min(df_clean$xg)
max(df_clean$xg)

# Univariate plot: Possession %
ggplot(df_clean, aes(x = poss)) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  geom_density(aes(y = ..count.. * 5), color = "darkred") +
  labs(title = "Distribution of Possession %", x = "Possession", y = "Count") +
  theme_minimal()
mean(df_clean$poss)
median(df_clean$poss)
sd(df_clean$poss)
min(df_clean$poss)
max(df_clean$poss)

# Bivariate: Possession vs xG
ggplot(df_clean, aes(x = poss, y = xg)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Possession vs Expected Goals", x = "Possession (%)", y = "xG") +
  theme_minimal()

# Univariate plot: Venue
ggplot(df_clean, aes(x = venue)) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(
    title = "Distribution of Venue", x = "Venue", y = "Count") +
  theme_minimal()
table(df_clean$venue)

# Bivariate: xG by Venue
ggplot(df_clean, aes(x = venue, y = xg)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "xG by Venue", x = "Venue", y = "xG") +
  theme_minimal()
