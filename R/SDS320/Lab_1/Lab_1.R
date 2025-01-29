data <- read_csv("SubsettingLab_biomass.csv")

head(data)

premontane <- data[data$Category == 'premontane wet forest',]
head(premontane)

mean(premontane$AGB)
sd(premontane$AGB)