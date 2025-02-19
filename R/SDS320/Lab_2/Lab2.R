data <- read.csv("DescriptivesLab_acupuncture.csv")

#does acupuncture (group) affect the severity of the headache (severity_diff)

library(ggplot2)

ggplot(data, aes(x=group, y=severity_diff)) +
  geom_boxplot()

acupuncture <- data %>% 
  group_by(group) %>% 
  filter(group == 'Acupuncture')

median(acupuncture$severity_diff, na.rm=TRUE)
IQR(acupuncture$severity_diff, na.rm=TRUE)
range(acupuncture$severity_diff, na.rm=TRUE)

control <- data %>% 
  group_by(group) %>% 
  filter(group == 'Control')

median(control$severity_diff, na.rm=TRUE)
IQR(control$severity_diff, na.rm=TRUE)
range(control$severity_diff, na.rm=TRUE)