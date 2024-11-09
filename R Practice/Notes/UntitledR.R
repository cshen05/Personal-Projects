ggplot(data=survey_year) +
  geom_bar(aes(x=year, fill=year)) +
  labs(title="Distribution of Grade Level", x="Class", y="Count", fill="Class")