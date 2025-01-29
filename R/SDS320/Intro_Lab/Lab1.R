connor <- read_csv("IntroLab_class_survey.csv")

activity_summary <- data.frame(
  videogames = table(connor$videogames),
  sports = table(connor$sports),
  clubs = table(connor$clubs)
)
activity_summary

n_students = nrow(connor)

vg <- 11/n_students
vg

sports <- 6/n_students
sports

clubs <- 15/n_students
clubs‹

