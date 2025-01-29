trips <- read.csv("IntroHandout_bikes.csv")

stdev <- sd(trips$speed)
stdev

n_participants <- nrow(trips[trips$employed == 1,])
n_participants