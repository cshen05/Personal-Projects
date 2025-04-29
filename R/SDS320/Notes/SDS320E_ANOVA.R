#
#SDS 320E - ANOVA
#


#Import Dataset
VideoGames <- read.csv('VideoGames.csv')
VideoGames$Rating <- factor(VideoGames$Rating, levels=c('E','T','M'))

#Check assumptions
boxplot(VideoGames$MetacriticScore~VideoGames$Rating, main='Video Game Scores by Rating', xlab='Rating', ylab='Metacritic Score (0-100)')

#Run ANOVA
library(car)
my_anova <- lm(VideoGames$MetacriticScore~VideoGames$Rating)
Anova(my_anova)

#Effect size
summary(my_anova)$r.squared

#Post-hoc analysis with Tukey adjustment
library(emmeans)
emmeans(my_anova, pairwise~Rating)

