library(ggplot2)
library(dplyr)
library(tidyverse)

head(mtcars)

ggplot(mtcars, aes(x=mpg, y=wt, shape = factor(gear))) + geom_point(aes(color = factor(gear)), size = 3)
