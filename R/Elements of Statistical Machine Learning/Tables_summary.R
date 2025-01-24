#################################################################################### 
##### This R file is for data summary and tables

### Install necessary packages
# ggplot2 is for data visualization
# dplyr is for data frame 
install.packages('ggplot2')
install.packages('dplyr')

#load the packages
library('ggplot2')
library('dplyr')

# Take a look at the "mpg" data
head(mpg, 6)

# Determine the number of cars in each class from the manufacturers
xtabs(~manufacturer + class, data = mpg)

# Average of city mileage based on classes
mpg %>% group_by(class) %>% summarize(mean_cty = mean(cty))

# Average of both city and highway mileage based on classes
mpg %>% group_by(class) %>% summarize(mean_cty = mean(cty), mean_hwy = mean(hwy))

# Max, min, average of city mileage based on the number of cylinders
mpg %>% group_by(cyl) %>% summarize(min_cty = min(cty), mean_cty = mean(cty), max_cty = max(cty))

# Max, min, average of city mileage and highway mileage based on the number of cylinders
mpg %>% group_by(manufacturer) %>% summarize(min_hwy = min(hwy), mean_hwy = mean(hwy), max_hwy = max(hwy), min_cty = min(cty), mean_cty = mean(cty), max_cty = max(cty))
