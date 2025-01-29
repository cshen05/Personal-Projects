######################################################################
################# This R file is for visualization with worldBank data
######################################################################

### First thing to do is load the libraries
library(ggplot2)
library(tidyr)
library(dplyr)

### Nex thing to do is to set up the working directory for R
### Guidance for how to set up working directory is in: http://www.sthda.com/english/wiki/running-rstudio-and-setting-up-your-working-directory-easy-r-programming

# Determine the current directory
getwd()

# Set the working directory (you need to choose different directory in your computer)
setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

################################################################################
############## We first try with the "agriculture_worldbank.csv"
agriculture = read.csv("agriculture_worldbank.csv", head = TRUE, check.names=FALSE)

# Take a brief look at data
summary(agriculture)

head(agriculture, 3)

# The data are not well-formatted. So, we will need to clean the data a bit
# We will merge the year together
agriculture <- agriculture %>% gather(key = year, value = agriculture, -Country)

# The years are not numeric and we need to turn them into numeric numbers
is.numeric(agriculture$year)
agriculture$year <- as.numeric(agriculture$year)

agriculture <- data.frame(agriculture)

############ Compare the GDP from agriculture of different countries via line plot
ggplot(agriculture, aes(x = year, y = agriculture, group = Country)) + geom_line(aes(color = Country), size = 1)

## We can make the line plots of each country more standout with points
ggplot(agriculture, aes(x = year, y = agriculture, group = Country)) + geom_line(aes(color = Country), size = 1) + geom_point(aes(color = Country), size = 1)

########### Create histogram, density histogram, and density plot for the GDP with agriculture of Brazil
### First we create data frame with only data from Brazil
agri_Brazil <- subset(agriculture, Country == "Brazil")

### Histogram with different binwidths
ggplot(agri_Brazil, aes(x = agriculture)) + geom_histogram(binwidth = 2, color = "black", fill = "blue")

ggplot(agri_Brazil, aes(x = agriculture)) + geom_histogram(binwidth = 1, color = "black", fill = "blue")

### Density Histogram
ggplot(agri_Brazil, aes(x = agriculture)) + geom_histogram(aes(y =..density..), binwidth = 2, color = "black", fill = "blue")

### Density plot
ggplot(agri_Brazil, aes(x = agriculture)) + geom_histogram(aes(y =..density..), binwidth = 2, color = "black", fill = "blue") + geom_density(color="red")

#################### Create boxplots and violin plots for GDP with agriculture of different countries

###### Boxplots
ggplot(agriculture, aes(x = factor(Country), y = agriculture)) + geom_boxplot()

# Another way to compare boxplots is to use "facet" from ggplot2
ggplot(agriculture, aes(y = agriculture)) + geom_boxplot() + facet_grid(cols = vars(Country))

# With colors
# Another way to compare boxplots is to use "facet" from ggplot2
ggplot(agriculture, aes(y = agriculture, color = factor(Country))) + geom_boxplot() + facet_grid(cols = vars(Country))

# The boxplots of Canada, Germany, UK, and US may seem hard to see (as they only have few data)
# Another way to take a subset of rows from a dataset
truncate <- agriculture[agriculture$Country == "Canada"| agriculture$Country == "Germany"| agriculture$Country == "United Kingdom"| agriculture$Country == "United States",]  

# Create boxplots with only Canada, Germany, UK, and US
ggplot(truncate, aes(y = agriculture)) + geom_boxplot() + facet_grid(cols = vars(Country))

###### Violin plots
# Default violin plots
ggplot(agriculture, aes(x = factor(Country), y = agriculture)) + geom_violin() 

# The default violin plots can look ugly. We can use "facets" to deal with it. We create facet with 5 columns
ggplot(agriculture, aes(x= "", y = agriculture)) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')

# With colors 
ggplot(agriculture, aes(x= "", y = agriculture, color = factor(Country))) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')

# As the violin plots of Canada, Germany, UK, and US also hard to see when we compare with other countries, we will only compare the violin plots among these four countries
ggplot(truncate, aes(x= "", y = agriculture)) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')

################################################################################
############## We now try with the "industry_worldbank.csv" and "service_worldbank.csv"

library(ggplot2)
library(tidyr)
library(dplyr)

# Determine the current directory
getwd()

# Set the working directory (you need to choose different directory in your computer)
setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

# We read the data from csv file
industry = read.csv("industry_worldbank.csv", head = TRUE, check.names=FALSE)
service = read.csv("service_worldbank.csv", head = TRUE, check.names=FALSE)

# We will need to clean the data a bit (similar to the agriculture case)
industry <- industry %>% gather(key = year, value = industry, -Country)
industry$year <- as.numeric(industry$year)
industry <- data.frame(industry)

service <- service %>% gather(key = year, value = service, -Country)
service$year <- as.numeric(service$year)
service <- data.frame(service)

############ Compare the GDP from industry and service of different countries via line plot
ggplot(industry, aes(x = year, y = industry, group = Country)) + geom_line(aes(color = Country), size = 1) + geom_point(aes(color = Country), size = 1)

ggplot(service, aes(x = year, y = service, group = Country)) + geom_line(aes(color = Country), size = 1) + geom_point(aes(color = Country), size = 1)

########### Create histogram, density histogram, and density plot for the GDP with industry and service of United States
### First we create data frames with only data from US
service_US <- subset(service, Country== "United States")
industry_US <- subset(industry, Country == "United States")

### Histogram with different binwidths
ggplot(service_US, aes(x = service)) + geom_histogram(binwidth = 0.5, color = "black", fill = "blue")
ggplot(industry_US, aes(x = industry)) + geom_histogram(binwidth = 0.5, color = "black", fill = "blue")

### Density Histogram
ggplot(service_US, aes(x = service)) + geom_histogram(aes(y =..density..), binwidth = 0.5, color = "black", fill = "blue")
ggplot(industry_US, aes(x = industry)) + geom_histogram(aes(y =..density..), binwidth = 0.5, color = "black", fill = "blue")

### Density plot
ggplot(service_US, aes(x = service)) + geom_histogram(aes(y =..density..), binwidth = 0.5, color = "black", fill = "blue") + geom_density(color="red")
ggplot(industry_US, aes(x = industry)) + geom_histogram(aes(y =..density..), binwidth = 0.5, color = "black", fill = "blue") + geom_density(color="red")

#################### Create boxplots and violin plots for GDP with service and industry of different countries

# We compare boxplots using "facet" from ggplot2
ggplot(service, aes(y = service)) + geom_boxplot() + facet_grid(cols = vars(Country))
ggplot(industry, aes(y = industry)) + geom_boxplot() + facet_grid(cols = vars(Country))

# With colors
ggplot(service, aes(y = service, color = factor(Country))) + geom_boxplot() + facet_grid(cols = vars(Country))
ggplot(industry, aes(y = industry, color = factor(Country))) + geom_boxplot() + facet_grid(cols = vars(Country))

###### Violin plots
# Violin plots with facets
ggplot(service, aes(x= "", y = service)) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')
ggplot(industry, aes(x= "", y = industry)) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')

# With colors 
ggplot(service, aes(x= "", y = service, color = factor(Country))) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')
ggplot(industry, aes(x= "", y = industry, color = factor(Country))) + geom_violin() + geom_boxplot(width = 0.1) + facet_wrap(~ Country, ncol = 5) + xlab('Country')



