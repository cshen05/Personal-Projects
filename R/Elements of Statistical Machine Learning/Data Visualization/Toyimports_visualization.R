################################################################################
################ This R file is for visualization with toy imports data "toyimports.csv"
###############################################################################

### First thing to do is load the necessary libraries
library(ggplot2)
library(tidyr)
library(dplyr)

# Set the working directory (you need to choose different directory in your computer)
setwd("/Users/nh23294/Box/Teaching/SDS_323/Data/")

## We first read the data from "toyimports.csv"
toy = read.csv("toyimports.csv", head = TRUE, check.names=FALSE)

## Let's explore first 6 rows of toy data
head(toy, 6)

## The average export of each country and average import of US according to each country 
toy %>% group_by(partner_name) %>% summarize(mean_export = mean(partner_report_export), mean_US_import = mean(US_report_import))

## The total export of each country and total import of US according to each country 
toy %>% group_by(partner_name) %>% summarize(sum_export = sum(partner_report_export), sum_US_import = sum(US_report_import))

################## We create a data frame with only data from Germany, Australia, China, Canada, Colombia
toy_select <- subset(toy, partner_name == "Germany" | partner_name == "Australia" | partner_name == "China" | partner_name == "Canada" | partner_name == "Colombia")

### Sum up all the exports of each country each year
toy_select_total <- toy_select %>% group_by(year, partner_name) %>% summarize(total_export = sum(partner_report_export))

### Plot the total export of Germany, Australia, China, Canada, Colombia over time
ggplot(toy_select_total, aes(x = year, y = total_export, group = partner_name)) + geom_line(aes(color = partner_name), size = 1) + geom_point(aes(color = partner_name), size = 1)

## We can observe that the scale of total export of Germany is much bigger than those of Australia and Colombia
## To account for the mismatch in scale, we can substract the total export of Germany by 25000
toy_select_total$total_export <- log( toy_select_total$total_export)

### Plot the new total export of Germany, Australia, Colombia over time
ggplot(toy_select_total, aes(x = year, y = total_export, group = partner_name)) + geom_line(aes(color = partner_name), size = 1) + geom_point(aes(color = partner_name), size = 1)
