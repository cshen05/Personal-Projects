################################################################################
########## This R file is for simple visualizations in R

# Install the ggplot2 and dplyr package
# install.packages('ggplot2')
# install.packages('dplyr')

#load the library ggplot2
library(ggplot2)
library(dplyr)

################################################################################
######### Scatter plot with mtcars

# A brief look at the first 6 rows of mtcars
head(mtcars, 6)  

# Summary of the data mtcars
summary(mtcars)

# Standard scatterplot for the variables "mpg" and "hp" in mtcars
ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point()

# Change the color/ size/ pattern in the scatterplot for the variables "mpg" 
# and "hp" in mtcars

ggplot(mtcars, aes(x=hp, y=mpg)) + geom_point(size=3, color = "red", shape=18)

# Scatter plot between "mpg" and "hp" where the shapes are based on the number of cylinders

ggplot(mtcars, aes(x=hp, y=mpg, shape = factor(cyl))) + geom_point(aes(color = factor(cyl)), size = 3)

############### Scatter plot between "mpg" and "wt" where the shapes are based on the number of gears

ggplot(mtcars, aes(x=mpg, y=wt, shape = factor(gear))) + geom_point(aes(color = factor(gear)), size = 3)

################################################################################
########### Line plot with economics data

# A brief look at the first 6 rows of mtcars
head(economics, 6)  

# Plot the variable psavert over time
ggplot(economics, aes(x = date, y = psavert)) + geom_line()

####### Plot multiple time series pce and unemploy over time
# Install necessary packages
# install.packages('tidyr')

# Load tidyr package, which helps create tidy data
library('tidyr')

# Select the columns date, pce, and psavert from the economics data
mydata <- economics %>% select(date, pce, unemploy) 

#Merge the values of pce and psavert into the same column
newdata <- mydata %>% gather(key = "label", value = "combine_val", -date)

#Take a brief look at the first 6 rows of newdata
head(newdata, 6)

# Plot the multiple time series
ggplot(newdata, aes(x = date, y = combine_val)) + geom_line(aes(color = label), size = 1) 

####### Plot multiple time series with different scales
### Take a look at the ranges of psavert and unemploy
range(economics$psavert)
range(economics$unemploy)

mydata <- economics %>% select(date, psavert, unemploy) 
newdata <- mydata %>% gather(key = "label", value = "combine_val", -date)
ggplot(newdata, aes(x = date, y = combine_val)) + geom_line(aes(color = label), size = 1)

### Take log with the unemploy and psavert
economics$unemploy <- log(economics$unemploy) 
economics$psavert <- log(economics$psavert) 

### Create newdata with data, psavert, unemploy
mydata <- economics %>% select(date, psavert, unemploy) 
newdata <- mydata %>% gather(key = "label", value = "combine_val", -date)
ggplot(newdata, aes(x = date, y = combine_val)) + geom_line(aes(color = label), size = 1)

################################################################################
###### Histogram with Airpassengers data
data(AirPassengers)

# We turn Airpassengers into data frame
mydata <- data.frame(value = AirPassengers)

# Plot the default histogram
ggplot(mydata, aes(x = value)) + geom_histogram()

# Control the bin size of the histogram
ggplot(mydata, aes(x = value)) + geom_histogram(binwidth = 30, color = "black", fill = "blue")

# Create a density histogram
ggplot(mydata, aes(x = value)) + geom_histogram(aes(y =..density..), binwidth = 30, color = "black", fill = "blue") + geom_density(color = "red")

################################################################################
####### Density plot with Airpassengers data

# We turn Airpassengers into data frame
mydata <- data.frame(value = AirPassengers)

# Create a density plot
ggplot(mydata, aes(x = value)) + geom_histogram(aes(y =..density..), binwidth = 20, color = "black", fill = "blue") + geom_density(color="red")

################################################################################
######## Boxplot with airquality data
data(airquality)

head(airquality, 6)

# Create boxplot for wind
ggplot(airquality, aes(y=Wind)) + geom_boxplot()

ggplot(airquality, aes(y=Wind)) + geom_boxplot(notch= TRUE)

# Create multiple boxplots for Ozone based on Month
ggplot(airquality, aes(x = factor(Month), y = Ozone)) + geom_boxplot() 

# If we want to change the title of x-axis, we can use "xlab"
ggplot(airquality, aes(x = factor(Month), y = Ozone)) + geom_boxplot() + xlab('Month')

# Different colors for each month
ggplot(airquality, aes(x = factor(Month), y = Ozone, color = factor(Month))) + geom_boxplot()

# Create multiple boxplots for Temperature based on Day
ggplot(airquality, aes(x = factor(Day), y = Temp)) + geom_boxplot() 

# Create multiple boxplots for Temperature based on Month
ggplot(airquality, aes(x = factor(Month), y = Temp)) + geom_boxplot() 

################################################################################
####### Violin plot with airquality data

# Create single violin plot for wind
ggplot(airquality, aes(x = "", y = Wind)) + geom_violin() + geom_boxplot(width=0.1)

# Create multiple violin plots for Ozone based on Month
ggplot(airquality, aes(x = factor(Month), y = Ozone)) + geom_violin() + geom_boxplot(width = 0.1)


ggplot(airquality, aes(x = factor(Month), y = Ozone, color = factor(Month))) + geom_violin() + geom_boxplot(width = 0.1)

################################################################################
####### Facets with airquality data

# Use facets with boxplots
ggplot(airquality, aes(y = Ozone)) + geom_boxplot() + facet_grid(cols = vars(Month))

# Use facets with violin plots
ggplot(airquality, aes(x = " ", y = Wind)) + geom_violin() + geom_boxplot(width = 0.1) + facet_grid(cols = vars(Month))

################################################################################
###### Heatmap visualization

# We plot the heatmap with data mtcars
# First, we turn mtcars into matrix form
#newmtcars <- as.matrix(mtcars)

# Heatmap for all the variables
#heatmap(newmtcars, scale = "col")

# Removing the clustering
#heatmap(newmtcars, Colv = NA, Rowv = NA, scale = "col")

################################################################################
###### Network visualization

### We install the built-in packages "visNetwork" and "geomnet"
### More information about these packages are in: https://github.com/sctyner/geomnet 
### and https://cran.r-project.org/web/packages/visNetwork/vignettes/Introduction-to-visNetwork.html

install.packages("visNetwork")
install.packages("geomnet")
install.packages("igraph")

# Load the libraries visNetwork and geomnet
library(visNetwork)
library(geomnet)
library(igraph)

### Load the data lesmis from the package geomnet
data(lesmis)

######### Explore Nodes
nodes <- as.data.frame(lesmis[2])
head(nodes)

### Create the new column names for nodes based on the format of library visNetwork
colnames(nodes) <- c("id", "label")
nodes$id <- nodes$label

######### Explore Edges
edges <- as.data.frame(lesmis[1])
head(edges)

### Create the new column names for edges based on the format of library visNetwork
colnames(edges) <- c("from", "to", "width")

######## Create a network from nodes and edges
## visIgraphLayout is for fast rendering of the graph
visNetwork(nodes, edges) %>% visIgraphLayout() 

##### Create colors for nodes and edges with network visualization
visNetwork(nodes, edges) %>% visIgraphLayout() %>% visNodes(color = "blue") %>% visEdges(color = "red")


################################################################################
##### Map visualization for world population
##### The package rnaturalearth yields a map of countries around the world and
##### other packages are useful for spatial data
install.packages("rnaturalearth")
install.packages("rnaturalearthdata")
install.packages('Rcpp')
install.packages("sf")
install.packages("rgeos")

library(rnaturalearth)
library(rnaturalearthdata)
library(Rcpp)
library(sf)
library(rgeos)

# Create world data
# The command ne.countries is used to pull the country data
world <- ne_countries(scale = "medium", returnclass = "sf")

# Plot the basic world map
ggplot(data = world) + geom_sf()

# Plot the world map based on world population
ggplot(data = world) + geom_sf(aes(fill = pop_est))

# Different kinds of colors
ggplot(data = world) + geom_sf(aes(fill = pop_est)) + scale_fill_viridis_c(option = "magma")

ggplot(data = world) + geom_sf(aes(fill = pop_est)) + scale_fill_viridis_c(option = "plasma")











