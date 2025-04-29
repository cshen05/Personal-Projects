#### This file is for an illustration of hierarchical clustering


library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics)

############################################# We generate data from 3 clusters with good separation
set.seed(1000)
N = 120
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/6), rep(1,N/6), rep(0,N/6))
X[Y == 1,1] = X[Y == 1,1] + 6.5
X[Y == - 1,2] = X[Y == -1,2] + 6.5
experiment_strong_data <- data.frame(X=X)

## Plot the data
ggplot(data = experiment_strong_data, aes(x = X[,2], y = X[,1])) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############### Hierarchical clustering

### Computing dissimilarity matrix
dissi <- dist(experiment_strong_data, method = "euclidean")
## dist() captures all the possible dissimilarities of data points

### Hierarchical clustering with complete shrinkage
my_hierarchical_complete = hclust(dissi, method = "complete")
# hclust = hierarchical clustering 

plot(my_hierarchical_complete)

### Hierarchical clustering with single shrinkage
my_hierarchical_single = hclust(dissi, method = "single")

plot(my_hierarchical_single)

### Hierarchical clustering with average shrinkage
my_hierarchical_average = hclust(dissi, method = "average")

plot(my_hierarchical_average)

### Hierarchical clustering with centroid shrinkage
my_hierarchical_centroid = hclust(dissi, method = "centroid")

plot(my_hierarchical_centroid)

#################### Determine label for each linkage

### Complete linkage
label_complete = cutree(my_hierarchical_complete, 4)
## cutree creates horizontal line to cut the tree based on the height

ggplot(data = experiment_strong_data, aes(x = X[,2], y = X[,1], color = (label_complete))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

### Single linkage
label_single = cutree(my_hierarchical_single, 4)

ggplot(data = experiment_strong_data, aes(x = X[,2], y = X[,1], color = (label_single))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

### Average linkage
label_average = cutree(my_hierarchical_average, 4)

ggplot(data = experiment_strong_data, aes(x = X[,2], y = X[,1], color = (label_average))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

### Centroid linkage
label_centroid = cutree(my_hierarchical_centroid, 4)

ggplot(data = experiment_strong_data, aes(x = X[,2], y = X[,1], color = (label_centroid))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 


######################################################## We generate data from 3 clusters with weak separation
set.seed(1000)
N = 1200
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/6), rep(1,N/6), rep(0,N/6))
X[Y == 1,1] = X[Y == 1,1] + 2.5
X[Y == - 1,2] = X[Y == -1,2] + 6.5
experiment_data <- data.frame(X=X)

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1])) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############### Hierarchical clustering

### Computing dissimilarity matrix
dissi <- dist(experiment_data, method = "euclidean")

### Hierarchical clustering with complete shrinkage
my_hierarchical_complete = hclust(dissi, method = "complete")

plot(my_hierarchical_complete)

### Hierarchical clustering with single shrinkage
my_hierarchical_single = hclust(dissi, method = "single")

plot(my_hierarchical_single)

### Hierarchical clustering with average shrinkage
my_hierarchical_average = hclust(dissi, method = "average")

plot(my_hierarchical_average)

### Hierarchical clustering with centroid shrinkage
my_hierarchical_centroid = hclust(dissi, method = "centroid")

plot(my_hierarchical_centroid)

#################### Determine label for each linkage

### Complete linkage
  label_complete = cutree(my_hierarchical_complete, 2)
  
  ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (label_complete))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 
  
  ### Single linkage
  label_single = cutree(my_hierarchical_single, 2)
  
  ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (label_single))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 
  
  ### Average linkage
  label_average = cutree(my_hierarchical_average, 2)
  
  ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (label_average))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 
  
  ### Centroid linkage
  label_centroid = cutree(my_hierarchical_centroid, 2)
  
  ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (label_centroid))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 
  







