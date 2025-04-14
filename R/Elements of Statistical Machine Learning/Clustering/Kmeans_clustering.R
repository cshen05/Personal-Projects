#### This file is for an illustration of K-means clustering

library(ggplot2)
library(tidyr)
library(dplyr)
library(MLmetrics)
library(LICORS) #This package includes K-means++

############################################################################
###################  We generate data from 3 clusters with good separation
set.seed(1000)
N = 1200
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/6), rep(1,N/6), rep(0,N/6))
X[Y == 1,1] = X[Y == 1,1] + 6.5
X[Y == - 1,2] = X[Y == -1,2] + 6.5
experiment_data <- data.frame(X=X)

## Plot the data
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1])) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

################### K-means clustering 

############# We perform K-means clustering with K = 3 clusters
my_kmeans = kmeans(experiment_data, 3, nstart = 20)
### nstart = 20 to make sure the initilization of K-means is sufficiently good
### 3 is for the number of clusterss
my_kmeans$cluster

## Plot the clustering results with 3 clusters
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeans$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############### We perform K-means clustering with K = 4 clusters
my_kmeans = kmeans(experiment_data, 4, nstart = 20)

## Plot the clustering results with 4 clusters
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeans$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############## We perform K-means clustering with K = 5 clusters
my_kmeans = kmeans(experiment_data, 5, nstart = 20)

## Plot the clustering results with 5 clusters
ggplot(data = experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeans$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############################################################################# We generate data from 3 clusters with weak separation
set.seed(1000)
N = 1200
X <- matrix(rnorm(N), ncol = 2)
Y <- c(rep(-1,N/6), rep(1,N/6), rep(0,N/6))
X[Y == 1,1] = X[Y == 1,1] + 5.5
X[Y == - 1,2] = X[Y == -1,2] + 2.5
weak_experiment_data <- data.frame(X=X)

## Plot the data
ggplot(data = weak_experiment_data, aes(x = X[,2], y = X[,1])) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

######################################### K-means clustering 

############ We perform K-means clustering with K = 2 clusters
my_kmeans = kmeans(weak_experiment_data, 2, nstart = 20)

## Plot the clustering results with 2 clusters
ggplot(data = weak_experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeans$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

############ We perform K-means with K = 3 clusters
my_kmeans = kmeans(X, 3, nstart = 20)

## Plot the clustering results with 3 clusters
ggplot(data = weak_experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeans$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank())

################################### K-means clustering with initialization from K-means ++

############## We perform K-means++ with K = 2 clusters
my_kmeanspp = kmeanspp(X, 2, nstart = 20)
### nstart here is to make sure that we can choose good intilization with 
### high probability (It is important because the whole idea of 
### K-means++ is based on the probability of data points)

### The kmeans++ initialization is in general (much) better than that
### from the kmeans

## Plot the clustering results with 3 clusters
ggplot(data = weak_experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeanspp$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

## We perform K-means++ with K = 3 clusters
my_kmeanspp = kmeanspp(X, 3, nstart = 20)

## Plot the clustering results with 3 clusters
ggplot(data = weak_experiment_data, aes(x = X[,2], y = X[,1], color = (my_kmeanspp$cluster))) + geom_point(size = 2) +  theme(legend.position = "none", axis.title.x=element_blank(), axis.title.y=element_blank()) 

########################################### Use elbow method to choose the number of clusters
Kval <-seq(1,10,by = 1) ## We consider the sequence of K from 1 to 10

## We compute the loss values at different choices of K
loss_val <- rep(0, 10)

for (i in 1:10) ##for loop here is for the number of clusters
{
  my_kmeans = kmeans(weak_experiment_data, Kval[i], nstart = 20)
  ### Kval[i] = i, which means that we have i number of clusters
  loss_val[i] = my_kmeans$tot.withinss
  ### This function my_kmeans$tot.withinss gives the loss value of K-means
}

## Plot the loss values at different K
plot(x = Kval, y = loss_val, type = "b", xlab = "Number of clusters", ylab = "Loss values")


