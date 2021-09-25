import numpy as np
import csv
from datetime import datetime
from numpy import genfromtxt
import pandas as pd


mydata= genfromtxt("Iris.csv",delimiter=',')
irisarray = np.ones((150,5))
temps = np.empty([150,5])

PATH= "Iris.csv"
print("Reading CSV file...")
print(irisarray.shape)
print(temps.shape)
with open(PATH) as csvfile:
  readCSV = csv.reader(csvfile, delimiter=',')

  for i ,row in enumerate(readCSV):

    irisarray[i,0]=row[0]
    irisarray[i, 1] = row[1]
    irisarray[i, 2] = row[2]
    irisarray[i, 3] = row[3]
    if row[4] == 'Iris-setosa':
      irisarray[i, 4] = 1
    elif row[4] == 'Iris-versicolor':
      irisarray[i, 4] = 2
    elif row[4] == 'Iris-virginica':
      irisarray[i, 4] = 3

def dist(datapoint,centroid):#to get distance between 2 points
  distance = pow(datapoint[0]-centroid[0],2) + pow(datapoint[1]-centroid[1],2) +pow(datapoint[2]-centroid[2],2) + pow(datapoint[3]-centroid[3],2)
  return distance


def inilzClusAnCentroids(irisarray): # as per found data set, Note: centroids chosen as follow; the average datapoint in each class
  centroids = np.empty([3, 5])
  clus1 = np.empty([0,5])
  clus2 = np.empty([0,5])
  clus3 = np.empty([0,5])

  for  row in irisarray:
    if(row[4]==1):
      clus1 = np.vstack([clus1,row])
    if (row[4] == 2):
      clus2 = np.vstack([clus2, row])
    if (row[4] == 3):
      clus3 = np.vstack([clus3, row])
  #recalculate centroids after each iteration
  centroids[0] = np.average(clus1, axis=0)
  centroids[1] = np.average(clus2, axis=0)
  centroids[2] = np.average(clus3, axis=0)

  return clus1,clus2,clus3 , centroids


def calculateDistance(irisarray): #to get distance between datapoints and their clasess's centroids
  clus1, clus2, clus3, centroids = inilzClusAnCentroids(irisarray)
  distance = 0

  for row in clus1:
    distance += dist(row, centroids[0])

  for row in clus2:
    distance += dist(row , centroids[1])

  for row in clus3:
    distance += dist(row , centroids[2])

  return distance



def Kmeans(irisarray):
  clus1, clus2, clus3, centroids = inilzClusAnCentroids(irisarray)


  clus1 = np.empty([0, 5])
  clus2 = np.empty([0, 5])
  clus3 = np.empty([0, 5])


  for row in irisarray:


    for centr in  centroids:

      oldDistance = dist(row, centroids[int(row[4] - 1)])

      dis = dist(row , centr)
      if (dis < oldDistance):
        row[4]=centr[4]   #if a centroid is closer than assaign that datapoint 'row' to that centroid

    # build new class as per best distances
    if(row[4] == 1):
      clus1 = np.vstack([clus1,row])
    if(row[4] == 2):
      clus2 = np.vstack([clus2, row])
    if(row[4] == 3):
      clus3 = np.vstack([clus3, row])

  # calculate distances after one iteration of above job
  distance = 0
  for row in clus1:
    distance += dist(row, centroids[0])

  for row in clus2:
    distance += dist(row, centroids[1])

  for row in clus3:
    distance += dist(row, centroids[2])

  centroids[0] = np.average(clus1, axis=0)
  centroids[1] = np.average(clus2, axis=0)
  centroids[2] = np.average(clus3, axis=0)

  return clus1,clus2,clus3,distance







for i in range (9):
 print("distance before iteration ", i ," = ", calculateDistance(irisarray))
 clus1,clus2,clus3,distance= Kmeans(irisarray)

 print("distance after iteration ", i ," = ", distance)
