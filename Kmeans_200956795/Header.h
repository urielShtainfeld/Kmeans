#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <mpi.h>
#include<math.h>
#include <string.h>
#include <malloc.h>
#include<omp.h>

#define NUM_OF_THREAD 1024
#define CUDA_THREAD 1000

typedef struct
{
	float x;
	float y;
	float Vx;
	float Vy;
	int cluster_Number;
	float distanceFromPointToCloseCluster;
}Point;

typedef struct
{
	int Number;
	float x;
	float y;
	float diameter;
	float q;
}Cluster;




void fileRead(int *N, int *K, float *T, float *dT, int *Limit, float *QM, Cluster **arrOfClusters, Cluster **arrOfPreviuseClusters, Point **arrOfPoints);
void InitClusters(Cluster **arrOfClusters, Cluster **arrOfPreviuseClusters, Point *arrOfPoints, int K);
MPI_Datatype CreatePointType();
MPI_Datatype CreateClusterType();
void bCast(float *dt, int *Limit, int *N, float *T, int *dividedSize, int *loopStoper, float *q, int *K, float *QM);
void bCastForClusters(Cluster *arrOfClusters, Cluster *arrOfPreviuseClusters, int K);
void classifyPointsToClusters(Cluster *arrOfClusters, Point **arrOfPoints, int begin, int until, int threadNumber, int K);
void calcDiameter(Cluster **arrOfClusters, Point *arrOfPoints, int K, int pointsAmount);
void calcDiameterInSpecificCluster(Cluster *cluster, Point *arrOfPoints, int pointsAmount);
void classifyPointsToClustersByDistance(Point *point, Cluster *arrOfClusters, int K);
float calcDis(float p1x, float p1y, float p2x, float p2y);
void calcClustersQuality(Cluster **arrOfClusters, int K);
void calcQualityForSpecificCluster(Cluster **arrOfClusters, Cluster* cluster, int sizeOfrrOfClusters);
void findClustersCenter(Cluster **arrOfClusters, Point *arrOfPoints, int begin, int N, int K);
void findClusterSpecificCenter(Cluster *cluster, Point *arrOfPoints, int begin, int N);
void calcAVGOfCluster(Cluster **cluster, float xOfPoint, float yOfPoint, int pointsInClusterAmount);
float isQualityFeets(Cluster *clusterArry, int clusterSizeArry);
bool CheckClusterChanges(Cluster *arrOfClusters, Cluster *arrOfPreviuseClusters, int K);
void saveClustersArray(Cluster *originalClusterArry, Cluster **CopyClusterArry, int K);
__global__ void setByTimeKernel(Point *arrOfPoints, float dt, int size);
cudaError_t movePointInTime(Point *arrOfPoints, float dt, int size);
void freePointsArrayByCuda(Point *arrOfPoints);
void writeOutputFile(char* fileName, Cluster* clusters, int K, int iter, int limitIter, double time, double maxTime, double quality, double qm);




