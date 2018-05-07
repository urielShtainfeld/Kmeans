#include "Header.h"


void findClustersCenter(Cluster **arrOfClusters, Point *arrOfPoints, int begin, int N, int K)
{
	int i;
	omp_set_num_threads(K);
#pragma omp parallel 
	{
		long numOfThreads = omp_get_num_threads();
		long tid = omp_get_thread_num();
#pragma omp parallel for 
		for (i = 0; i < K; i++)
		{
			// finds the center of a specific cluster 
			findClusterSpecificCenter(&((*arrOfClusters)[i]), arrOfPoints, begin, N);
		}
	}
}
void findClusterSpecificCenter(Cluster *cluster, Point *arrOfPoints, int begin, int N)
{
	int i, pointsInClusterAmount = 0;
	float xOfPoint = 0, yOfPoint = 0;
	for (i = begin; i < N; i++)
	{
		// if found that the point is in this cluster -> recalc point
		if ((*cluster).Number == arrOfPoints[i].cluster_Number)
		{
			xOfPoint += arrOfPoints[i].x;
			yOfPoint += arrOfPoints[i].y;
			// counting the amount of the points to calc AVG
			pointsInClusterAmount++;
		}
	}
	if (pointsInClusterAmount != 0)
	{
		// calc avg of cluster points
		calcAVGOfCluster(&cluster, xOfPoint, yOfPoint, pointsInClusterAmount);
	}
}
void calcAVGOfCluster(Cluster **cluster, float xOfPoint, float yOfPoint, int pointsInClusterAmount)
{
	(**cluster).x = xOfPoint / pointsInClusterAmount;
	(**cluster).y = yOfPoint / pointsInClusterAmount;
}

// calculate the diameter of the cluster
void calcDiameter(Cluster **arrOfClusters, Point *arrOfPoints, int K, int pointsAmount)
{
	int i;
	omp_set_num_threads(K);
#pragma omp parallel 
	{
		long numOfThreads = omp_get_num_threads();
		long tid = omp_get_thread_num();
#pragma omp parallel for 
		for (i = 0; i < K; i++)
		{
			// sending each cluster and calc its own diameter
			calcDiameterInSpecificCluster(&((*arrOfClusters)[i]), arrOfPoints, pointsAmount);
		}
	}
}
void calcDiameterInSpecificCluster(Cluster *cluster, Point *arrOfPoints, int pointsAmount)
{
	int i, j;
	float max = 0, tempMax;
	for (i = 0; i < pointsAmount; i++)
	{
		// find the first point in the specific cluster
		if ((*cluster).Number == arrOfPoints[i].cluster_Number)
		{
			for (j = 0; j < pointsAmount; j++)
			{
				// find the second point in the specific cluster
				if ((*cluster).Number == arrOfPoints[j].cluster_Number)
				{
					// calc ther dis between 2 points and put in tempMax and then check if tempMax bigger then real max
					tempMax = calcDis(arrOfPoints[i].x, arrOfPoints[i].y, arrOfPoints[j].x, arrOfPoints[j].y);
					if (tempMax > max)
					{
						max = tempMax;
					}
				}
			}
		}
	}
	(*cluster).diameter = max;
}

void classifyPointsToClusters(Cluster *arrOfClusters, Point **arrOfPoints, int begin, int until, int threadNumber, int K)
{
	int i;

	omp_set_num_threads(threadNumber);
#pragma omp parallel 
	{
		long numOfThreads = omp_get_num_threads();
		long tid = omp_get_thread_num();
#pragma omp parallel for 
		for (i = begin; i < until; i++)
		{
			classifyPointsToClustersByDistance(&((*arrOfPoints)[i]), arrOfClusters, K);
		}
	}
}

void classifyPointsToClustersByDistance(Point *point, Cluster *arrOfClusters, int K)
{
	int i;
	float minDistance = -1, tempminDistance;
	for (i = 0; i < K; i++)
	{
		tempminDistance = calcDis((*point).x, (*point).y, arrOfClusters[i].x, arrOfClusters[i].y);
		if (minDistance < 0 || minDistance>tempminDistance)
		{
			minDistance = tempminDistance;
			(*point).distanceFromPointToCloseCluster = minDistance;
			(*point).cluster_Number = arrOfClusters[i].Number;
		}
	}
}
void calcClustersQuality(Cluster **arrOfClusters, int K)
{
	int i;
	omp_set_num_threads(K);
#pragma omp parallel 
	{
		long numOfThreads = omp_get_num_threads();
		long tid = omp_get_thread_num();
#pragma omp parallel for 
		for (i = 0; i < K; i++)
		{
			// find the Quality between 2 different clusters
			calcQualityForSpecificCluster(arrOfClusters, &((*arrOfClusters)[i]), K);
		}
	}
}
void calcQualityForSpecificCluster(Cluster **arrOfClusters, Cluster* cluster, int sizeOfrrOfClusters)
{
	int i;
	float quality = 0, Dxy;
	for (i = 0; i < sizeOfrrOfClusters; i++)
	{
		if ((*cluster).Number != (*arrOfClusters)[i].Number)
		{
			Dxy = calcDis((*cluster).x, (*cluster).y, (*arrOfClusters)[i].x, (*arrOfClusters)[i].y);
			// (di/Dxy)
			quality += (*cluster).diameter / Dxy;
		}
	}
	(*cluster).q = quality;
}

// calc the dis between 2 points
float calcDis(float p1x, float p1y, float p2x, float p2y)
{
	float x, y, dis;
	x = p1x - p2x;
	y = p1y - p2y;
	x = pow(x, 2);
	y = pow(y, 2);
	dis = sqrt((x + y));
	return dis;
}