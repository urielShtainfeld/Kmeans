#include "Header.h"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	int  namelen, numprocs, myid;
	int i, j;
	float QM, T, dt, q, time = 0, finishTime, pointsStayInSameCluster = 1;
	bool isPointsMoved = false;
	Point *arrOfPoints;
	Point *dividedArr;
	Cluster *arrOfClusters;
	Cluster *arrOfPreviuseClusters;
	int  Limit, N, K, loopStoper, dividedSize;

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;
	if (myid == 0)
	{
		// read from file
		fileRead(&N, &K, &T, &dt, &Limit, &QM, &arrOfClusters, &arrOfPreviuseClusters, &arrOfPoints);
		// set the divided size
		dividedSize = N / numprocs;
		// init Clusters
		InitClusters(&arrOfClusters, &arrOfPreviuseClusters, arrOfPoints, K);
		loopStoper = 0;
		q = 0;
	}
	// sending bCast and allocate arrrays for p0
	bCast(&dt, &Limit, &N, &T, &dividedSize, &loopStoper, &q, &K, &QM);


	// allocate the arrays for the other proccesses
	if (myid != 0)
	{
		arrOfClusters = (Cluster*)calloc(K, sizeof(Cluster));
		arrOfPreviuseClusters = (Cluster*)calloc(K, sizeof(Cluster));
	}
	bCastForClusters(arrOfClusters, arrOfPreviuseClusters, K);
	// alloacte divided array (each proccess have same size, in our case 3 prossecces) 
	dividedArr = (Point*)calloc(dividedSize, sizeof(Point));

	while (time<T && loopStoper == 0)
	{
		for (i = 0; i < Limit && pointsStayInSameCluster == 1; i++)
		{
			MPI_Scatter(arrOfPoints, dividedSize, CreatePointType(), dividedArr, dividedSize, CreatePointType(), 0, MPI_COMM_WORLD);
			classifyPointsToClusters(arrOfClusters, &dividedArr, 0, dividedSize, NUM_OF_THREAD, K); //OMP
			MPI_Gather(dividedArr, dividedSize, CreatePointType(), arrOfPoints, dividedSize, CreatePointType(), 0, MPI_COMM_WORLD);

			if (myid == 0)
			{
				findClustersCenter(&arrOfClusters, arrOfPoints, 0, N, K); //OMP
				isPointsMoved = CheckClusterChanges(arrOfClusters, arrOfPreviuseClusters, K);
				if (isPointsMoved == true)
				{
					pointsStayInSameCluster = 1; //  points are still changing clusters
					saveClustersArray(arrOfClusters, &arrOfPreviuseClusters, K);
				}
				else  // points are not changing clusters
				{
					calcDiameter(&arrOfClusters, arrOfPoints, K, N);
					calcClustersQuality(&arrOfClusters, K);
					q = isQualityFeets(arrOfClusters, K); //return float quality
					pointsStayInSameCluster = 0;
				}
			}
			// bCast because arrays changed
			MPI_Bcast(arrOfClusters, K, CreateClusterType(), 0, MPI_COMM_WORLD);
			MPI_Bcast(arrOfPreviuseClusters, K, CreateClusterType(), 0, MPI_COMM_WORLD);
			MPI_Bcast(&pointsStayInSameCluster, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&q, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&loopStoper, 1, MPI_INT, 0, MPI_COMM_WORLD);

		}
		MPI_Bcast(&pointsStayInSameCluster, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (myid == 0)
		{
			if (pointsStayInSameCluster == 1) // points are still changing clusters
			{
				time += dt;
				loopStoper = 0;
				movePointInTime(arrOfPoints, dt, N); // Cuda
			}
			else
			{
				// when the points are not changing clusters i check if the quality is ok
				if (q < QM)
				{
					loopStoper = 1; // finish
					finishTime = time;
					time = T;
				}
				else
				{
					time += dt;
					loopStoper = 0;
					movePointInTime(arrOfPoints, dt, N); // Cuda
				}
			}
		}
		MPI_Bcast(&time, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	if (myid == 0)
	{
		int w;
		if (loopStoper == 1)
		{
			printf(" First occurrence at t = %f with q = %f\n", finishTime, q);
			printf("Centers of the clusters: \n\n");

			for (w = 0; w < K; w++)
			{
				printf("%f %f \n", arrOfClusters[w].x, arrOfClusters[w].y);
			}
		}
		else
		{
			printf("Clusters Not Found\n");
		}
	}
	MPI_Finalize();
	return 0;
}


bool CheckClusterChanges(Cluster *arrOfClusters, Cluster *arrOfPreviuseClusters, int K)
{
	bool isPointsMoved = false;
	int i;
	for (i = 0; i < K; i++)
	{
		if (arrOfClusters[i].x != arrOfPreviuseClusters[i].x || arrOfClusters[i].y != arrOfPreviuseClusters[i].y)
		{
			isPointsMoved = true;
		}
	}
	return isPointsMoved;
}
float isQualityFeets(Cluster *arrOfClusters, int K)
{
	float q = 0.00;
	int i;
	for (i = 0; i < K; i++)
	{
		q += arrOfClusters[i].q;
	}
	q = q / (K*(K - 1));
	return q;
}
MPI_Datatype CreatePointType()
{
	MPI_Datatype PointType;
	MPI_Aint disp[] = { offsetof(Point,x),offsetof(Point,y),offsetof(Point,Vx),offsetof(Point,Vy),offsetof(Point,cluster_Number),offsetof(Point,distanceFromPointToCloseCluster) };
	MPI_Datatype type[] = { MPI_FLOAT,MPI_FLOAT ,MPI_FLOAT,MPI_FLOAT ,MPI_INT,MPI_FLOAT };
	int blocklenType[] = { 1, 1, 1, 1, 1,1 };
	MPI_Type_create_struct(6, blocklenType, disp, type, &PointType);
	MPI_Type_commit(&PointType);
	return PointType;
}

MPI_Datatype CreateClusterType()
{
	MPI_Datatype ClusterType;
	MPI_Aint disp[] = { offsetof(Cluster,Number),offsetof(Cluster,x),offsetof(Cluster,y),offsetof(Cluster,diameter),offsetof(Cluster,q) };
	MPI_Datatype type[] = { MPI_INT ,MPI_FLOAT,MPI_FLOAT,MPI_FLOAT,MPI_FLOAT };
	int blocklenType[] = { 1, 1, 1, 1, 1 };
	MPI_Type_create_struct(5, blocklenType, disp, type, &ClusterType);
	MPI_Type_commit(&ClusterType);
	return ClusterType;
}


void fileRead(int *N, int *K, float *T, float *dt, int *Limit, float *QM, Cluster **arrOfClusters, Cluster **arrOfPreviuseClusters, Point **arrOfPoints)
{
	int i;
	FILE *f = fopen("D:\input.txt", "r");
	fscanf(f, "%d %d %f %f %d %f", N, K, T, dt, Limit, QM);
	(*arrOfClusters) = (Cluster*)calloc((*K), sizeof(Cluster));
	(*arrOfPreviuseClusters) = (Cluster*)calloc((*K), sizeof(Cluster));
	(*arrOfPoints) = (Point*)calloc((*N), sizeof(Point));

	for (i = 0; i < (*N); i++)
	{
		fscanf(f, "%f %f %f %f", &(((*arrOfPoints)[i]).x), &(((*arrOfPoints)[i]).y), &(((*arrOfPoints)[i]).Vx), &(((*arrOfPoints)[i]).Vy));
		((*arrOfPoints)[i]).distanceFromPointToCloseCluster = -1;
	}

	for (i = 0; i < (*K); i++)
	{
		((*arrOfClusters)[i]).Number = i;
		((*arrOfPreviuseClusters)[i]).Number = i;
	}

	fclose(f);
}
void InitClusters(Cluster **arrOfClusters, Cluster **arrOfPreviuseClusters, Point *arrOfPoints, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{
		// init arrOfClusters
		((*arrOfClusters)[i]).x = ((arrOfPoints)[i]).x;
		((*arrOfClusters)[i]).y = ((arrOfPoints)[i]).y;

		// init arrOfPreviuseClusters
		((*arrOfPreviuseClusters)[i]).x = ((arrOfPoints)[i]).x;
		((*arrOfPreviuseClusters)[i]).y = ((arrOfPoints)[i]).y;
	}
}

void saveClustersArray(Cluster *originalClusterArry, Cluster **CopyClusterArry, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{
		((*CopyClusterArry)[i]).Number = originalClusterArry[i].Number;
		((*CopyClusterArry)[i]).x = originalClusterArry[i].x;
		((*CopyClusterArry)[i]).y = originalClusterArry[i].y;
	}
}

void bCast(float *dt, int *Limit, int *N, float *T, int *dividedSize, int *loopStoper, float *q, int *K, float *QM)
{
	MPI_Bcast(dividedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(K, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(T, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(Limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(QM, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(q, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(loopStoper, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void bCastForClusters(Cluster *arrOfClusters, Cluster *arrOfPreviuseClusters, int K)
{
	MPI_Bcast(arrOfClusters, K, CreateClusterType(), 0, MPI_COMM_WORLD);
	MPI_Bcast(arrOfPreviuseClusters, K, CreateClusterType(), 0, MPI_COMM_WORLD);
}
