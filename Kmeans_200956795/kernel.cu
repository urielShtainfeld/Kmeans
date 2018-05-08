#include"Header.h"


__global__ void setByTimeKernel(Point *arrOfPoints, float dt, int size)
{
	unsigned long id;
	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size)
	{
		arrOfPoints[id].x = arrOfPoints[id].x + (dt * arrOfPoints[id].Vx);
		arrOfPoints[id].y = arrOfPoints[id].y + (dt * arrOfPoints[id].Vy);
	}
}


cudaError_t movePointInTime(Point *arrOfPoints, float dt, int size)
{
	Point *pointsArray = 0;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	int numOfThreadsPerBlock;

	cudaGetDeviceProperties(&prop, 0);
	numOfThreadsPerBlock = prop.maxThreadsPerBlock;
	int numOfBlock = (size / numOfThreadsPerBlock);
	if (size % numOfThreadsPerBlock != 0)
	{
		numOfBlock += 1;
	}

	//check for the cuda device if have error reaching him
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}
	//check if there are error in malloc space
	cudaStatus = cudaMalloc((void**)&pointsArray, size * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}
	// check if copy there is error in copy from host
	cudaStatus = cudaMemcpy(pointsArray, arrOfPoints, size * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}

	setByTimeKernel << < numOfBlock, size >> >(pointsArray, dt, size);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arrOfPoints, pointsArray, size * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freePointsArrayByCuda(pointsArray);
		return cudaStatus;
	}


}
void freePointsArrayByCuda(Point *arrOfPoints)
{
	cudaFree(arrOfPoints);
}
