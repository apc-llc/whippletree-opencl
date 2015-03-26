#include <cstdlib>
#include <stdio.h>
//#include <cublas_v2.h>
#include <iostream>
#include <memory>
#include <time.h>
#include <tools/utils.h>

#include "queueDistLocks.h"
#include "queueShared.h"
#include "queuingPerProc.h"
#include "techniqueMegakernel.h"
//#include "techniqueKernels.h"
//#include "techniqueDynamicParallelism.h"
#include "segmentedStorage.h"

#include "procedureInterface.h"
#include "procinfoTemplate.h"
#include "random.h"

#include "DynamicTaskManager.h"

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>


cl_context context;
cl_device_id *devices;
cl_command_queue cmdQueue;
cl_kernel * kernels;
cl_uint numDevices;
int used_cl_device;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_program program;
cl_mem dev_globalvars;
Megakernel::globalvarsT host_globalvars;

extern void compile_device_code();

struct dim2 { uint x, y; };

struct MatmulConfig
{
	cl_mem A, B, C;
	size_t n;
	dim2 blockDim_;
	dim2 gridDim_;
};


cl_mem config;

 __inline__ void whippletree_matmul(int threadId, int numThreads, void* ptaskid, volatile uint* shared); 

#ifdef OPNECL_CODE


/*__device__*/ __inline__ void whippletree_matmul(int threadId, int numThreads, __global void* ptaskid, __global volatile uint* shared) 
{
	/*float*& A = config.A;
	float*& B = config.B;
	float*& C = config.C;
	size_t& n = config.n;
	dim2& blockDim_ = config.blockDim_;
	dim2& gridDim_ = config.gridDim_;
	const uint taskid = *(uint*)ptaskid;

	struct { uint x, y; } blockIdx_;
	blockIdx_.x = taskid % gridDim_.x;
	blockIdx_.y = taskid / gridDim_.x;
	
	struct { uint x, y; } threadIdx_;
	threadIdx_.x = threadId % blockDim_.x;
	threadIdx_.y = threadId / blockDim_.x;

	float sum = 0.0f;

#ifndef MATMUL_USE_SHARED
	int ia = (blockDim_.y * blockIdx_.y + threadIdx_.y) * n;
	int ib = blockDim_.x * blockIdx_.x + threadIdx_.x;
	int ic = ia + ib;

	// Multiply two matrices
	for (int k = 0; k < n; k++)
		sum += A [ia + k] * B [ib + k * n];
#else
	// Base indexes inside A and B
	int ia = (blockDim_.y * blockIdx_.y) * n;
	int ib = blockDim_.x * blockIdx_.x;

	// Subindex inside a "tile"
	int tileidx = n * threadIdx_.y + threadIdx_.x;

	// Index in C
	int ic = ia + ib + tileidx;

	// Shared memory for the "tile" sub-matrix of A and B
	float* As = (float*)shared;
	float* Bs = (float*)shared + blockDim_.x * blockDim_.y;

	// Go through "tiles" of size blockDim.x * blockDim.y
	for (uint aoff = 0, boff = 0; aoff < n; aoff += blockDim_.x, boff += blockDim_.y * n)
	{
		// Load the "tile" matrices from global memory to shared memory
		As [threadIdx_.y * blockDim_.x + threadIdx_.x] = A [ia + aoff + tileidx];
		Bs [threadIdx_.y * blockDim_.x + threadIdx_.x] = B [ib + boff + tileidx];

		// Synchronize to make sure the matrices are loaded
		Context::sync();

		// Multiply the two matrices
		for (int k = 0; k < blockDim_.x; k++)
			sum += As [threadIdx_.y * blockDim_.x + k] * Bs [k * blockDim_.y + threadIdx_.x];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		Context::sync();
	}
#endif
	// Write the block sub-matrix to global memory
	// each thread writes one element
	C [ic] = sum;
	*/
}
#endif

class MatmulTask : public ::Procedure
{
public:
	static const int NumThreads = BLOCK_SIZE * BLOCK_SIZE;
	static const bool ItemInput = false; // false results in a lvl 1	task
	static const int sharedMemory = 2 * sizeof(float) * NumThreads;	// shared memory requirements 
	
	typedef uint ExpectedData;

	#ifdef OPENCL_CODE
	template<class Q, class Context>
	static /*__device__*/ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* ptaskid, volatile uint* shared) 
	{
		whippletree_matmul(threadId, numThreads, ptaskid, shared);
	}

	template<class Q>
	/*__device__*/ __inline__ static void init(Q* q, int id)
	{
		q->template enqueueInitial<MatmulTask>(id);
	}
	#endif
	
};

enum MatmulVersion
{
	CUBLAS,
	CUDA,
	WHIPPLETREE,
	TASMAN
};


class Matmul
{
public :
	//lets use a dist locks queue for each procedure, which can hold 12k elements
	template<class ProcInfo>
	class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 96 * 1024, false>::Type<ProcInfo> { };

	//and lets use a Megakernel which can execute multiple workpackages concurrently (dynamic)
	//and offers a maximum of 16k shared memory
	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<MatmulTask> > MyTechnique;

	Matmul(float* Ah, float* Bh, float* Ch, size_t n, MatmulVersion version, float* time = NULL)
	{
		MatmulConfig hconfig;
		cl_mem& A = hconfig.A;
		cl_mem& B = hconfig.B;
		cl_mem& C = hconfig.C;
		hconfig.n = n;
		cl_int status;
		
		
		config=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(MatmulConfig), NULL, &status);
		CL_CHECKED_CALL(status);
		A=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, &status);
		CL_CHECKED_CALL(status);
		B=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, &status);
		CL_CHECKED_CALL(status);
		C=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, &status);
		CL_CHECKED_CALL(status);


		cmdQueue = clCreateCommandQueue(context, devices[used_cl_device], 0, &status);
				
	
		CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, A, CL_TRUE, 0, sizeof(float) * n * n, &Ah, 0, NULL, NULL));
		CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, B, CL_TRUE, 0, sizeof(float) * n * n, &Bh, 0, NULL, NULL));
		CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, C, CL_TRUE, 0, sizeof(float) * n * n, &Ch, 0, NULL, NULL));
		
		if (version == MatmulVersion::WHIPPLETREE)
		{
			hconfig.blockDim_.x = BLOCK_SIZE;
			hconfig.blockDim_.y = BLOCK_SIZE;
			hconfig.gridDim_.x = n / hconfig.blockDim_.x;
			hconfig.gridDim_.y = n / hconfig.blockDim_.y;

			CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, config, CL_TRUE, 0, sizeof(MatmulConfig), &hconfig, 0, NULL, NULL));
			//*CUDA_CHECKED_CALL(cudaMemcpyToSymbol(config, &hconfig, sizeof(MatmulConfig)));

			MyTechnique technique;
			technique.init();

			technique.insertIntoQueue<MatmulTask>(hconfig.gridDim_.x * hconfig.gridDim_.y);

			volatile struct timespec start;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&start);

			technique.execute(0);
			//*CUDA_CHECKED_CALL(cudaDeviceSynchronize());

			volatile struct timespec finish;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&finish);

			if (time)
				*time = (float)((double)0.000000001 * (finish.tv_nsec - start.tv_nsec) +
					finish.tv_sec - start.tv_sec);
		}
		if (version == MatmulVersion::TASMAN)
		{
			// Dynamic task manager has its own fixed block size of 32
			hconfig.blockDim_.x = 8;
			hconfig.blockDim_.y = 4;
			hconfig.gridDim_.x = n / hconfig.blockDim_.x;
			hconfig.gridDim_.y = n / hconfig.blockDim_.y;

			CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, config, CL_TRUE, 0, sizeof(MatmulConfig), &hconfig, 0, NULL, NULL));
			//*CUDA_CHECKED_CALL(cudaMemcpyToSymbol(config, &hconfig, sizeof(MatmulConfig)));

			using namespace std;
			using namespace tasman;
		
			// Register dynamic tasks.
			// XXX Note: all dynamic tasks must be registered BEFORE
			// starting dynamic task manager.
			
			unique_ptr<DynamicTask> task(DynamicTask::Create<whippletree_matmul>());

			// Get dynamic task manager instance (unique singleton atm).
			DynamicTaskManager& dtm = DynamicTaskManager::get();
			
			int ntasks = hconfig.gridDim_.x * hconfig.gridDim_.y;
	
			// Create sample data for the given number of tasks.
			// XXX Note: all device memory allocations must happen BEFORE
			// starting dynamic task manager.
			uint* hindexes = new uint[ntasks];
			for (int i = 0; i < ntasks; i++)
				hindexes[i] = i;
			cl_mem dindexes = NULL;
			float *host_dindexes=NULL;
			host_dindexes=(float*)malloc(sizeof(uint)*ntasks);
						
			dindexes=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint) * ntasks, NULL, &status);
			CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, dindexes, CL_TRUE, 0, sizeof(uint) * ntasks, &hindexes, 0, NULL, NULL));
			//*CUDA_CHECKED_CALL(cudaMalloc(&dindexes, sizeof(uint) * ntasks));
			//*CUDA_CHECKED_CALL(cudaMemcpy(dindexes, hindexes, sizeof(uint) * ntasks, cudaMemcpyHostToDevice));

			volatile struct timespec start;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&start);

			// Launch dynamic task manager (that is, it will be resident in
			// GPU until stopped).
			dtm.start();

			// Dynamically add tasks into task manager.
			for (int i = 0; i < ntasks; i++)
				dtm.enqueue(task.get(), host_dindexes,dindexes,i,ntasks);

			// Signal dynamic task manager to shutdown (after all tasks
			// are done).
			dtm.stop();
			//*CUDA_CHECKED_CALL(cudaDeviceSynchronize());

			volatile struct timespec finish;
			clock_gettime(CLOCK_REALTIME, (struct timespec*)&finish);

			if (time)
				*time = (float)((double)0.000000001 * (finish.tv_nsec - start.tv_nsec) +
					finish.tv_sec - start.tv_sec);

			CL_CHECKED_CALL(clReleaseMemObject(dindexes));
			delete[] hindexes;
		}

		CL_CHECKED_CALL(clEnqueueReadBuffer(cmdQueue, C , CL_TRUE, 0, sizeof(float) * n * n, Ch, 0, NULL, NULL));
		//*CUDA_CHECKED_CALL(cudaMemcpy(Ch, C, sizeof(float) * n * n, cudaMemcpyDeviceToHost));
		clErrchk(clReleaseCommandQueue(cmdQueue));

		CL_CHECKED_CALL(clReleaseMemObject(A));
		CL_CHECKED_CALL(clReleaseMemObject(B));
		CL_CHECKED_CALL(clReleaseMemObject(C));
	}
};

int main(int argc, char** argv)
{

	cl_int status;
	using namespace std;

	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " <n>" << endl;
		return 1;
	}

	/*int count;
	CUDA_CHECKED_CALL(cudaGetDeviceCount(&count));
	if (!count)
	{
		cerr << "No CUDA devices available" << endl;
		return -1;
	}
	cudaDeviceProp deviceProp;
	CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, 0));
	cout << "Using device: " << deviceProp.name << endl;
*/

//================================================================================================
//MyCL Initialization

	used_cl_device = 0;

    //Initializing platform
    clErrchk(clGetPlatformIDs(0, NULL, &numPlatforms));
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    clErrchk(clGetPlatformIDs(numPlatforms, platforms,NULL));
    

    //Initialize device
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));
	if (!numDevices)
    {
       std::cout << "No CL devices available" << std::endl;
       return -1;
    }
	if (numDevices<=used_cl_device)
    {
       std::cout << "No such CL device ID" << std::endl;
       return -1;
    }

    //Creating context
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	clErrchk(status);	
	kernels=(cl_kernel*)malloc(10*sizeof(cl_kernel));

	//compiling OPENCL_CODE
	compile_device_code();
//================================================================================================	
	
	size_t n = (size_t)strtoull(argv[1], NULL, 0);
	if (n % BLOCK_SIZE)
	{
		cerr << "For simplisity, we require n (" << n <<
			") to be exact multiplier of BLOCK_SIZE (" <<
			std::to_string(static_cast<long long>(BLOCK_SIZE)) << ")" << endl;
		return -1;
	}

	float *A = new float[n * n];
	float *B = new float[n * n];
	float *C1 = new float[n * n], *C2 = new float[n * n], *C3 = new float[n * n], *C4 = new float[n * n];

	// Generate random input matrices.
	double dinvrandmax = (double)1.0 / RAND_MAX;
	srand(time(NULL));
	for (size_t i = 0, length = n * n; i < length; i++)
	{
		A[i] = rand() * dinvrandmax;
		B[i] = rand() * dinvrandmax;
	}
	//memset(C1, 0, sizeof(float) * n * n);
	//memset(C2, 0, sizeof(float) * n * n);
	//memset(C3, 0, sizeof(float) * n * n);
	//memset(C4, 0, sizeof(float) * n * n);

	float time;
	
	Matmul(A, B, C3, n, MatmulVersion::WHIPPLETREE, &time);
	cout << "WHIPPLETREE version completed in " << time << " sec" << endl;

	//Matmul(A, B, C4, n, MatmulVersion::TASMAN, &time);
	cout << "TASMAN      version completed in " << time << " sec" << endl;

	/*
	// Compare C1 and C4 results.
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < n; i++)
		{
			float c1 = C1[i + j * n];
			float c4 = C4[i * n + j];
			if (fabsf(c1 - c4) > 0.1f)
			{
				cerr << "Mismatching C4 result @ [" << i << "][" << j << "]: " << c1 << " != " << c4 << endl;
				status = -1;
				break;
			}
		}
		if (status == -1) break;
	}
*/
	delete[] A;
	delete[] B;
	delete[] C1; delete[] C2; delete[] C3; delete[] C4;

	return status;
}

