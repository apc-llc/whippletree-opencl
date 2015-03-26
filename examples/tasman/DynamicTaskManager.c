#include "DynamicTaskManager.h"

extern cl_context context;
extern cl_device_id *devices;
extern cl_command_queue cmdQueue;
extern cl_kernel * kernels;
extern cl_uint numDevices;
extern int used_cl_device;
extern cl_uint numPlatforms;
extern cl_platform_id *platforms;
extern cl_program program;

extern "C" void dynamicTaskManagerStart(cl_command_queue stream);

namespace tasman
{
	DynamicTask::~DynamicTask()
	{
		CL_CHECKED_CALL(clReleaseMemObject(info));
	}

	DynamicTaskManager::DynamicTaskManager() : started(false)
	{
		cl_int status;
		// Create two streams: one for megakernel, and another one -
		// for finish indicator.
		stream1 = clCreateCommandQueue(context, devices[used_cl_device], 0, &status);
		stream2 = clCreateCommandQueue(context, devices[used_cl_device], 0, &status);

	
		// Determine address of finishing marker to supply it into
		// the technique.
		//*CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	DynamicTaskManager::~DynamicTaskManager()
	{
		// Destroy streams.
		clErrchk(clReleaseCommandQueue(stream1));
		clErrchk(clReleaseCommandQueue(stream2));
	}

	DynamicTaskManager& DynamicTaskManager::get()
	{
		static DynamicTaskManager dtm;
		return dtm;
	}

	void DynamicTaskManager::start()
	{
		if (started) return;
		started = true;

		// Initialize finishing marker with "false" to make uberkernel
		// to run infinitely.
		int value = 0;
		CL_CHECKED_CALL(clEnqueueWriteBuffer(stream2, finish, CL_FALSE, 0, sizeof(int), &value, 0, NULL, NULL));
		CL_CHECKED_CALL(clFinish(stream2));	
		dynamicTaskManagerStart(stream1);
	}

	void DynamicTaskManager::stop()
	{
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CL_CHECKED_CALL(clEnqueueReadBuffer(stream2, submission , CL_TRUE, 0, sizeof(void*), &busy, 0, NULL, NULL));
			CL_CHECKED_CALL(clFinish(stream2));
			if (!busy) break;
		}

		// Signal shut down to uberkernel.
		int value = 1;
		CL_CHECKED_CALL(clEnqueueWriteBuffer(stream2, finish, CL_FALSE, 0, sizeof(int), &value, 0, NULL, NULL));
		CL_CHECKED_CALL(clFinish(stream2));
	
		// Wait for uberkernel to finish.
		CL_CHECKED_CALL(clFinish(stream1));
	
		started = false;
	}

	void DynamicTaskManager::enqueue(const DynamicTask* task, void* host_data, cl_mem data, int id,int ntasks) const
	{	
		// Wait until queue gets empty.
		while (true)
		{
			DynamicTaskInfo* busy = NULL;
			CL_CHECKED_CALL(clEnqueueReadBuffer(stream2, submission , CL_TRUE, 0, sizeof(void*), busy, 0, NULL, NULL));
			CL_CHECKED_CALL(clFinish(stream2));
			if (!busy) break;
		}

		// Copy data to device memory.
		//CL_CHECKED_CALL(clEnqueueWriteBuffer(stream2, task->info->data, CL_FALSE, 0, sizeof(void*)*ntasks, data, 0, NULL, NULL));
		
		// Submit task into queue.
		//CL_CHECKED_CALL(clEnqueueWriteBuffer(stream2, submission, CL_FALSE, 0, sizeof(DynamicTaskInfo*), &task->info, 0, NULL, NULL));
		CL_CHECKED_CALL(clFinish(stream2));
	}
}

