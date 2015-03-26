#include <cstdio>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <tools/utils.h>

extern cl_context context;
extern cl_device_id *devices;
extern cl_command_queue cmdQueue;
extern cl_kernel * kernels;
extern cl_uint numDevices;
extern int used_cl_device;
extern cl_uint numPlatforms;
extern cl_platform_id *platforms;
extern cl_program program;

namespace tasman
{
	typedef void (*DynamicTaskFunction)(int threadId, int numThreads, void* data, volatile uint* shared);

	struct DynamicTaskInfo;
}

namespace
{
	using namespace tasman;

	#ifdef OPENCL_CODE
	template<DynamicTaskFunction Func>	
	__kernel void getfuncaddress(DynamicTaskInfo* info);
	#endif
}

namespace tasman
{
	extern /*__device__*/ cl_mem submission;
	extern /*__device__*/ cl_mem finish;

	struct DynamicTaskInfo
	{
		DynamicTaskFunction func;
		void* data;
	};

	class DynamicTaskManager;

	class DynamicTask
	{
		//DynamicTaskInfo* info;
		cl_mem info;

		friend class DynamicTaskManager;

	public :

		template<DynamicTaskFunction Func>
		static DynamicTask* Create()
		{
			cl_int status;
			DynamicTask* task = new DynamicTask();

			//CUDA_CHECKED_CALL(cudaMalloc(&task->info, sizeof(DynamicTaskInfo)));
			task->info = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DynamicTaskInfo), NULL, &status);
			//CL_CHECKED_CALL(status);
			// Determine the given task function address on device.
			//*getfuncaddress<Func><<<1, 1>>>(task->info);
			
			//*CUDA_CHECKED_CALL(cudaDeviceSynchronize());
		
			return task;
		}

		~DynamicTask();
	};

	template <class T>
	class NonCopyable
	{
	protected:
		NonCopyable() { }
		~NonCopyable() { } // Protected non-virtual destructor

	private: 
		NonCopyable(const NonCopyable &);
		NonCopyable& operator=(const NonCopyable &);
	};

	class DynamicTaskManager : private NonCopyable<DynamicTaskManager>
	{
		cl_command_queue stream1, stream2;
		int* address;
		
		bool started;
	
		DynamicTaskManager();

		~DynamicTaskManager();

	public :

		static DynamicTaskManager& get();

		void start();
	
		void stop();

		void enqueue(const DynamicTask* task, void* host_data, cl_mem data, int id, int ntasks) const;
	};
}

namespace
{
	using namespace tasman;

	#ifdef OPENCL_CODE
	template<DynamicTaskFunction Func>	
	__kernel void getfuncaddress(__global DynamicTaskInfo* info)
	{
		info->func = Func;
	}
	#endif
}

