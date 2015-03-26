#include <tools/utils.h>

#include "queueDistLocks.h"
#include "queueShared.h"
#include "queuingPerProc.h"
#include "techniqueMegakernel.h"
//#include "techniqueKernels.h"

#include "procedureInterface.h"
#include "procinfoTemplate.h"

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

namespace tasman
{
	/*__device__*/ cl_mem submission;
	/*__device__*/ cl_mem finish;
}

namespace
{
	class Task : public ::Procedure
	{
	public:
		static const int NumThreads = 32;
		static const bool ItemInput = false; // false results in a lvl 1 task
		static const int sharedMemory = 0; // shared memory requirements 

		typedef DynamicTaskInfo ExpectedData;
	
		#ifdef OPENCL_CODE
		template<class Q, class Context>
		static /*__device__*/ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, volatile uint* shared)
		{
			// Execute given task with the given argument.
			DynamicTaskInfo* info = (DynamicTaskInfo*)data;
			info->func(threadId, numThreads, info->data, shared);
		}

		template<class Q>
		
		/*__device__*/ __inline__ static void init(Q* q, int id)
		{
			// Not supposed to have any initial queue.
			__trap();
		}
		#endif
	};

	// Lets use a dist locks queue for each procedure, which can hold 96k elements
	typedef PerProcedureQueueTyping<QueueDistLocksOpt_t, 96 * 1024, false> TQueue;

	template<class ProcInfo>
	class MyQueue : public TQueue::Type<ProcInfo>
	{
	public :
		static const int globalMaintainMinThreads = 1;
		#ifdef OPENCL_CODE
		__inline__ /*__device__*/ void globalMaintain()
		{
			if (get_local_id(0) == 0)
			{
				if (submission)
				{
					TQueue::Type<ProcInfo>::template enqueue<Task>(*submission);
					submission = NULL;
				}
			}			 
		}
		#endif
	};

	typedef Megakernel::SimplePointed16336<MyQueue, ProcInfo<Task>, void, Megakernel::ShutdownIndicator> MyTechnique;

	class DynamicTaskManagerInternal : private NonCopyable<DynamicTaskManagerInternal>
	{
		cl_command_queue stream1, stream2;
		int* address;
	
		MyTechnique technique;

		DynamicTaskManagerInternal();

	public :

		static DynamicTaskManagerInternal& get();

		void start(cl_command_queue stream);
	};

	DynamicTaskManagerInternal::DynamicTaskManagerInternal()
	{
		// Determine address of finishing marker to supply it into
		// the technique.
		//*CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&address, finish));
	}

	DynamicTaskManagerInternal& DynamicTaskManagerInternal::get()
	{
		static DynamicTaskManagerInternal dtmi;
		return dtmi;
	}

	void DynamicTaskManagerInternal::start(cl_command_queue stream)
	{
		// Start megakernel in a dedicated stream.
		technique.init();
		technique.execute(0, stream, address);
	}
}

namespace tasman
{
	extern "C" void dynamicTaskManagerStart(cl_command_queue stream)
	{
		DynamicTaskManagerInternal::get().start(stream);
	}
}

