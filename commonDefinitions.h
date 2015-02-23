
//#ifdef OPENCL_CODE
#include "queueDistLocks.h"
#include "queueShared.h"
#include "queuingPerProc.h"
#include "segmentedStorage.h"
//#endif


#include "examples/queuing/proc0.h"
#include "examples/queuing/proc1.h"
#include "examples/queuing/proc2.h"


//somehow we need to get something into the queue
//the init proc does that for us
class InitProc
{
public:
  template<class Q>
  __inline__ //__device__
  static void init(Q* q, int id)
  {
    //so lets put something into the queues
    cl_int4 d;
	d.s[0] = id+1;
	d.s[0] = 0;
	d.s[0] = 1;
	d.s[0] = 2;
    q-> template enqueueInitial<Proc0>(d);
  }
};


typedef ProcInfo<Proc0,N<Proc1,N<Proc2> > >TestProcInfo;


//lets use a dist locks queue for each procedure, which can hold 12k elements
template<class ProcInfo>
class MyQueue : public PerProcedureQueueTyping<QueueDistLocksOpt_t, 12*1024, false>::Type<ProcInfo> {};


//and lets use a Megakernel which can execute multiple workpackages concurrently (dynamic)
//and offers a maximum of 16k shared memory

#ifndef OPENCL_CODE
typedef Megakernel::DynamicPointed16336<MyQueue, TestProcInfo> MyTechnique;
#endif

//typedef KernelLaunches::TechniqueMultiple<MyQueue, TestProcInfo> MyTechnique;

//typedef DynamicParallelism::TechniqueQueuedNoCopy<MyQueue, InitProc, TestProcInfo> MyTechnique;
