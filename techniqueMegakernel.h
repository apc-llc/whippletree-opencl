//  Project Whippletree
//  http://www.icg.tugraz.at/project/parallel
//
//  Copyright (C) 2014 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
//              Michael Kenzel - kenzel ( at ) icg.tugraz.at
//              Pedro Boechat - boechat ( at ) icg.tugraz.at
//              Bernhard Kerbl - kerbl ( at ) icg.tugraz.at
//              Mark Dokter - dokter ( at ) icg.tugraz.at
//              Dieter Schmalstieg - schmalstieg ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

//#pragma once commented out for a while (to compile separately)
#pragma once
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#ifdef OPENCL_CODE
#include "../../commonDefinitions.h"
#endif

#ifndef OPENCL_CODE
#include <memory>
#include <vector>
#include <CL/cl.h>
#include "tools/utils.h"
#include "tools/cl_memory.h"
#include <iostream>
#include "timing.h"
#include "techniqueInterface.h"

extern cl_context context;
extern cl_device_id *devices;
extern cl_command_queue cmdQueue;
extern cl_kernel * kernels;
#endif

#include "delay.h"
//#include "procinfoTemplate.h"
#include "queuingMultiPhase.h"
#include "techniqueMegakernelVars.h"


namespace SegmentedStorage
{
	void checkReinitStorage();
}

namespace Megakernel
{

  enum MegakernelStopCriteria
  {
    // Stop megakernel, when the task queue is empty.
    EmptyQueue,

    // Stop megakernel, when the task queue is empty,
    // and "shutdown" indicator is filled with "true" value.
    ShutdownIndicator,
  };
#ifdef OPENCL_CODE
//commented out because no nonconstant global scope variables available in OpenCL
//  extern  volatile int doneCounter;
//  extern  volatile int endCounter;

  template<class InitProc, class Q>
  __kernel void initData(Q* q, int num)
  {
    int id = get_global_id(0);
    for( ; id < num; id += get_global_size(0))
    {
      InitProc::template init<Q>(q, id);
    }
  }

  template<class Q>
  __kernel void recordData(Q* q)
  {
    q->record();
  }
  template<class Q>
  __kernel void resetData(Q* q)
  {
    q->reset();
  }
#endif

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool Itemized,  bool MultiElement>
  class FuncCaller;


  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, false>
  {
  public:
     #ifdef OPENCL_CODE
     __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      int nThreads;
      if(PROC::NumThreads != 0)
        nThreads = PROC::NumThreads;
      else
        nThreads = get_local_size(0);
      if(PROC::NumThreads == 0 || get_local_id(0) < nThreads)
        PROC :: template execute<Q, Context<PROC::NumThreads, false, CUSTOM> >(get_local_id(0), nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
    #endif
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, true>
  {
  public:
	#ifdef OPENCL_CODE
     __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      
      if(PROC::NumThreads != 0)
      {
        int nThreads;
        nThreads = PROC::NumThreads;
        int tid = get_local_id(0) % PROC::NumThreads;
        int offset = get_local_id(0) / PROC::NumThreads;
        if(get_local_id(0) < hasData)
          PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(tid, nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared + offset*PROC::sharedMemory/sizeof(uint) );
      }
      else
      {
        PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(get_local_id(0), get_local_size(0), queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
      }
      
    }
    #endif
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool MultiElement>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, true, MultiElement>
  {
  public:
  	#ifdef OPENCL_CODE
     __inline__
    static void call(Q* queue, void* data, int numData, uint* shared)
    {
      if(get_local_id(0) < numData)
        PROC :: template execute<Q, Context<PROC::NumThreads, MultiElement, CUSTOM> >(get_local_id(0), numData, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
    #endif
  };

  
  ////////////////////////////////////////////////////////////////////////////////////////
  
  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallCopyVisitor
  {
    int* execproc;
    #ifdef OPENCL_CODE
    const uint4 & sharedMem;
	#else
    const cl_uint4 & sharedMem;
	#endif
	
    Q* q;
    void* execData;
    uint* s_data;
	 int hasResult;
	#ifdef OPENCL_CODE
    __inline__  ProcCallCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data) { }
    template<class TProcedure, class CUSTOM>
     __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          return true;
      }
      return false;
    }
    #endif
  };

  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallNoCopyVisitor
  {
    int* execproc;
	#ifdef OPENCL_CODE
    const uint4 & sharedMem;
    #else
    const cl_uint4 & sharedMem;
    #endif
    Q* q;
    void* execData;
    uint* s_data;
    int hasResult;
	#ifdef OPENCL_CODE
    __inline__  ProcCallNoCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }
    template<class TProcedure, class CUSTOM>
     __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1);
          q-> template finishRead<TProcedure>(execproc[1],  n);
          return true;
      }
      return false;
    }
    #endif
  };

	#ifdef OPENCL_CODE
  #define PROCCALLNOCOPYPART(LAUNCHNUM) \
  template<class Q, class ProcInfo, bool MultiElement> \
  struct ProcCallNoCopyVisitorPart ## LAUNCHNUM \
  { \
    int* execproc; \
    const uint4 & sharedMem; \
    Q* q; \
    void* execData; \
    uint* s_data; \
    int hasResult; \
    __inline__  ProcCallNoCopyVisitorPart ## LAUNCHNUM  (Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }  \
    template<class TProcedure, class CUSTOM>  \
     __inline__ bool visit()  \
    {  \
      if(*execproc == TProcedure::ProcedureId)  \
      {  \
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );   \
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1); \
          q-> template finishRead ## LAUNCHNUM  <TProcedure>(execproc[1],  n);  \
          return true;  \
      }  \
      return false;   \
    }   \
  };

  PROCCALLNOCOPYPART(1)
  PROCCALLNOCOPYPART(2)
  PROCCALLNOCOPYPART(3)

#undef PROCCALLNOCOPYPART
#endif
	//commented out because no nonconstant global scope variables available in OpenCL
	//  extern  int maxConcurrentBlocks;
	//  extern  volatile int maxConcurrentBlockEvalDone;


  template<class Q, MegakernelStopCriteria StopCriteria, bool Maintainer>
  class MaintainerCaller;

  template<class Q, MegakernelStopCriteria StopCriteria>
  class MaintainerCaller<Q, StopCriteria, true>
  {
  public:
    #ifdef OPENCL_CODE
    static __inline__  bool RunMaintainer(Q* q, int* shutdown, __global globalvarsT * globalvars)
    {
      
      if(get_group_id(0) == 1)
      {
        //__local 
        bool run;
        run = true;
        barrier(CLK_LOCAL_MEM_FENCE);
        int runs = 0;
        while(run)
        {
          q->globalMaintain();
          barrier(CLK_LOCAL_MEM_FENCE);
          if(runs > 10)
          {
            if(globalvars->endCounter == 0)
            {
              if(StopCriteria == EmptyQueue)
                run = false;
              else if (shutdown)
              {
                if(*shutdown)
                  run = false;
             }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
          }
          else
            ++runs;
        }
      }
      return false;
    }
    #endif
  };
  template<class Q, MegakernelStopCriteria StopCriteria>
  class MaintainerCaller<Q, StopCriteria, false>
  {
  public:
	#ifdef OPENCL_CODE
    static __inline__  bool RunMaintainer(Q* q, int* shutdown, __global globalvarsT * globalvars)
    {
      return false;
    }
    #endif
  };

  template<class Q, class PROCINFO, class CUSTOM, bool CopyToShared, bool MultiElement, bool tripleCall>
  class MegakernelLogics;

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement, bool tripleCall>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, true, MultiElement, tripleCall>
  {
  public:
	#ifdef OPENCL_CODE
    static   __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __local uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeue<MultiElement> (execData, execproc, sizeof(uint)*(sharedMemDist.y + sharedMemDist.z));
      
      barrier(CLK_LOCAL_MEM_FENCE);

      if(hasResult)
      {
        ProcCallCopyVisitor<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
    #endif
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, false>
  {
  public:
  #ifdef OPENCL_CODE
    static   __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __local uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeueStartRead<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      barrier(CLK_LOCAL_MEM_FENCE);

      if(hasResult)
      {
        ProcCallNoCopyVisitor<Q, PROCINFO,  MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
    #endif
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, true>
  {
  public:
	#ifdef OPENCL_CODE
    static   __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __local uint s_data[];
      void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x + sharedMemDist.w);
      int* execproc = reinterpret_cast<int*>(s_data + sharedMemDist.w);

      int hasResult = q-> template dequeueStartRead1<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> >(visitor);      
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead2<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
     
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> >(visitor);          
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead3<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> >(visitor);         
      }

      return hasResult;
    }
    #endif
  };

  template<ulong StaticLimit, bool Dynamic>
  struct TimeLimiter;

  template<>
  struct TimeLimiter<0, false>
  {
     __inline__ TimeLimiter() { }
     __inline__ bool stop(int tval)
    {
      return false;
    }
  };

//commented out because no clock() functions available in opencl

  template<ulong StaticLimit>
  struct TimeLimiter<StaticLimit, false>
  {
    ulong  TimeLimiter_start;
    #ifdef OPENCL_CODE
    __inline__ TimeLimiter() 
	    {
      if(get_local_id(0) == 0)
        TimeLimiter_start = 0;//clock64();
    }
    __inline__ bool stop(int tval)
    {
      return 0;//(clock64() - TimeLimiter_start) > StaticLimit;
    }
    #endif
  };

  template<>
  struct TimeLimiter<0, true>
  {
    ulong TimeLimiter_start;
	#ifdef OPENCL_CODE
   __inline__ TimeLimiter() 
    {
      if(get_local_id(0) == 0)
        TimeLimiter_start = 0;//clock64();
    }
    __inline__ bool stop(int tval)
    {
      return 0;//(clock64() - TimeLimiter_start)/1024 > tval;
    }
    #endif
  };

}

#ifdef OPENCL_CODE
 template<class Q, class PROCINFO, class CUSTOM, class CopyToShared, class MultiElement, bool Maintainer, class TimeLimiter, Megakernel::MegakernelStopCriteria StopCriteria>
  __kernel void megakernel(Q* q, uint4 sharedMemDist, int t, int* shutdown,volatile __global Megakernel::globalvarsT * globalvars)  
  {  
    if(q == 0)
    {
      if(globalvars->maxConcurrentBlockEvalDone != 0)
        return;
      if(get_local_id(0) == 0)
        atom_add(&(globalvars->maxConcurrentBlocks), 1);
      DelayFMADS<10000,4>::delay();
      barrier(CLK_LOCAL_MEM_FENCE);
      globalvars->maxConcurrentBlockEvalDone = 1;
		write_mem_fence(CLK_GLOBAL_MEM_FENCE);
		//__threadfence();
      return;
    }
    
    __local volatile int runState;

    if(Megakernel::MaintainerCaller<Q, StopCriteria, Maintainer>::RunMaintainer(q, shutdown, globalvars))
      return;

    __local TimeLimiter timelimiter;

    if(get_local_id(0) == 0)
    {
      if(globalvars->endCounter == 0)
        runState = 0;
      else
      {
       atom_add(&globalvars->doneCounter,1);
        if(atom_add(&globalvars->endCounter,1) == 2597)
          atom_sub(&globalvars->endCounter, 2597);
        runState = 1;
      }
    }
    q->workerStart();
    barrier(CLK_LOCAL_MEM_FENCE);
	
    while(runState)
    {
      int hasResult = 0;//MegakernelLogics<Q, PROCINFO, CUSTOM, CopyToShared, MultiElement, Q::needTripleCall>::run(q, sharedMemDist);
      if(get_local_id(0) == 0)
      {
        if(timelimiter.stop(t))
          runState = 0;
        else if(hasResult)
        {
          if(runState == 3)
          {
            //back on working
            runState = 1;
            atom_add(&globalvars->doneCounter,1);
            atom_add(&globalvars->endCounter,1);
          }
          else if(runState == 2)
          {
            //back on working
            runState = 1;
            atom_add(&globalvars->doneCounter,1);
          }
        }
        else
        {
          //RUNSTATE UPDATES
          if(runState == 1)
          {
            //first time we are out of work
            atom_sub(&globalvars->doneCounter,1);
            runState = 2;
          }
          else if(runState == 2)
          {
            if(globalvars->doneCounter == 0)
            {
              //everyone seems to be out of work -> get ready for end
              atom_sub(&globalvars->endCounter,1);
              runState = 3;
            }
          }
          else if(runState == 3)
          {
            int d = globalvars->doneCounter;
            int e = globalvars->endCounter;
            //printf("%d %d %d\n",get_group_id(0) , d, e);
            if(globalvars->doneCounter != 0)
            {
              //someone started to work again
              atom_add(&globalvars->endCounter,1);
              runState = 2;
            }
            else if(globalvars->endCounter == 0)
            {
              //everyone is really out of work
              if(StopCriteria == Megakernel::EmptyQueue)
                runState = 0;
              else if (shutdown)
              {
                if(*shutdown)
                  runState = 0;
              }
            }
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      q->workerMaintain();
    }
    q->workerEnd();
  }


template __attribute__((mangled_name(megakernel1))) 
__kernel void megakernel <MyQueue<TestProcInfo>, TestProcInfo, void, bool, bool, true, Megakernel::TimeLimiter<0,false>, Megakernel::EmptyQueue> (MyQueue<TestProcInfo> * q, uint4 sharedMemDist, int t, int* shutdown, volatile __global Megakernel::globalvarsT * globalvars);

#endif

namespace Megakernel {
#ifndef OPENCL_CODE
/*--------------------------------------------------------------------------------------------------------------------------------------*/

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit  = false, bool DynamicTimelimit = false>
  class TechniqueCore
  {
    friend struct InitPhaseVisitor;
  public:

    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  protected:    
    
    std::unique_ptr<cl_mem, cuda_deleter> q;

    int blockSize[PROCINFO::NumPhases];
    int blocks[PROCINFO::NumPhases];
    cl_uint4 sharedMem[PROCINFO::NumPhases];
    uint sharedMemSum[PROCINFO::NumPhases];

    int freq;

    struct InitPhaseVisitor
    {
      TechniqueCore &technique;
      InitPhaseVisitor(TechniqueCore &technique) : technique(technique) { }
      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        technique.blockSize[Phase] = TProcInfo:: template OptimalThreadCount<MultiElement>::Num;
        
        int temp1,temp2;
        temp1=technique.blockSize[Phase];
        temp2=TQueue::globalMaintainMinThreads;
        if(TQueue::globalMaintainMinThreads > 0)
         technique.blockSize[Phase] = std::max(temp1,temp2);

        uint queueSharedMem = TQueue::requiredShared;

        //get shared memory requirement
        technique.sharedMem[Phase] = TProcInfo:: template requiredShared<MultiElement>(technique.blockSize[Phase], LoadToShared, maxShared - queueSharedMem, false);
        //if(!LoadToShared)
        //  sharedMem.x = 16;
        technique.sharedMem[Phase].s[0] /= 4;
        technique.sharedMem[Phase].s[1] = technique.sharedMem[Phase].s[1]/4;
        technique.sharedMem[Phase].s[2] = technique.sharedMem[Phase].s[2]/4;
     
        //x .. procids
        //y .. data
        //z .. shared mem for procedures
        //w .. sum


        //w ... -> shared mem for queues...
        technique.sharedMemSum[Phase] = technique.sharedMem[Phase].s[3] + queueSharedMem;
        technique.sharedMem[Phase].s[3] = queueSharedMem/4;

        temp1=technique.sharedMemSum[Phase];
        temp2=TQueue::globalMaintainSharedMemory(technique.blockSize[Phase]);

        if(TQueue::globalMaintainMinThreads > 0)
          technique.sharedMemSum[Phase] = std::max(temp1,temp2);

        //get number of blocks to start - gk110 screwes with mutices...
		cl_mem dev_globalvars;
		globalvarsT host_globalvars;
		cl_int status;
        dev_globalvars = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(globalvarsT), NULL, &status);
		CL_CHECKED_CALL(status);
        host_globalvars.maxConcurrentBlockEvalDone=0;
		CL_CHECKED_CALL(clEnqueueWriteBuffer(cmdQueue, dev_globalvars, CL_TRUE, 0, sizeof(globalvarsT), &host_globalvars, 0, NULL, NULL));
        
        //CL_CHECKED_CALL(cudaMemcpyToSymbol(maxConcurrentBlockEvalDone, &nblocks, sizeof(int)));
		        
		//1megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<StaticTimelimit?1000:0, DynamicTimelimit>, MegakernelStopCriteria::EmptyQueue> <<<512, technique.blockSize[Phase], technique.sharedMemSum[Phase]>>> (0, technique.sharedMem[Phase], 0, NULL);


        //CL_CHECKED_CALL(cudaMemcpyFromSymbol(&nblocks, maxConcurrentBlocks, sizeof(int)));
		CL_CHECKED_CALL(clEnqueueReadBuffer(cmdQueue, dev_globalvars , CL_TRUE, 0, sizeof(globalvarsT), &host_globalvars, 0, NULL, NULL));
        technique.blocks[Phase] = host_globalvars.maxConcurrentBlocks;
        std::cout << "blocks: " << technique.blocks << std::endl;
        if(technique.blocks[Phase]  == 0)
          printf("ERROR: in Megakernel confguration: dummy launch failed. Check shared memory consumption\n");
        return false;
      }
    };


    void preCall(cl_command_queue cmdQueue)
    {
      int magic = 2597, null = 0;
      //CL_CHECKED_CALL(cudaMemcpyToSymbolAsync(doneCounter, &null, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
      //CL_CHECKED_CALL(cudaMemcpyToSymbolAsync(endCounter, &magic, sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    }

    void postCall(cl_command_queue cmdQueue)
    {
    }

  public:

    void init()
    {
      q = std::unique_ptr<cl_mem, cuda_deleter>(cudaAlloc<Q>());

      int magic = 2597, null = 0;
      //CL_CHECKED_CALL(cudaMemcpyToSymbol(doneCounter, &null, sizeof(int)));
      //CL_CHECKED_CALL(cudaMemcpyToSymbol(endCounter, &magic, sizeof(int)));

      SegmentedStorage::checkReinitStorage();
      //initQueue<Q> <<<512, 512>>>(q.get());
      //CL_CHECKED_CALL(clFlush());


      InitPhaseVisitor v(*this);
      Q::template staticVisit<InitPhaseVisitor>(v);

      //cudaDeviceProp props;
      int dev;
      //CL_CHECKED_CALL(cudaGetDevice(&dev));
      //CL_CHECKED_CALL(cudaGetDeviceProperties(&props, dev));
      //freq = static_cast<int>(static_cast<unsigned long long>(props.clockRate)*1000/1024);
    }

    void resetQueue()
    {
      init();
    }

    void recordQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::recordQueue(): queue does not support reuse init\n";
      else
      {
        //recordData<Q><<<1, 1>>>(q.get());
        //CL_CHECKED_CALL(clFlush());
      }
    }

    void restoreQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::restoreQueue(): queue does not support reuse init\n";
      //else
        //resetData<Q><<<1, 1>>>(q.get());
    }


    template<class InsertFunc>
    void insertIntoQueue(int num)
    {
      typedef CurrentMultiphaseQueue<Q, 0> Phase0Q;


      //Phase0Q::pStart();

      //Phase0Q::CurrentPhaseProcInfo::print();

      //int b = min((num + 512 - 1)/512,104);
      //initData<InsertFunc, Phase0Q><<<b, 512>>>(reinterpret_cast<Phase0Q*>(q.get()), num);
      //CL_CHECKED_CALL(clFlush());
    }

    int BlockSize(int phase = 0) const
    {
      return blockSize[phase];
    }
    int Blocks(int phase = 0) const
    {
      return blocks[phase];
    }
    uint SharedMem(int phase = 0) const
    {
      return sharedMemSum[phase];
    }

    std::string name() const
    {
      return std::string("Megakernel") + (MultiElement?"Dynamic":"Simple") + (LoadToShared?"":"Globaldata") + ">" + Q::name();
    }

    void release()
    {
      delete this;
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit = false, bool DynamicTimelimit = false>
  class Technique;
  
  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, false, false> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, false>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      cl_uint4 sharedMem;
      Q* q;
      cl_command_queue cmdQueue;
      int* shutdown;
      //LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, cl_uint4 sharedMem, cl_command_queue cmdQueue, int* shutdown) :
        //phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), q(q), cmdQueue(cmdQueue), shutdown(shutdown) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          //megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<false,false>, StopCriteria><<<blocks, blockSize, sharedMemSum, stream>>> (reinterpret_cast<TQueue*>(q), sharedMem, 0, shutdown);
          return true;
        }
        return false;
      }
    };
  public:
    void execute(int phase = 0, cl_command_queue cmdQueue = 0, int* shutdown = NULL)
    {
      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,false> TCore;

      TCore::preCall(cmdQueue);

      //LaunchVisitor v(TCore::q.get(), phase, TCore::blocks[phase], TCore::blockSize[phase], TCore::sharedMemSum[phase], TCore::sharedMem[phase], cmdQueue, shutdown);
      //Q::template staticVisit<LaunchVisitor>(v);

      TCore::postCall(cmdQueue);
    }
  };


  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, true, false> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, true, false>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  public:
    template<int Phase, int TimeLimitInKCycles>
    void execute(cl_command_queue cmdQueue = 0, int* shutdown = NULL)
    {
      typedef CurrentMultiphaseQueue<Q, Phase> ThisQ;

      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,true,false> TCore;

      TCore::preCall(cmdQueue);

     // megakernel<ThisQ, typename ThisQ::CurrentPhaseProcInfo, ApplicationContext, LoadToShared, MultiElement, (ThisQ::globalMaintainMinThreads > 0)?true:false,TimeLimiter<TimeLimitInKCycles,false>, StopCriteria><<<TCore::blocks[Phase], TCore::blockSize[Phase], TCore::sharedMemSum[Phase], stream>>>(TCore::q.get(), TCore::sharedMem[Phase], 0, shutdown);

      TCore::postCall(cmdQueue);
    }

    template<int Phase>
    void execute(cl_command_queue cmdQueue = 0)
    {
      return 0;//execute<Phase, 0>(stream);
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, MegakernelStopCriteria StopCriteria, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE, PROCINFO, ApplicationContext, StopCriteria, maxShared, LoadToShared, MultiElement, false, true> : public TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, true>
  {
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      cl_uint4 sharedMem;
      int timeLimit;
      Q* q;
      int* shutdown;
      LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, cl_uint4 sharedMem, int timeLimit, int* shutdown) : phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), timeLimit(timeLimit), q(q), shutdown(shutdown) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          //megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false,TimeLimiter<false,true>, StopCriteria><<<blocks, blockSize, sharedMemSum>>>(reinterpret_cast<TQueue*>(q), sharedMem, timeLimit, shutdown);
          return true;
        }
        return false;
      }
    };
  public:
    void execute(int phase = 0, cl_command_queue cmdQueue = 0, double timelimitInMs = 0, int* shutdown = NULL)
    {
      typedef TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,true> TCore;

      TCore::preCall(cmdQueue);

      LaunchVisitor v(TCore::q.get(),phase, TCore::blocks[phase], TCore::blockSize[phase], TCore::sharedMemSum[phase], TCore::sharedMem[phase], timelimitInMs/1000*TCore::freq, cmdQueue, shutdown);
      Q::template staticVisit<LaunchVisitor>(v);

      TCore::postCall(cmdQueue);
    }
  };

  // convenience defines

  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class SimpleShared : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, true, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class SimplePointed : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, false, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class DynamicShared : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, true, true>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, MegakernelStopCriteria StopCriteria = EmptyQueue, int maxShared = 16336>
  class DynamicPointed : public Technique<Q, PROCINFO, CUSTOM, StopCriteria, maxShared, false, true>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimpleShared16336 : public SimpleShared<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  { };

    template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimpleShared49000: public SimpleShared<Q, PROCINFO, CUSTOM, StopCriteria, 49000>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed24576 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 24576>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed16336 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class SimplePointed12000 : public SimplePointed<Q, PROCINFO, CUSTOM, StopCriteria, 12000>
  {  };


  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicShared16336 : public DynamicShared<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed16336 : public DynamicPointed<Q, PROCINFO, CUSTOM, StopCriteria, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed12000 : public DynamicPointed<Q, PROCINFO, CUSTOM, StopCriteria, 12000>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void, MegakernelStopCriteria StopCriteria = EmptyQueue>
  class DynamicPointed11000 : public DynamicPointed<Q,  PROCINFO, CUSTOM, StopCriteria, 11000>
  {  };

#endif
}
