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

#ifndef OPENCL_CODE
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <tools/utils.h>
#include "comp.h"

extern cl_context context;
extern cl_device_id *devices;
#endif

//ALL CUDA HEADERS TO BE REWRITTEN

#include "queueDistLocks.h"
#include "queueShared.h"
#include "queuingPerProc.h"
#include "techniqueMegakernel.h"
//#include "techniqueKernels.cuh"
//#include "techniqueDynamicParallelism.cuh"
#include "segmentedStorage.h"

#include "proc0.h"
#include "proc1.h"
#include "proc2.h"


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

typedef Megakernel::DynamicPointed16336<MyQueue, TestProcInfo> MyTechnique;

//typedef KernelLaunches::TechniqueMultiple<MyQueue, TestProcInfo> MyTechnique;

//typedef DynamicParallelism::TechniqueQueuedNoCopy<MyQueue, InitProc, TestProcInfo> MyTechnique;

#ifndef OPENCL_CODE
void runTest(int used_cl_device)
{
	cl_int status;	
	cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[used_cl_device], 0, &status);
	clErrchk(status);

  //create everything
  MyTechnique technique;
  
  compile_device_code();
  
  technique.init();
  std::cout<<"init completed\n";
  
  technique.insertIntoQueue<InitProc>(10);
  std::cout<<"insert completed\n";
  
	cl_event event;	
	double time = 0;

	cl_ulong time_start=0, time_end=0;
	//clErrchk(clWaitForEvents(1, &event));
	//clErrchk(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL));
		

	technique.execute(0, cmdQueue);
	std::cout<<"execution completed\n";
	

	//clErrchk(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL));
	//time += (time_end - time_start)/1e+6;
	clErrchk(clReleaseCommandQueue(cmdQueue));

  printf("run completed in %fs\n", time);
}
#endif
