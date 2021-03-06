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

#pragma once
#include "queueExternalFetch.h"
#include "queueInterface.h"
#include "procedureInterface.h"
#include "procinfoTemplate.h"
#include "tools/common.h"
#include "random.h"


template<class PROCINFO, class PROCEDURE,  template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool Itemized, bool InitialQueue >
struct QueueSelector;

template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PacakgeQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, true, false> : public InternalItemQueue<sizeof(typename PROCEDURE::ExpectedData), ItemQueueSize, void>
{
	#ifdef OPENCL_CODE
  const bool Itemized = true;
	#else
  static const bool Itemized = true;
  	#endif
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;

	#ifdef OPENCL_CODE
  __inline__ /*__device__*/ void record() { }
  __inline__ /*__device__*/ void reset() { }
	#endif
};
template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, false, false> : public InternalPackageQueue<sizeof(typename PROCEDURE::ExpectedData), PackageQueueSize, void>
{
	#ifdef OPENCL_CODE
  const bool Itemized = false;
	#else
  static const bool Itemized = false;
  	#endif
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;

	#ifdef OPENCL_CODE
  __inline__ /*__device__*/ void record() { }
  __inline__ /*__device__*/ void reset() { }
  	#endif
};


template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool TItemized >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, TItemized, true> : public InitialDataQueue<sizeof(typename PROCEDURE::ExpectedData), InitialDataQueueSize, void>
{
  #ifdef OPENCL_CODE
  const bool Itemized = TItemized;
  #else
  static const bool Itemized = TItemized;
  #endif
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;
};
  




template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, bool RandomSelect = false>
class PerProcedureVersatileQueue : public ::Queue<> 
{
  
  template<class TProcedure>
  struct QueueAttachment : public QueueSelector<ProcedureInfo, TProcedure, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize,  QueueExternalFetch, 128*1024,  TProcedure::ItemInput, TProcedure::InitialProcedure >
  { };

  Attach<QueueAttachment, ProcedureInfo> queues;

  int dummy[32]; //compiler alignment mismatch hack

  template<bool MultiProcedure>
  class Visitor
  {
    uint _haveSomething;
    int*& _procId;
    void*& _data;
    const int _itemizedThreshold;
    int _maxShared;
  public:
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ Visitor(int*& procId, void*& data, int minItems, int maxShared) : 
         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
    { }
    __inline__ /*__device__*/ uint haveSomething() const
    {
      return _haveSomething;
    }
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      typedef typename TQAttachment::Procedure Procedure;
      const bool Itemized = TQAttachment::Itemized;
      
      __shared__ volatile int ssize;
      ssize = q.size();
      __syncthreads();
      int size = ssize;
      __syncthreads();
      if(size == 0) 
        return false;


      if(Itemized || MultiProcedure)
      {
        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
        if(size*itemThreadCount >= _itemizedThreshold)
        {
          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(typename Procedure::ExpectedData) + Procedure::sharedMemory)) :  min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(typename Procedure::ExpectedData)));
          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
          _haveSomething = q.dequeue(_data, nItems);
          if(threadIdx.x < _haveSomething*itemThreadCount)
          {
            _data = reinterpret_cast<char*>(_data) + sizeof(typename Procedure::ExpectedData)*(threadIdx.x/itemThreadCount);
            _haveSomething *= itemThreadCount; 
            _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
          }
          return _haveSomething > 0;
        }
        return false;
      }
      else
      {
        _haveSomething = q.dequeue(_data, 1) * (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
        _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
        return _haveSomething > 0;
      }
    }
    #endif
  };


  template<bool MultiProcedure>
  class ReadVisitor
  {
    uint _haveSomething;
    int*& _procId;
    void*& _data;
    const int _itemizedThreshold;
    int _maxShared;
  public:
    #ifdef OPENCL_CODE
    __inline__ /*__device__*/ ReadVisitor(int*& procId, void*& data, int minItems, int maxShared) : 
         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
    { }
    __inline__ /*__device__*/ uint haveSomething() const
    {
      return _haveSomething;
    }
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      typedef typename TQAttachment::Procedure Procedure;
      const bool Itemized = TQAttachment::Itemized;

      __shared__ volatile int ssize;
      ssize = q.size();
      __syncthreads();
      int size = ssize;
      __syncthreads();
      if(size == 0) 
        return false;

      if(Itemized || MultiProcedure)
      {
        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
        if(size*itemThreadCount >= _itemizedThreshold)
        {
          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / Procedure::sharedMemory) : blockDim.x/itemThreadCount;
          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
          _haveSomething = q.reserveRead(nItems);
          if(_haveSomething != 0)
          {
            int id = q.startRead(_data, threadIdx.x/itemThreadCount, _haveSomething);
            _haveSomething *= itemThreadCount; 
            _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
            _procId[1] = id;
            return true;
          }
        }
      }
      else
      {
        _haveSomething = q.reserveRead(1);
        if(_haveSomething != 0)
        {
          int id = q.startRead(_data, 0, _haveSomething);
          _haveSomething *= (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
          _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
          _procId[1] = id;
          return true;
        }
      }
      return false;
    }
    #endif
  };

  struct NameVisitor
  {
    #ifndef OPENCL_CODE
    std::string name;
    template<class Procedure>
    bool visit()
    {
      if(name.size() > 0)
        name += ",";
      name += Procedure::name();
      return false;
    }
    #endif
  };

  struct InitVisitor
  {
	#ifdef OPENCL_CODE
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(__global TQAttachment& q)
    {
      q.init();
      return false;
    }
    #endif
  };

  template<class TProcedure>
  struct EnqueueInitialVisitor
  {
    typename TProcedure::ExpectedData& data;
    bool res;
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ EnqueueInitialVisitor(typename TProcedure::ExpectedData& d) : data(d) { }
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q.template enqueueInitial<typename TProcedure::ExpectedData>(data);
      return true;
    }
    #endif
  };

  template<class TProcedure>
  struct EnqueueVisitor
  {
    typename TProcedure::ExpectedData& data;
    bool res;
    #ifdef OPENCL_CODE
    __inline__ /*__device__*/ EnqueueVisitor(typename TProcedure::ExpectedData& d) : data(d) { }
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q.template enqueue <typename TProcedure::ExpectedData>(data);
      return true;
    }
    #endif
  };

  template< int Threads, class TProcedure>
  struct EnqueueThreadsVisitor
  {
    typename TProcedure::ExpectedData* data;
    bool res;
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ EnqueueThreadsVisitor(typename TProcedure::ExpectedData* d) : data(d) { }
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q.template enqueue <Threads, typename TProcedure::ExpectedData>(data);
      return true;
    }
    #endif
  };
   
  template<bool MultiProcedure>
  struct DequeueSelectedVisitor
  {
    void*& data;
    int maxNum;
    int res;
	#ifdef OPENCLC_ODE
    __inline__ /*__device__*/ DequeueSelectedVisitor(void*& data, int maxNum) : data(data), maxNum(maxNum) { }

    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q.dequeueSelected(data, TQAttachment::ProcedureId, maxNum);
      return true;
    }
    #endif
  };

  template<class TProcedure>
  struct ReserveReadVisitor
  {
    int maxNum;
    int res;
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ ReserveReadVisitor(int maxNum) : maxNum(maxNum) { }

    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q. reserveRead (maxNum);
      return true;
    }
    #endif
  };

  template<class TProcedure>
  struct StartReadVisitor
  {
    void*& data;
    int num;
    int res;

	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ StartReadVisitor(void*& data, int num) : data(data), num(num) { }

    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      res = q . startRead  (data, getThreadOffset<TProcedure, true>(), num);
      return true;
    }
    #endif
  };

  template<class TProcedure>
  struct FinishReadVisitor
  {
    int id;
    int num;
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ FinishReadVisitor(int id, int num) : id(id), num(num) { }

    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      q . finishRead (id, num);
      return true;
    }
    #endif
  };


  struct NumEntriesVisitor
  {
    int* counts;
    int i;
	#ifdef OPENCL_CODE
    __inline__ /*__device__*/ NumEntriesVisitor(int* counts) : counts(counts), i(0) { }

    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      counts[i] = q.size();
      ++i;
      return false;
    }
    #endif
  };


  struct RecordVisitor
  {
    #ifdef OPENCL_CODE
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      q.record();
      return false;
    }
    #endif
  };

  struct ResetVisitor
  {
	#ifdef OPENCL_CODE
    template<class TQAttachment>
    __inline__ /*__device__*/ bool visit(TQAttachment& q)
    {
      q.reset();
      return false;
    }
    #endif
  };

public:

	#ifdef OPENCL_CODE
  const bool supportReuseInit = true;
	#else
  static const bool supportReuseInit = true;
	#endif

#ifndef OPENCL_CODE
  static std::string name()
  {
    //NameVisitor v;
    //ProcInfoVisitor<ProcedureInfo>::Visit<NameVisitor>(v);
    //return std::string("DistributedPerProcedure[") + v.name() + "]";
    return std::string("DistributedPerProcedure[") + InternalPackageQueue<16, PackageQueueSize, void>::name() + "," + InternalItemQueue<16, ItemQueueSize, void>::name() + "]" ;
  }
#endif

	#ifdef OPENCL_CODE
  __inline__ /*__device__*/ void init() 
  {
    InitVisitor visitor;
    queues . template VisitAll<InitVisitor>(visitor);
  }
	#endif

	#ifdef OPENCL_CODE
  template<class PROCEDURE>
  __inline__ /*__device__*/ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    EnqueueInitialVisitor<PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueInitialVisitor<PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }
	#endif
	
	#ifdef OPENCL_CODE
  template<class PROCEDURE>
  /*__device__*/ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {        
    EnqueueVisitor<PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueVisitor<PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }
	#endif
	
	#ifdef OPENCL_CODE
  template<int threads, class PROCEDURE>
  __inline__ /*__device__*/ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    EnqueueThreadsVisitor<threads, PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueThreadsVisitor<threads, PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }

  template<bool MultiProcedure>
  __inline__ /*__device__*/ int dequeue(void*& data, int*& procId, int maxShared = 100000)
  {     
    if(!RandomSelect)
    {
      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template visitAll<Visitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template visitAll<Visitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
    else
    {
      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAllRandStart<Visitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAllRandStart<Visitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }    
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ /*__device__*/ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    DequeueSelectedVisitor<MultiProcedure> visitor(data, maxNum);
    visitor.res = 0;
    queues . template VisitSpecific<DequeueSelectedVisitor<MultiProcedure> >(visitor, procId);
    return visitor.res;
  }

  template<bool MultiProcedure>
  __inline__ /*__device__*/ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
  {
    if(!RandomSelect)
    {
      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAll<ReadVisitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAll<ReadVisitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
    else
    {
      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAllRandStart<ReadVisitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAllRandStart<ReadVisitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
   
    return 0;
  }

  template<class PROCEDURE>
  __inline__ /*__device__*/ int reserveRead(int maxNum = -1)
  {
    if(maxNum == -1)
      maxNum = blockDim.x / (PROCEDURE::NumThreads>0 ? PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? 1 : blockDim.x));

    ReserveReadVisitor<PROCEDURE> visitor(maxNum);
    queues . template VisitSpecific<ReserveReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
    return visitor.res;
  }
  template<class PROCEDURE>
  __inline__ /*__device__*/ int startRead(void*& data, int num)
  {
    StartReadVisitor<PROCEDURE> visitor(data, num);
    queues . template VisitSpecific<StartReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
    return visitor.res;
  }
  template<class PROCEDURE>
  __inline__ /*__device__*/ void finishRead(int id,  int num)
  {
    FinishReadVisitor<PROCEDURE> visitor(id, num);
    queues . template VisitSpecific<FinishReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
  }

  __inline__ /*__device__*/ void numEntries(int* counts)
  { 
    NumEntriesVisitor visitor(counts);
    queues . template VisitAll<NumEntriesVisitor>(visitor);
  }

  __inline__ /*__device__*/ void record()
  {
    RecordVisitor visitor;
    queues . template VisitAll<RecordVisitor>(visitor);
  }

  __inline__ /*__device__*/ void reset()
  {
    ResetVisitor visitor;
    queues . template VisitAll<ResetVisitor>(visitor);
  }
  #endif
};





template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
class PerProcedureQueue : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect>
{
};

template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint PackageQueueSize,  uint ItemQueueSize, bool RandomSelect = false>
struct PerProcedureQueueDualSizeTyping 
{
  template<class ProcedureInfo>
  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, PackageQueueSize, InternalQueue, ItemQueueSize, RandomSelect> {}; 
};


template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
struct PerProcedureQueueTyping 
{
  template<class ProcedureInfo>
  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect> {}; 
};
