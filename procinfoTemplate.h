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
#ifndef OPENCL_CODE
#include <algorithm>
#endif

#include "procedureInterface.h"
#include "random.h"

#include "tools/common.h"





template<int ProcThreads, bool Warp, bool MultiElement>
class GroupOpsSelect;

// features for items ... none
template<bool MultiElement>
class GroupOpsSelect<1, true, MultiElement> {};



// features for sub warp groups
template<int ProcThreads, bool MultiElement>
class GroupOpsSelect<ProcThreads, true, MultiElement> 
{ 
	#ifdef OPENCL_CODE
	const int MaxWarps =32;
    const unsigned int Mask = ((1u << ProcThreads)-1);
	#else
  static const int MaxWarps =32;
  static const unsigned int Mask = ((1u << ProcThreads)-1);
  	#endif
public:
	#ifdef OPENCL_CODE
  static /*__device__*/ __inline__ bool any(int arg)
  {
    unsigned int ballotres = __ballot(arg);
    return ((ballotres << (Tools::laneid()/ProcThreads)) & Mask) != 0u;
  }
  static /*__device__*/ __inline__ bool all(int arg)
  {
    unsigned int ballotres = __ballot(arg);
    return (ballotres << (Tools::laneid()/ProcThreads)) == Mask;
  }
  static /*__device__*/ __inline__ unsigned int ballot(int arg)
  {
    unsigned int ballotres = __ballot(arg);
    return ((ballotres << (Tools::laneid()/ProcThreads)) & Mask);
  }
  static /*__device__*/ __inline__ int shfl(int value, int srcThread)
  {
    #if __CUDA_ARCH__ < 300
    __shared__ volatile int comm[MaxWarps];
    int runid = 0;
    int res = value;
    for(int offset = 0; offset < 32; offset += ProcThreads)
    {
      for(int within = 0; within < ProcThreads; ++within)
      {
        if(Tools::laneid() == runid)
          comm[get_local_id(0)/32] = value;
        if( Tools::laneid() >= offset 
          && Tools::laneid() < offset + ProcThreads 
          && (runid % ProcThreads) == ((srcThread + 32) % ProcThreads) )
          res = comm[get_local_id(0)/32];
        ++runid;
      }
    }
    return res;
    #else
    return __shfl(value, srcThread, ProcThreads);
    #endif
  }
#endif
};



// features for single workpackage execution
template<int ProcThreads>
class GroupOpsSelect<ProcThreads, false, false> 
{ 
public:
#ifdef OPENCL_CODE
  static /*__device__*/ __inline__ void sync()
  {
    Tools::syncthreads(1, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_count(int predicate)
  {
    return Tools::syncthreads_count(predicate, 1, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_and(int predicate)
  {
    return Tools::syncthreads_and(predicate, 1, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_or(int predicate)
  {
    return Tools::syncthreads_or(predicate, 1, ProcThreads);
  }
#endif
};

// features for multiple workpackage execution
template<int ProcThreads>
class GroupOpsSelect<ProcThreads, false, true> 
{ 
public:
#ifdef OPENCL_CODE
  static /*__device__*/ __inline__ void sync()
  {
    Tools::syncthreads(1 + get_local_id(0)/ProcThreads, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_count(int predicate)
  {
    return Tools::syncthreads_count(predicate, 1 + get_local_id(0)/ProcThreads, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_and(int predicate)
  {
    return Tools::syncthreads_and(predicate, 1 + get_local_id(0)/ProcThreads, ProcThreads);
  }
  static /*__device__*/ __inline__ int sync_or(int predicate)
  {
    return Tools::syncthreads_or(predicate, 1 + get_local_id(0)/ProcThreads, ProcThreads);
  }
#endif

};



template <int ProcThreads, bool MultiElement, class CustomType>
class Context : public GroupOpsSelect<ProcThreads, ProcThreads <= 32, MultiElement> 
{
public:
  typedef CustomType Application;
};



#ifndef OPENCL_CODE
template <int a, int b> 
struct maxOperator 
{ 
  static const int result = a > b ? a : b; 
};

template <int a, int b> 
struct minOperator 
{ 
  static const int result = a < b ? a : b; 
};

template <bool test, int a, int b>
struct switchOperator
{
  static const int result = test ? a : b;
};

template<int VAL>
struct AvoidZero
{
  static const int val = VAL;
};

template<>
struct AvoidZero<0>
{
  static const int val = 1;
};


template<class A, class B>
struct Equals
{
  static const bool result = false;
};

template<class A>
struct Equals<A,A>
{
  static const bool result = true;
};
#endif



#ifdef OPENCL_CODE
template<int Phase>
struct NoPriority
{
  const unsigned int MinPriority = 1;
  const unsigned int MaxPriority = 1;

  template<class TProc>
  __inline__ /*__device__*/ static unsigned int eval(typename TProc::ExpectedData* data)
  {
    return 1;
  }

  template<class ProcedureIdentifier>
  __inline__ /*__device__*/ static unsigned int eval(ProcedureIdentifier procIdentifier, void* data)
  {
    return 1;
  }
};
#else
template<int Phase>
struct NoPriority
{
  static const unsigned int MinPriority = 1;
  static const unsigned int MaxPriority = 1;

  template<class TProc>
  __inline__ /*__device__*/ static unsigned int eval(typename TProc::ExpectedData* data)
  {
    return 1;
  }

  template<class ProcedureIdentifier>
  __inline__ /*__device__*/ static unsigned int eval(ProcedureIdentifier procIdentifier, void* data)
  {
    return 1;
  }
};

#endif



#ifdef OPENCL_CODE
template<class Proc, int Phase>
struct AllPhasesActiveTrait
{
  const bool Active = true;
};
#else
template<class Proc, int Phase>
struct AllPhasesActiveTrait
{
  static const bool Active = true;
};
#endif

template <int ElementSize>
struct DataAlignment;

#ifndef OPENCL_CODE
class ProcInfoEnd
{
public:

  typedef NoProcedure Procedure;
  typedef ProcInfoEnd Next;


  static void print()
  {
    printf("\n");
  }

  static const int ProcedureId = 0;

  static const int MaxId = 0;
  static const int MaxDataSize = 0;
  static const int NumProcedures = 0;
  static const bool ItemizedOnly = true;
  static const int CombMaxNumThreads = 0;

  static const int MinThreadsAmongWorkpackages = 2048;

  template<int ThreadCount>
  struct GetOccupancy
  {
    static const bool Runable = true;
    static const int SumOccupancy = 0;
  };

  template<bool MultiPackage>
  static void updateRequiredShared(int numThreads, cl_uint4& sharedMem, bool copyToShared, int maxShared, bool MultiExecIdentifieres)
  {
  }

  template<class Proc>
  struct Contains
  {
    static const bool value = false;
  };

  template<bool MultiElement>
  struct OptimalThreadCount
  {
    static const int Num = 0;
  };

  template<bool MultiPackage>
  static cl_uint4 requiredShared(int numThreads, bool copyToShared = true, int maxShared = 49100, bool MultiExecIdentifieres = false)
  {
    cl_uint4 sharedMem;
    sharedMem.s[0] = sharedMem.s[1] = sharedMem.s[2] = sharedMem.s[3] = 0;
    return sharedMem;
  }

};
#else
//OPENCL COPY
class ProcInfoEnd
{
public:

  typedef NoProcedure Procedure;
  typedef ProcInfoEnd Next;


  static void print()
  {
    printf("\n");
  }

   const int ProcedureId = 0;

   const int MaxId = 0;
   const int MaxDataSize = 0;
   const int NumProcedures = 0;
   const bool ItemizedOnly = true;
   const int CombMaxNumThreads = 0;

   const int MinThreadsAmongWorkpackages = 2048;

 
};
#endif

#ifndef OPENCL_CODE
template<class TProcInfo, int MinThreadCount, int MaxThreadCount, int Step = 32>
struct IterateOccupancy
{
  typedef IterateOccupancy<TProcInfo, (MinThreadCount + Step > MaxThreadCount ?  MaxThreadCount : MinThreadCount + Step), MaxThreadCount, Step> TNext;
  static const int ThisOccupancy = TProcInfo:: template GetOccupancy<MinThreadCount>::AverageOccupancy;
  static const int OptimalOccupancy = TNext::OptimalOccupancy > ThisOccupancy ? TNext::OptimalOccupancy :  ThisOccupancy;
  static const int OptimalThreadCount = TNext::OptimalOccupancy > ThisOccupancy ? TNext::OptimalThreadCount :  MinThreadCount;
};

template<class TProcInfo, int MaxThreadCount, int Step>
struct IterateOccupancy<TProcInfo, MaxThreadCount, MaxThreadCount, Step>
{
  static const int ThisOccupancy = TProcInfo:: template GetOccupancy<MaxThreadCount>::AverageOccupancy;
  static const int OptimalOccupancy =  ThisOccupancy;
  static const int OptimalThreadCount =  MaxThreadCount;
};

template<class TProc, class TNext = ProcInfoEnd>
class ProcInfo
{
public:
  typedef TProc Procedure;
  typedef TNext Next;

  static void print()
  {
    printf("%d->", Procedure::myid);
    Next::print();
  }

  static const int ProcedureId = Next::ProcedureId + 1;

  static const int NumPhases = 1;
  template<class TTProc, int Phase> 
  struct PhaseTraits : public AllPhasesActiveTrait<TTProc,Phase>{ };

  template<int Phase>
  struct Priority : public NoPriority<Phase> { };

  static const int MaxId =  maxOperator< ProcedureId, Next::MaxId>::result;
  static const int MaxDataSize =  maxOperator< sizeof(typename Procedure::ExpectedData), Next::MaxDataSize>::result;
  typedef typename DataAlignment<MaxDataSize>::Type QueueDataContainer;
  static const int NumProcedures = Next::NumProcedures + 1;
  
  static const bool ItemizedOnly = Procedure::ItemInput && Next::ItemizedOnly;
  static const int CombMaxNumThreads = maxOperator< Procedure::ItemInput?0:Procedure::NumThreads, Next::CombMaxNumThreads>::result;

  static const int MinThreadsAmongWorkpackages = Procedure::ItemInput ? Next::MinThreadsAmongWorkpackages : minOperator<Procedure::NumThreads, Next::MinThreadsAmongWorkpackages>::result;

  
  template<int ThreadCount>
  struct GetOccupancy
  {
    static const int UseThreads = maxOperator< Procedure::NumThreads, 1>::result;
    static const int Occupancy = ((ThreadCount / UseThreads) * UseThreads * 1000) / maxOperator<ThreadCount, 1>::result;
    static const bool Runable = ThreadCount >= UseThreads && Next:: template GetOccupancy<ThreadCount>::Runable;

    static const int SumOccupancy = Runable ? Next:: template GetOccupancy<ThreadCount>:: SumOccupancy + Occupancy : 0;
    static const int AverageOccupancy = SumOccupancy / NumProcedures;
  };

  static const int OptimalThreadCountNonMulti = ItemizedOnly || CombMaxNumThreads == 0 ? 512 : CombMaxNumThreads;
  static const int OptimalThreadCountMulti = ItemizedOnly || CombMaxNumThreads == 0 ? 256 : IterateOccupancy<ProcInfo, CombMaxNumThreads, 1024, 32>::OptimalThreadCount;

  template<bool MultiElement>
  struct OptimalThreadCount
  {
    static const int Num = MultiElement ? OptimalThreadCountMulti : OptimalThreadCountMulti;
  };
 
  //x .. procids
  //y .. data
  //z .. shared mem for procedures
  //w .. sum
  template<bool MultiPackage>
  static cl_uint4 requiredShared(int numThreads, bool copyToShared = true, int maxShared = 49100, bool MultiExecIdentifieres = false)
  {
    
    cl_uint4 sharedMem;
    sharedMem.s[0] = sharedMem.s[1] = sharedMem.s[2] = sharedMem.s[3] = 0;
    if(maxShared < 0)
    {
      printf("ERROR: cannot run with negative amount of shared memory!!\n");
      return sharedMem;
    }
    updateRequiredShared<MultiPackage>(numThreads, sharedMem, copyToShared, maxShared, MultiExecIdentifieres);

    sharedMem.s[0] = (sharedMem.s[0] + 15)/16*16;
    sharedMem.s[1] = (sharedMem.s[1] + 15)/16*16;
    sharedMem.s[2] = (sharedMem.s[2] + 15)/16*16;
    sharedMem.s[3] = sharedMem.s[0] + sharedMem.s[1] + sharedMem.s[2];

    return sharedMem;
  }

  template<class Proc>
  struct Contains
  {
    static const bool value = Equals<Proc, Procedure>::result || Next:: template Contains<Proc>::value;
  };


  template<bool MultiPackage>
  static void updateRequiredShared(int numThreads, cl_uint4 & sharedMem, bool copyToShared, int maxShared, bool MultiExecIdentifieres)
  {
    int num  = numThreads;
    cl_uint4 mysharedAbs = cl_uint4(0,0,0,0);
    cl_uint4 mysharedMul = cl_uint4(0,0,0,0);
    if(Procedure::ItemInput)
    { 
      num = numThreads;
      if(Procedure::NumThreads != 0)
        num = numThreads /  AvoidZero<Procedure::NumThreads>::val;
      if(MultiExecIdentifieres)
        mysharedMul.s[0] = sizeof(uint);
      else
        mysharedAbs.s[0] = std::max<uint>(sharedMem.s[0], sizeof(uint));
    }
    else
    {
      num = 1;
      if(Procedure::NumThreads != 0 && MultiPackage)
        num = numThreads / AvoidZero<Procedure::NumThreads>::val;
      mysharedMul.s[0] = sizeof(uint);
    }
    if(copyToShared)
      mysharedMul.s[1] = sizeof(typename Procedure::ExpectedData);
    mysharedMul.s[2] = Procedure::sharedMemory;

    cl_uint4 current;
    current.s[0] = std::max<uint>(sharedMem.s[0], (mysharedMul.s[0]*num + mysharedAbs.s[0] +15)/16*16);
    current.s[1] = std::max<uint>(sharedMem.s[1], (mysharedMul.s[1]*num + mysharedAbs.s[1] +15)/16*16);
    current.s[2] = std::max<uint>(sharedMem.s[2], (mysharedMul.s[2]*num + mysharedAbs.s[2] +15)/16*16);
    while(current.s[0] + current.s[1] + current.s[2] > static_cast<unsigned>(maxShared) )
    {
      if(--num <= 0)
      {
        printf("ERROR: cannot fulfill shared memory requirements!!\n");
        num = 1;
        break;
      }
      current.s[0] = std::max<uint>(sharedMem.s[0], (mysharedMul.s[0]*num + mysharedAbs.s[0] +15)/16*16);
      current.s[1] = std::max<uint>(sharedMem.s[1], (mysharedMul.s[1]*num + mysharedAbs.s[1] +15)/16*16);
      current.s[2] = std::max<uint>(sharedMem.s[2], (mysharedMul.s[2]*num + mysharedAbs.s[2] +15)/16*16);
    }
      
    //printf("proc %d can execute %d items at maximum: %d %d %d\n", PROCEDURE::ProcedureId, num, (mysharedMul.x*num + mysharedAbs.x +15)/16*16, (mysharedMul.y*num + mysharedAbs.y +15)/16*16, (mysharedMul.z*num + mysharedAbs.z +15)/16*16);
    sharedMem.s[0] = current.s[0];
    sharedMem.s[1] = current.s[1];
    sharedMem.s[2] = current.s[2];

    Next:: template updateRequiredShared<MultiPackage>(numThreads, sharedMem, copyToShared, maxShared, MultiExecIdentifieres);
  }

};
#endif


template <typename A, typename B>
struct typesEqual
{
	 const bool value = false;
};

template <typename A>
struct typesEqual<A, A>
{
	 const bool value = true;
};

template <bool b, int id_a, int id_b>
struct selectProcId;

template <int id_a, int id_b>
struct selectProcId<true, id_a, id_b>
{
	 const int value = id_a;
};

template <int id_a, int id_b>
struct selectProcId<false, id_a, id_b>
{
	 const int value = id_b;
};

#ifdef OPENCL_CODE
template <typename ProcInfo, typename Proc>
struct findProcId
{
	const int value = selectProcId<typesEqual<typename ProcInfo::Procedure, Proc>::value, ProcInfo::ProcedureId, findProcId<typename ProcInfo::Next, Proc>::value>::value;
};
#else
template <typename ProcInfo, typename Proc>
struct findProcId
{
	static const int value = selectProcId<typesEqual<typename ProcInfo::Procedure, Proc>::value, ProcInfo::ProcedureId, findProcId<typename ProcInfo::Next, Proc>::value>::value;
};
#endif

template <typename Proc>
struct findProcId<ProcInfoEnd, Proc>
{
	 const int value = -1;
};


#ifdef OPENCL_CODE
template<class TProcInfo>
class ProcedureIdentifier
{
  int procId;
public:
  __inline__ /*__device__*/ ProcedureIdentifier(int procId) : procId(procId) { };
  
  template<class TProc>
  __inline__ /*__device__*/ static ProcedureIdentifier create()
  {
    return ProcedureIdentifier(findProcId<TProcInfo, TProc>::value);
  }

  template<class TProc>
  __inline__ /*__device__*/ bool equals()
  {
    return findProcId<TProcInfo, TProc>::value == procId;
  }

  __inline__ /*__device__*/ operator int()
  {
    return procId;
  }
};
#endif

#ifndef OPENCL_CODE
template<class Procedure, class Next = ProcInfoEnd>
class N : public ProcInfo<Procedure, Next>
{ };

template<class TProcInfo, int id>
struct Select
{
  typedef typename Select<typename TProcInfo::Next, id-1>::Procedure Procedure;
};

template<class TProcInfo>
struct Select<TProcInfo, 0>
{
  typedef typename TProcInfo::Procedure Procedure;
};
#endif

#ifdef OPENCL_CODE
template <int ElementSize>
struct DataAlignment
{
  struct Type
  {
    const int NumElements = (ElementSize+3)/4;
    unsigned int data[NumElements];

    /*__host__*/ /*__device__*/ Type()
    {
    }

    /*__host__*/ /*__device__*/ __inline__ 
    const Type& operator = ( const Type& other)
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
    /*__host__*/ /*__device__*/ __inline__ 
    const volatile Type& operator = ( const  Type& other) volatile
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
    /*__host__*/ /*__device__*/ __inline__ 
    const Type& operator = ( const volatile Type& other)
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
  };
};
#else
template <int ElementSize>
struct DataAlignment
{
  struct Type
  {
    static const int NumElements = (ElementSize+3)/4;
    unsigned int data[NumElements];

    /*__host__*/ /*__device__*/ Type()
    {
    }

    /*__host__*/ /*__device__*/ __inline__ 
    const Type& operator = ( const Type& other)
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
    /*__host__*/ /*__device__*/ __inline__ 
    const volatile Type& operator = ( const  Type& other) volatile
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
    /*__host__*/ /*__device__*/ __inline__ 
    const Type& operator = ( const volatile Type& other)
    {
      for(int i = 0; i < NumElements; ++i)
        data[i] = other.data[i];
      return *this;
    }
  };
};
#endif

#ifndef OPENCL_CODE
template <>
struct DataAlignment<1>
{
  typedef unsigned int Type;
};
template <>
struct DataAlignment<2>
{
  typedef unsigned int Type;
};
template <>
struct DataAlignment<3>
{
  typedef unsigned int Type;
};
template <>
struct DataAlignment<4>
{
  typedef unsigned int Type;
};



template<class PriorityEvaluation, class TProcInfo>
class ProcInfoWithPriority : public TProcInfo
{
public:
  template<int Phase>
  class Priority : public PriorityEvaluation { };
};

template<int TNumPhases, template<class /*Proc*/, int /*Phase*/> class TPhaseTraits, template <int /*Phase*/> class TPriority, class TProcInfo>
class ProcInfoMultiPhase : public TProcInfo
{
public:
  static const int NumPhases = TNumPhases;
  template<class TTProc, int Phase> 
  class PhaseTraits : public TPhaseTraits<TTProc,Phase>{ };
  
  template<int Phase>
  class Priority : public TPriority<Phase> { };
};

template<class A, class B, bool useA>
struct ClassSelector;

template<class A, class B>
struct ClassSelector<A,B, true>
{
  typedef A TheClass;
};

template<class A, class B>
struct ClassSelector<A,B, false>
{
  typedef B TheClass;
};
#endif

template<class TThisAttachment, class Visitor, class ThisProc, class TargetProc, class Next>
class VisitSpecificSelector
{
public:
#ifdef OPENCL_CODE
  __inline__ /*__device__*/ static bool visit(Next& next, Visitor& visitor, TThisAttachment& data)
  {
    return next . template VisitSpecific<Visitor, TargetProc>(visitor);
  }
#endif
};

template<class TThisAttachment, class Visitor, class MatchProc, class Next>
class VisitSpecificSelector<TThisAttachment, Visitor,MatchProc,MatchProc,Next>
{
public:
#ifdef OPENCL_CODE
  __inline__ /*__device__*/ static bool visit(Next& next, Visitor& visitor, TThisAttachment& data)
  {
     return visitor . template visit< TThisAttachment >(data);
  }
#endif
};


template<template<class /*Procedure*/> class TAttachment, class TProcInfo>
class Attach
{
  typedef TAttachment<typename TProcInfo::Procedure> TThisAttachment;
  TThisAttachment data;
  Attach<TAttachment, typename TProcInfo::Next> next;

  template<class Visitor>
  struct RandVisitorBeg
  {
    int notBefore, i;
    Visitor & v;
#ifdef OPENCL_CODE
    __inline__ /*__device__*/ RandVisitorBeg(Visitor & v, int randOffset) : notBefore(randOffset), i(0), v(v) { }
    template<class T>
    __inline__ /*__device__*/ bool visit(T& data)
    {
      if(i < notBefore)
      {
        ++i;
        return false;
      }
      return v . template visit<T>(data);
    }
#endif
  };


  template<class Visitor>
  struct RandVisitorEnd
  {
    int notBefore, i;
    Visitor & v;
    bool runOver; 
#ifdef OPENCL_CODE
	__inline__ /*__device__*/ RandVisitorEnd(Visitor & v, int randOffset) : notBefore(randOffset), i(0), v(v), runOver(false) { }
    template<class T>
    __inline__ /*__device__*/ bool visit(T& data)
    {
      if(i < notBefore)
      {
        ++i;
        return v . template visit<T>(data);
      }
      runOver = true;
      return true;
     
    }
#endif
  };
public:
#ifdef OPENCL_CODE
	template<class Visitor>
   __inline__ /*__device__*/ bool VisitAll(Visitor& visitor)
  {
    if(visitor.template visit<TThisAttachment >(data))
      return true;
    return next. template VisitAll<Visitor>(visitor);
  }

   template<class Visitor>
   __inline__ /*__device__*/ bool VisitAllRandStart(Visitor& visitor)
  {
    int offset = whippletree::random::rand() % TProcInfo :: NumProcedures;
    RandVisitorBeg<Visitor> v(visitor, offset);
    if(VisitAll<RandVisitorBeg<Visitor> > (v))
      return true;
    RandVisitorEnd<Visitor> v2(visitor, offset);
    VisitAll<RandVisitorEnd<Visitor> > (v2);
    return !v2.runOver;
  }

  template<class Visitor, class TargetProc>
   __inline__ /*__device__*/ bool VisitSpecific(Visitor& visitor)
  {
    //printf("%d vs %d\n", TargetProc::myid, TProcInfo::Procedure::myid);
    return VisitSpecificSelector<TThisAttachment,Visitor, typename TProcInfo::Procedure, TargetProc,  Attach<TAttachment, typename TProcInfo::Next> >::visit(next, visitor, data);
  }
  template<class Visitor>
   __inline__ /*__device__*/ bool VisitSpecific(Visitor& visitor, int ProcId)
  {
    if(ProcId == TProcInfo::Procedure::ProcedureId)
      return visitor . template visit< TThisAttachment >(data);
    return next . template VisitSpecific<Visitor>(visitor, ProcId);
  }
#endif
};


template<template<class /*Procedure*/> class TAttachment>
class Attach<TAttachment, ProcInfoEnd>
{
public:
#ifdef OPENCL_CODE
  template<class Visitor>
   __inline__ /*__device__*/ bool VisitAll(Visitor& visitor)
  {
    return false;
  }
  template<class Visitor, class TargetProc>
   __inline__ /*__device__*/ bool VisitSpecific(Visitor& visitor)
  {
    return false;
  }
  template<class Visitor>
   __inline__ /*__device__*/ bool VisitSpecific(Visitor& visitor, int ProcId)
  {
    return false;
  }
#endif
};


template<class TProcInfo, class TCustom = void>
class ProcInfoVisitor
{
public:
  template<class Visitor>
  static bool HostVisit(Visitor& visitor)
  {
    if(visitor.template visit<typename TProcInfo::Procedure, TCustom >())
      return true;
    return ProcInfoVisitor<typename TProcInfo::Next, TCustom> :: template HostVisit<Visitor>(visitor);
  }

#ifdef OPENCL_CODE
  template<class Visitor>
  static __inline__ /*__device__*/ bool Visit(Visitor& visitor)
  {
    if(visitor.template visit<typename TProcInfo::Procedure, TCustom >())
      return true;
    return ProcInfoVisitor<typename TProcInfo::Next, TCustom > :: template Visit<Visitor>(visitor);
  }
#endif
};



template<class TCustom>
class ProcInfoVisitor<ProcInfoEnd, TCustom>
{
public:
  template<class Visitor>
  static bool HostVisit(Visitor& visitor)
  {
    return false;
  }

#ifdef OPENCL_CODE
  template<class Visitor>
  static __inline__ /*__device__*/  bool Visit(Visitor& visitor)
  {
    return false;
  }
#endif
};
