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

#include <CL/cl.h>
#include "random.h"

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __RET_PTR   "l"
#else
#define __RET_PTR   "r"
#endif

template<class TAdditionalData>
struct AdditionalDataInfo
{
  static const int size = sizeof(TAdditionalData);
};

template<>
struct AdditionalDataInfo<void>
{
  static const int size = 0;
};

#ifdef OPENCL_CODE
template<int Mod, int MaxWarps>
/*__device__*/ __inline__ int warpBroadcast(int val, int who)
{
#if __CUDA_ARCH__ < 300
  __shared__ volatile int comm[MaxWarps];
  for(int offset = 0; offset < 32; offset += Mod)
  {
    if(Tools::laneid() - offset == who)
      comm[get_local_id(0)/32] = val;
    if(Tools::laneid() < offset + Mod)
      return comm[get_local_id(0)/32];
  }
  return val;
#else
   return __shfl(val, who, Mod);
#endif
}
#endif


#ifdef OPENCL_CODE
template<int Mod>
/*__device__*/ __inline__ int warpBroadcast(int val, int who)
{
  return warpBroadcast<Mod,32>(val, who);
}

template<int Mod, int MaxWarps>
/*__device__*/ __inline__ int warpShfl(int val, int who)
{
#if __CUDA_ARCH__ < 300
  __shared__ volatile int comm[MaxWarps];
  int runid = 0;
  int res = val;
  for(int offset = 0; offset < 32; offset += Mod)
  {
    for(int within = 0; within < Mod; ++within)
    {
      if(Tools::laneid() == runid)
        comm[get_local_id(0)/32] = val;
      if( Tools::laneid() >= offset 
        && Tools::laneid() < offset + Mod 
        && (runid % Mod) == ((who + 32) % Mod) )
        res = comm[get_local_id(0)/32];
      ++runid;
    }
  }
  return res;
#else
   return __shfl(val, who, Mod);
#endif
}
#endif


#ifdef OPENCL_CODE
template<int Mod>
/*__device__*/ __inline__ int warpShfl(int val, int who)
{
  return warpShfl<Mod,32>(val, who);
}
#endif

#ifdef OPENCL_CODE
template<int Maxrand>
/*__device__*/ __inline__ void backoff(int num)
{

  volatile int local = get_local_id(0);
  for(int i = 0; i < (whippletree::random::rand() % Maxrand); ++i)
  {
    local += num*get_local_id(0)/(i+1234);
    __threadfence();
  }
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uint4& load(cl_uint4& dest, const volatile cl_uint4& src)
{
	asm("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(dest.s[0]), "=r"(dest.s[1]), "=r"(dest.s[2]), "=r"(dest.s[3]) : __RET_PTR(&src));
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uint2& load(cl_uint2& dest, const volatile cl_uint2& src)
{
	asm("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(dest.s[0]), "=r"(dest.s[1]) : __RET_PTR(&src));
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ uint& load(uint& dest, const volatile uint& src)
{
	dest = src;
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uint1& load(cl_uint1& dest, const volatile cl_uint1& src)
{
	dest.s[0] = src.s[0];
	return dest;
}
#endif


#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uchar3& load(cl_uchar3& dest, const volatile cl_uchar3& src)
{
	dest.s[0] = src.s[0];
	dest.s[1] = src.s[1];
	dest.s[2] = src.s[2];
	return dest;
}
#endif


#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uchar2& load(cl_uchar2& dest, const volatile cl_uchar2& src)
{
	dest.s[0] = src.s[0];
	dest.s[1] = src.s[1];
	return dest;
}
#endif


#ifdef OPENCL_CODE
__inline__ /*__device__*/ cl_uchar1& load(cl_uchar1& dest, const volatile cl_uchar1& src)
{
	dest.s[0] = src.s[0];
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uint4& store(volatile cl_uint4& dest, const cl_uint4& src)
{
	asm("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" : : __RET_PTR(&dest), "r"(src.s[0]), "r"(src.s[1]), "r"(src.s[2]), "r"(src.s[3]));
	return dest;
}
#endif


#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uint2& store(volatile cl_uint2& dest, const cl_uint2& src)
{
	asm("st.volatile.global.v2.u32 [%0], {%1, %2};" : : __RET_PTR(&dest), "r"(src.s[0]), "r"(src.s[1]));
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile uint& store(volatile uint& dest, const uint& src)
{
	dest = src;
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uint1& store(volatile cl_uint1& dest, const cl_uint1& src)
{
	dest.s[0] = src.s[0];
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uchar3& store(volatile cl_uchar3& dest, const cl_uchar3& src)
{
	dest.s[0] = src.s[0];
	dest.s[1] = src.s[1];
	dest.s[2] = src.s[2];
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uchar2& store(volatile cl_uchar2& dest, const cl_uchar2& src)
{
	dest.s[0] = src.s[0];
	dest.s[1] = src.s[1];
	return dest;
}
#endif

#ifdef OPENCL_CODE
__inline__ /*__device__*/ volatile cl_uchar1& store(volatile cl_uchar1& dest, const cl_uchar1& src)
{
	dest.s[0] = src.s[0];
	return dest;
}
#endif


template<uint TElementSize>
struct StorageElement16
{
	static const int num_storage_owords = (TElementSize + 15) / 16;

	cl_uint4 storage[num_storage_owords];
};


#ifdef OPENCL_CODE
template <int i>
struct StorageDude16
{
	template<uint ElementSize>
	__inline__ /*__device__*/ static StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::assign(dest, src);
		dest.storage[i] = src.storage[i];
		return dest;
	}

	template<uint ElementSize>
	__inline__ /*__device__*/ static StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::load(dest, src);
		::load(dest.storage[i], src.storage[i]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ /*__device__*/ static volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::store(dest, src);
		::store(dest.storage[i], src.storage[i]);
		return dest;
	}
};
#endif

#ifdef OPENCL_CODE
template <>
struct StorageDude16<0>
{
	template<uint ElementSize>
	__inline__ /*__device__*/ static StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		dest.storage[0] = src.storage[0];
		return dest;
	}

	template<uint ElementSize>
	__inline__ /*__device__*/ static StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
	{
		::load(dest.storage[0], src.storage[0]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ /*__device__*/ static volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		::store(dest.storage[0], src.storage[0]);
		return dest;
	}
};
#endif

#ifdef OPENCL_CODE
template<uint ElementSize>
__inline__ /*__device__*/ StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::assign(dest, src);
}
#endif

#ifdef OPENCL_CODE
template<uint ElementSize>
__inline__ /*__device__*/ StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::load(dest, src);
}
#endif


#ifdef OPENCL_CODE
template<uint ElementSize>
__inline__ /*__device__*/ volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::store(dest, src);
}
#endif

struct StorageElement8
{
	cl_uint2 storage;
};

#ifdef OPENCL_CODE
__inline__ /*__device__*/ StorageElement8& assign(StorageElement8& dest, const StorageElement8& src)
{
	dest.storage = src.storage;
	return dest;
}

__inline__ /*__device__*/ StorageElement8& load(StorageElement8& dest, const volatile StorageElement8& src)
{
	load(dest.storage, src.storage);
	return dest;
}

__inline__ /*__device__*/ volatile StorageElement8& store(volatile StorageElement8& dest, const StorageElement8& src)
{
	store(dest.storage, src.storage);
	return dest;
}
#endif

template<uint TElementSize, bool take_eight>
struct StorageElementSelector
{
	typedef StorageElement16<TElementSize> type;
};

template<uint TElementSize>
struct StorageElementSelector<TElementSize, true>
{
	typedef StorageElement8 type;
};

template<uint TElementSize>
struct StorageElementTyping
{
  typedef typename StorageElementSelector<TElementSize, TElementSize <= 8>::type Type;
};

template<>
struct StorageElementTyping<0>;  // life finds a way...

template<>
struct StorageElementTyping<1>
{
  typedef unsigned char Type;
};
template<>
struct StorageElementTyping<2>
{
  typedef cl_uchar2 Type;
};
template<>
struct StorageElementTyping<3>
{
  typedef cl_uchar3 Type;
};
template<>
struct StorageElementTyping<4>
{
  typedef uint Type;
};



template <unsigned int width>
struct selectVectorCopyType;

template <>
struct selectVectorCopyType<16U>
{
	typedef cl_uint4 type;
};

template <>
struct selectVectorCopyType<8U>
{
	typedef cl_uint2 type;
};

template <>
struct selectVectorCopyType<4U>
{
	typedef cl_uint1 type;
};

template <>
struct selectVectorCopyType<3U>
{
	typedef cl_uchar3 type;
};

template <>
struct selectVectorCopyType<2U>
{
	typedef cl_uchar2 type;
};

template <>
struct selectVectorCopyType<1U>
{
	typedef cl_uchar1 type;
};

#ifdef OPENCL_CODE
template <unsigned int bytes, int threads = 1>
struct vectorCopy
{
	static const unsigned int byte_width = bytes >= 16 ? 16 : bytes >= 8 ? 8 : bytes >= 4 ? 4 : 1;
	static const unsigned int iterations = bytes / byte_width;
	static const unsigned int max_threads = iterations < threads ? iterations : threads;
	static const unsigned int iterations_threaded = iterations / max_threads;
	static const unsigned int vectors_copied = max_threads * iterations_threaded;

	typedef typename selectVectorCopyType<byte_width>::type vector_type;

	/*__device__*/ __inline__ static void storeThreaded(volatile void* dest, const void* src, int i);
	/*__device__*/ __inline__ static void loadThreaded(void* dest, const volatile void* src, int i);
};
#endif

#ifdef OPENCL_CODE
template <int threads>
struct vectorCopy<0, threads>
{
	/*__device__*/ __inline__ static void storeThreaded(volatile void* dest, const void* src, int i) {}
	/*__device__*/ __inline__ static void loadThreaded(void* dest, const volatile void* src, int i) {}
};
#endif

#ifdef OPENCL_CODE
template <unsigned int bytes, int threads>
/*__device__*/ __inline__ void vectorCopy<bytes, threads>::storeThreaded(volatile void* dest, const void* src, int i)
{
	volatile vector_type* const destv = static_cast<volatile vector_type*>(dest);
	const vector_type* const srcv = static_cast<const vector_type*>(src);

	if (i < max_threads)
	{
		volatile vector_type* d = destv + i;
		const vector_type* s = srcv + i;
		#pragma unroll
		for (int j = 0; j < iterations_threaded; ++j)
		{
			store(*d, *s);
			d += max_threads;
			s += max_threads;
		}
	}

	vectorCopy<bytes - byte_width * vectors_copied, threads>::storeThreaded(destv + vectors_copied, srcv + vectors_copied, i);
}
#endif


#ifdef OPENCL_CODE
template <unsigned int bytes, int threads>
/*__device__*/ __inline__ void vectorCopy<bytes, threads>::loadThreaded(void* dest, const volatile void* src, int i)
{
	vector_type* const destv = static_cast<volatile vector_type*>(dest);
	const volatile vector_type* const srcv = static_cast<const volatile vector_type*>(src);

	if (i < max_threads)
	{
		vector_type* d = destv + i;
		const volatile vector_type* s = srcv + i;
		#pragma unroll
		for (int j = 0; j < iterations_threaded; ++j)
		{
			load(*d, *s);
			d += max_threads;
			s += max_threads;
		}
	}

	vectorCopy<bytes - byte_width * vectors_copied, threads>::loadThreaded(destv + vectors_copied, srcv + vectors_copied, i);
}
#endif

#ifdef OPENCL_CODE
template<int Threads, class T>
/*__device__*/ __inline__  void multiWrite(volatile T* data_out, T* data)
{
	vectorCopy<sizeof(T), Threads>::storeThreaded(data_out, data, Tools::laneid() % Threads);

	//if (Tools::laneid() % Threads == 0)
	//{
	//	for (int i = 0; i < sizeof(T); ++i)
	//		reinterpret_cast<volatile char*>(data_out)[i] = reinterpret_cast<char*>(data)[i];
	//}
}
#endif

#ifdef OPENCL_CODE
template<int Threads, class T>
/*__device__*/ __inline__  void multiRead(T* data, volatile T* data_in)
{
	vectorCopy<sizeof(T), Threads>::loadThreaded(data, data_in, Tools::laneid() % Threads);

	//if (Tools::laneid() % Threads == 0)
	//{
	//	for (int i = 0; i < sizeof(T); ++i)
	//		reinterpret_cast<volatile char*>(data_in)[i] = reinterpret_cast<char*>(data)[i];
	//}
}
#endif

//__inline__ /*__device__*/ void readStorageElement(void* data, const volatile void* stored, uint size)
//{
  //uint* pData = reinterpret_cast<uint*>(data);
  //const volatile uint* pReadData =  reinterpret_cast<const volatile uint*>(stored); 

  //while(size >= 32)
  //{
  //  *reinterpret_cast<StorageElementTyping<32>::Type*>(pData) = 
  //    *reinterpret_cast<const volatile typename StorageElementTyping<32>::Type*>(pReadData);
  //  size -= 32;
  //  pData += 8; 
  //  pReadData += 8;
  //}
  //if(size >= 16)
  //{
  //  *reinterpret_cast<StorageElementTyping<16>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<16>::Type*>(pReadData);
  //  size -= 16;
  //  pData += 4;
  //  pReadData += 4;
  //}
  //if(size >= 8)
  //{
  //  *reinterpret_cast<StorageElementTyping<8>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<8>::Type*>(pReadData);
  //  size -= 8;
  //  pData += 2;
  //  pReadData += 2;
  //}
  //if(size > 0)
  //{
  //  *reinterpret_cast<StorageElementTyping<4>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<4>::Type*>(pReadData);
  //}
//}


template<uint TElementSize, class TAdditionalData, uint TQueueSize>
class QueueStorage
{
protected:
  typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;
  typedef typename StorageElementTyping<sizeof(TAdditionalData)>::Type QueueAddtionalData_T;
  QueueData_T volatile storage[TQueueSize];
  QueueAddtionalData_T volatile additionalStorage[TQueueSize];

public:

  static std::string name()
  {
    return "";
  }
  
  __inline__ /*__device__*/ void init()
  {
  }

  template<class T>
  __inline__ /*__device__*/ uint prepareData(T data, TAdditionalData additionalData)
  {
    return 0;
  }

  template<int TThreadsPerElenent, class T>
  __inline__ /*__device__*/ uint prepareDataParallel(T* data, TAdditionalData additionalData)
  {
    return 0;
  }

  template<class T>
  __inline__ /*__device__*/ void writeData(T data, TAdditionalData additionalData, cl_uint2 pos)
  {
     pos.s[0] = pos.s[0]%TQueueSize;

    storage[pos.s[0]] = *reinterpret_cast<QueueData_T*>(&data);
    additionalStorage[pos.s[0]] = *reinterpret_cast<QueueAddtionalData_T*>(&additionalData);
  }

    template<int TThreadsPerElenent, class T>
  __inline__ /*__device__*/ void writeDataParallel(T* data, TAdditionalData additionalData, cl_uint2 pos)
  {
    pos.s[0] = pos.s[0]%TQueueSize;
    multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(storage + pos.s[0]), data);
    multiWrite<TThreadsPerElenent, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(additionalStorage + pos.s[0]), &additionalData);

    ////TODO this could be unrolled in some cases...
    //for(int i = Tools::laneid()%TThreadsPerElenent; i < TElementSize/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(storage + pos.s[0])[i] = reinterpret_cast<uint*>(data)[i];

    //for(int i = Tools::laneid()%TThreadsPerElenent; i < sizeof(TAdditionalData)/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(additionalStorage + pos.s[0])[i] = reinterpret_cast<uint*>(&additionalData)[i];
  }

  __inline__ /*__device__*/ void readData(void* data, TAdditionalData* additionalData, uint pos)
  {
    pos = pos%TQueueSize;
    *reinterpret_cast<QueueData_T*>(data) = storage[pos];
    *reinterpret_cast<QueueAddtionalData_T*>(additionalData) = additionalStorage[pos];
  }

  __inline__ /*__device__*/ void* readDataPointers(TAdditionalData* additionalData, uint pos)
  {
    pos = pos%TQueueSize;
    *reinterpret_cast<QueueAddtionalData_T*>(additionalData) = additionalStorage[pos];
    return (void*)(storage + pos);
  }
  __inline__ /*__device__*/ void storageFinishRead(cl_uint2 pos)
  {
  }
};

template<uint TElementSize, uint TQueueSize>
class QueueStorage<TElementSize, void, TQueueSize>
{
protected:
  typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;
  QueueData_T volatile storage[TQueueSize];

public:

  static std::string name()
  {
    return "";
  }

  __inline__ /*__device__*/ void init()
  {
  }

  template<class T>
  __inline__ /*__device__*/ uint prepareData(T data)
  {
    return 0;
  }

  template<int TThreadsPerElenent, class T>
  __inline__ /*__device__*/ uint prepareDataParallel(T* data)
  {
    return 0;
  }

  template<class T>
  __inline__ /*__device__*/ void writeData(T data, cl_uint2 pos)
  {
    pos.s[0] = pos.s[0]%TQueueSize;
    //storage[pos.s[0]] = *reinterpret_cast<QueueData_T*>(&data);
    store(storage[pos.s[0]], *reinterpret_cast<QueueData_T*>(&data));
//printf("TQueueSize: %d, Elementsize %d, offset0: %llx, offset1 %llx\n", TQueueSize,TElementSize, &storage[0], &storage[1]);
  }

  template<int TThreadsPerElenent, class T>
  __inline__ /*__device__*/ void writeDataParallel(T* data, cl_uint2 pos)
  {
    pos.s[0] = pos.s[0]%TQueueSize;
    multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(storage + pos.s[0]), data);

    ////TODO this could be unrolled in some cases...
    //for(int i = Tools::laneid()%TThreadsPerElenent; i < TElementSize/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(storage + pos.s[0])[i] = reinterpret_cast<uint*>(data)[i];
  }

  __inline__ /*__device__*/ void readData(void* data, uint pos)
  {
    pos = pos%TQueueSize;
    load(*reinterpret_cast<QueueData_T*>(data), storage[pos]);
  }

  __inline__ /*__device__*/ void* readDataPointers(uint pos)
  {
    pos = pos%TQueueSize;
    return (void*)(storage + pos);
  }

  __inline__ /*__device__*/ void storageFinishRead(cl_uint2 pos)
  {
  }
};


template<uint TElementSize, uint TQueueSize, class TAdditionalData, class QueueStub, class TQueueStorage >
class QueueBuilder : public ::BasicQueue<TAdditionalData>, protected TQueueStorage, public QueueStub
{
  static const uint ElementSize = (TElementSize + sizeof(uint) - 1)/sizeof(uint);

public:

  __inline__ /*__device__*/ void init()
  {
    QueueStub::init();
    TQueueStorage::init();
  }

  static std::string name()
  {
    return QueueStub::name() + TQueueStorage::name();
  }

  template<class Data>
  __inline__ /*__device__*/ bool enqueueInitial(Data data, TAdditionalData additionalData) 
  {
    return enqueue<Data>(data, additionalData);
  }

  template<class Data>
  /*__device__*/ bool enqueue(Data data, TAdditionalData additionalData) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo = prepareData (data, additionalData);
    do
    {
      pos = QueueStub:: template enqueuePrep<1>(pos);
      if(pos.s[0] >= 0)
      {
          writeData(data, additionalData, make_cl_uint2(pos.s[0], addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<1>(pos);
      }
    } while(pos.s[0] == -2);
    return pos.s[0] >= 0;
  }

  template<int TThreadssPerElment, class Data>
  /*__device__*/ bool enqueue(Data* data, TAdditionalData additionalData) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo =  TQueueStorage :: template prepareDataParallel<TThreadssPerElment> (data, additionalData);
    do
    {
      pos = QueueStub:: template enqueuePrep<TThreadssPerElment>(pos);
      if(pos.s[0] >= 0)
      {
           TQueueStorage :: template writeDataParallel<TThreadssPerElment> (data, additionalData, make_cl_uint2(pos.s[0], addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<TThreadssPerElment>(pos);
      }
    } while(pos.s[0] == -2);
    return pos.s[0] >= 0;
  }

  __inline__ /*__device__*/ int dequeue(void* data, TAdditionalData* addtionalData, int num)
  {
      
    cl_uint2 offset_take = QueueStub::dequeuePrep(num);

    if(get_local_id(0) < offset_take.s[1])
    {
      readData(reinterpret_cast<uint*>(data) + get_local_id(0) * ElementSize, addtionalData + get_local_id(0), offset_take.s[0] + get_local_id(0));
      __threadfence();
    }
    __syncthreads();
    QueueStub::dequeueEnd(offset_take); 
    TQueueStorage::storageFinishRead(offset_take);
    return offset_take.s[1];
  }
};

template<uint TElementSize, uint TQueueSize, class QueueStub, class TQueueStorage >
class QueueBuilder<TElementSize, TQueueSize, void, QueueStub, TQueueStorage>
  : public ::BasicQueue<void>, protected TQueueStorage, public QueueStub
{
  static const uint ElementSize = (TElementSize + sizeof(uint) - 1)/sizeof(uint);
public:

  __inline__ /*__device__*/ void init()
  {
    QueueStub::init();
    TQueueStorage::init();
  }

  static std::string name()
  {
    return QueueStub::name() + TQueueStorage::name();
  }

  template<class Data>
  __inline__ /*__device__*/ bool enqueueInitial(Data data) 
  {
    return enqueue<Data>(data);
  }

  template<class Data>
  /*__device__*/ bool enqueue(Data data) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo = prepareData(data);
    do
    {
      pos = QueueStub::template enqueuePrep<1>(pos);
      if(pos.s[0] >= 0)
      {
        writeData(data, make_cl_uint2(pos.s[0], addinfo));
        __threadfence();
        QueueStub:: template enqueueEnd<1>(pos);
      }
    } while(pos.s[0] == -2);
    return pos.s[0] >= 0;
  }

   template<int TThreadssPerElment, class Data>
  /*__device__*/ bool enqueue(Data* data) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo =  TQueueStorage :: template prepareDataParallel<TThreadssPerElment> (data);
    do
    {
      pos = QueueStub:: template enqueuePrep<TThreadssPerElment>(pos);
      if(pos.s[0] >= 0)
      {
           TQueueStorage :: template writeDataParallel<TThreadssPerElment> (data, make_cl_uint2(pos.s[0], addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<TThreadssPerElment>(pos);
      }
    } while(pos.s[0] == -2);
    return pos.s[0] >= 0;
  }

  __inline__ /*__device__*/ int dequeue(void* data, int num)
  {
    cl_uint2 offset_take = QueueStub::dequeuePrep(num);
    if(get_local_id(0) < offset_take.s[1])
    {
      TQueueStorage::readData(reinterpret_cast<uint*>(data) + get_local_id(0) * ElementSize, offset_take.s[0] + get_local_id(0));
      __threadfence();
    }
    __syncthreads();
    QueueStub::dequeueEnd(offset_take); 
    TQueueStorage::storageFinishRead(offset_take);
    return offset_take.s[1];
  }
};





//FIXME: class is not overflowsave / has no free!!!!
template<uint MemSize>
class MemoryAllocFastest
{
  static const uint AllocElements = MemSize/sizeof(uint);
  uint allocPointer;
public:
  cl_uint4 volatile dataAllocation[AllocElements/4];

  __inline__ /*__device__*/ void init()
  {
    uint lid = get_local_id(0) + get_group_id(0)*get_local_size(0);
    if(lid == 0)
      allocPointer = 0;
  }
  __inline__ /*__device__*/  uint allocOffset(uint size)
  {
    size = size/sizeof(uint);
    uint p = atomicAdd(&allocPointer,size)%AllocElements;
    while(p + size > AllocElements)
      p = atomicAdd(&allocPointer,size)%AllocElements;
    return  p;
  }

  __inline__ /*__device__*/ volatile uint* offsetToPointer(uint offset)
  {
    return  reinterpret_cast<volatile uint*>(dataAllocation) + offset;
  }
  __inline__ /*__device__*/ volatile uint* alloc(uint size)
  {
    return  offsetToPointer(allocOffset(size));
  }

  __inline__ /*__device__*/  void free(void *p, int size)
  {
  }
    __inline__ /*__device__*/  void freeOffset(int offset, int size)
  {
  }
};

//FIXME: allocator is only safe for elements with are a power of two mulitple of 16 bytes (or smaller than 16 bytes)
// and the multiple must be <= 32*16 bytes
template<uint MemSize>
class MemoryAlloc
{
  static const uint AllocSize = 16;
  static const uint AllocElements = MemSize/AllocSize;
  
  uint flags[(AllocElements + 31)/32];
  uint allocPointer;
public:
  cl_uint4 volatile dataAllocation[AllocElements];

  __inline__ /*__device__*/ void init()
  {
    uint lid = get_local_id(0) + get_group_id(0)*get_local_size(0);
    for(int i = lid; i < (AllocElements + 31)/32; i += get_local_size(0)*gridDim.x)
      flags[i] = 0;
    if(lid == 0)
      allocPointer = 0;
  }
  __inline__ /*__device__*/  int allocOffset(uint size)
  {
    size = (size+AllocSize-1)/AllocSize;
    for(uint t = 0; t < AllocElements/AllocSize; ++t)
    {
      int p = atomicAdd(&allocPointer,size)%AllocElements;
      if(p + size > AllocElements)
        p = atomicAdd(&allocPointer,size)%AllocElements;
      //check bits
      int bigoffset = p / 32;
      int withinoffset = p - bigoffset*32;
      uint bits = ((1u << size)-1u) << withinoffset;
      uint oldf = atomicOr(flags + bigoffset, bits);
      if((oldf & bits) == 0)
        return p;
      atomicAnd(flags + bigoffset, oldf | (~bits));
    }
    //printf("could not find a free spot!\n");
    return -1;
  }

  __inline__ /*__device__*/ volatile uint* offsetToPointer(int offset)
  {
    return  reinterpret_cast<volatile uint*>(dataAllocation + offset);
  }
  __inline__ /*__device__*/ int pointerToOffset(void *p)
  {
    return (reinterpret_cast<volatile cl_uint4*>(p)-dataAllocation);
  }
  __inline__ /*__device__*/ volatile uint* alloc(uint size)
  {
    return  offsetToPointer(allocOffset(size));
  }

  __inline__ /*__device__*/  void free(void *p, int size)
  {
    freeOffset(pointerToOffset(p), size);
  }
    __inline__ /*__device__*/  void freeOffset(int offset, int size)
  {
    //printf("free called for %d %d\n",offset, size);
    size = (size+AllocSize-1)/AllocSize;
    int bigoffset = offset / 32;
    int withinoffset = offset - bigoffset*32;
    uint bits = ((1u << size)-1u) << withinoffset;
    atomicAnd(flags + bigoffset, ~bits);
  }
};

template<uint TAvgElementSize, class TAdditionalData, uint TQueueSize, bool TCheckSet = false, template<uint > class MemAlloc = MemoryAlloc>
class AllocStorage : private MemAlloc<TQueueSize*(TAvgElementSize + (TAvgElementSize > 8 || AdditionalDataInfo<TAdditionalData>::size > 8 ? (sizeof(TAdditionalData)+15)/16*16 :  TAvgElementSize > 4 || AdditionalDataInfo<TAdditionalData>::size > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4))>
{

protected:
  static const  int ForceSize = TAvgElementSize > 8 ? 16 :
                                TAvgElementSize > 4 ? 8 : 4;
  static const  int PureAdditionalSize = (sizeof(TAdditionalData)+sizeof(uint)-1)/sizeof(uint);
  static const  int AdditionalSize = TAvgElementSize > 8 || sizeof(TAdditionalData) > 8 ? (sizeof(TAdditionalData)+15)/16*16 :
                                     TAvgElementSize > 4 || sizeof(TAdditionalData) > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4;

  typedef typename StorageElementTyping<sizeof(TAdditionalData)>::Type AdditonalInfoElement;
  typedef typename StorageElementTyping<sizeof(cl_uint2)>::Type OffsetData_T;
  typedef MemAlloc<TAvgElementSize*TQueueSize> TMemAlloc;

  OffsetData_T volatile offsetStorage[TQueueSize];

public:

  static std::string name()
  {
    return std::string("Alloced");// + std::to_string((unsigned long long)AdditionalSize) + " " + std::to_string((unsigned long long)TAvgElementSize);
  }
  
  __inline__ /*__device__*/ void init()
  {
    MemAlloc<TQueueSize*(TAvgElementSize + (TAvgElementSize > 8 || AdditionalDataInfo<TAdditionalData>::size > 8 ? (sizeof(TAdditionalData)+15)/16*16 :  TAvgElementSize > 4 || AdditionalDataInfo<TAdditionalData>::size > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4))>::init();
    if(TCheckSet)
    {
       uint lid = get_local_id(0) + get_group_id(0)*get_local_size(0);
       for(uint i = lid; i < 2*TQueueSize; i+=get_local_size(0)*gridDim.x)
         ((uint*)offsetStorage)[i] = 0;
    }
  }

  template<class T>
  __inline__ /*__device__*/ uint prepareData(T data, TAdditionalData additionalData)
  {
    uint p = allocOffset((sizeof(T) + AdditionalSize + ForceSize - 1)/ForceSize*ForceSize);
    *reinterpret_cast<volatile AdditonalInfoElement*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p) = *reinterpret_cast<AdditonalInfoElement*>(&additionalData);
    *reinterpret_cast<volatile typename StorageElementTyping<sizeof(T)>::Type*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p + AdditionalSize/sizeof(uint) ) = *reinterpret_cast<typename StorageElementTyping<sizeof(T)>::Type*>(&data);
    return p;
  }

  template<int TThreadsPerElement, class T>
  __inline__ /*__device__*/ uint prepareDataParallel(T* data, TAdditionalData additionalData)
  {
    if(TThreadsPerElement == 1)
      return prepareData(*data, additionalData);

    int p;
    if(Tools::laneid()%TThreadsPerElement == 0)
      p = allocOffset((sizeof(T) + AdditionalSize + ForceSize - 1)/ForceSize*ForceSize);
    p = warpBroadcast<TThreadsPerElement>(p, 0);
    //p = __shfl(p, 0, TThreadsPerElement);    
    multiWrite<TThreadsPerElement, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p), &additionalData);
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p + AdditionalSize/sizeof(uint)), data);

    return p;
  }

  template<class T>
  __inline__ /*__device__*/ void writeData(T data, TAdditionalData additionalData, cl_uint2 pos)
  {
    pos.s[0] = pos.s[0]%TQueueSize;
    cl_uint2 o = make_cl_uint2(pos.s[1], sizeof(T));

    if(TCheckSet)
    {
      o.s[0] += 1;
      while(*(((volatile uint*)offsetStorage) + 2*pos.s[0]) != 0)
        __threadfence();
    }

    offsetStorage[pos.s[0]] = *reinterpret_cast<OffsetData_T*>(&o);
  }

  template<int TThreadsPerElement,class T>
  __inline__ /*__device__*/ void writeDataParallel(T* data, TAdditionalData additionalData, cl_uint2 pos)
  {
    if(Tools::laneid()%TThreadsPerElement == 0)
      writeData(*data,  additionalData, pos);
  }

  /*__inline__ /*__device__*/ void readData(void* data, TAdditionalData* additionalData, uint pos)
  {
    OffsetData_T offsetData;
    pos = pos%TQueueSize;
    offsetData  = offsetStorage[pos];
    cl_uint2 offset = *reinterpret_cast<cl_uint2*>(&offsetData);

    if(TCheckSet)
    {
      while( offset.s[0] == 0 || offset.s[1] == 0)
      {
        __threadfence();
        offsetData  = offsetStorage[pos];
        offset = *reinterpret_cast<cl_uint2*>(&offsetData);
      }
      offset.s[0] -= 1;
    }
    
    *reinterpret_cast<AdditonalInfoElement*>(additionalData) = *reinterpret_cast<volatile AdditonalInfoElement*>(reinterpret_cast<volatile uint*>(dataAllocation) + offset.s[0]);
    readStorageElement(data, reinterpret_cast<volatile uint*>(dataAllocation) + offset.s[0] + AdditionalSize/sizeof(uint), offset.s[1]);
   
  }*/
  __inline__ /*__device__*/ void storageFinishRead(cl_uint2 pos)
  {
     
    if(get_local_id(0) < pos.s[1])
    {
      uint p = (pos.s[0] + get_local_id(0)) % TQueueSize;

      OffsetData_T offsetData;
      offsetData  = offsetStorage[p];
      cl_uint2 offset = *reinterpret_cast<cl_uint2*>(&offsetData);

      TMemAlloc::freeOffset(offset.s[0], offset.s[1]);
      if(TCheckSet)
      {
        __threadfence();
        cl_uint2 o = make_cl_uint2(0, 0);
        offsetStorage[p] = *reinterpret_cast<OffsetData_T*>(&o);
      }
    }
  }
};

template<uint TAvgElementSize, uint TQueueSize, bool TCheckSet, template<uint > class MemAlloc>
class AllocStorage<TAvgElementSize, void, TQueueSize, TCheckSet, MemAlloc> : private MemAlloc<TAvgElementSize*TQueueSize>
{
protected:
  static const  int ForceSize = TAvgElementSize > 8 ? 16 :
                                TAvgElementSize > 4 ? 8 : 4;
  
  typedef typename StorageElementTyping<sizeof(cl_uint2)>::Type OffsetData_T;
  typedef MemAlloc<TAvgElementSize*TQueueSize> TMemAlloc;

  OffsetData_T volatile offsetStorage[TQueueSize];

public:

  static std::string name()
  {
    return "Alloced";
  }
  
  __inline__ /*__device__*/ void init()
  {
    MemAlloc<TAvgElementSize*TQueueSize>::init();
    if(TCheckSet)
    {
       uint lid = get_local_id(0) + get_group_id(0)*get_local_size(0);
       for(uint i = lid; i < 2*TQueueSize; i+=get_local_size(0)*gridDim.x)
         ((uint*)offsetStorage)[i] = 0;
    }
  }

  template<class T>
  __inline__ /*__device__*/ uint prepareData(T data)
  {
    uint p = allocOffset((sizeof(T) + ForceSize - 1)/ForceSize*ForceSize);
    *reinterpret_cast<volatile typename StorageElementTyping<sizeof(T)>::Type*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p ) = *reinterpret_cast<typename StorageElementTyping<sizeof(T)>::Type*>(&data);
    return p;
  }

  template<int TThreadsPerElement, class T>
  __inline__ /*__device__*/ uint prepareDataParallel(T* data)
  {
    if(TThreadsPerElement == 1)
      return prepareData(*data);

    int p;
    if(Tools::laneid()%TThreadsPerElement == 0)
      p = allocOffset((sizeof(T) + ForceSize - 1)/ForceSize*ForceSize);
    //p = __shfl(p, 0, TThreadsPerElement);
    p = warpBroadcast<TThreadsPerElement>(p, 0);
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(reinterpret_cast<volatile uint*>(TMemAlloc::dataAllocation) + p), data);
    return p;
  }

  template<class T>
  __inline__ /*__device__*/ void writeData(T data, cl_uint2 pos)
  {
    pos.s[0] = pos.s[0]%TQueueSize;    
    cl_uint2 o = make_cl_uint2(pos.s[1], sizeof(T));

    if(TCheckSet)
    {
      o.s[0] += 1;
      while(*(((volatile uint*)offsetStorage) + 2*pos.s[0]) != 0)
        __threadfence();
    }

    offsetStorage[pos.s[0]] =  *reinterpret_cast<OffsetData_T*>(&o);
  }

  template<int TThreadsPerElement, class T>
  __inline__ /*__device__*/ void writeDataParallel(T* data, cl_uint2 pos)
  {
    if(Tools::laneid()%TThreadsPerElement == 0)
      writeData(*data, pos);
  }

  /*__inline__ __device__ void readData(void* data, uint pos)
  {
    OffsetData_T offsetData;
    pos = pos%TQueueSize;  
    offsetData  = offsetStorage[pos];
    cl_uint2 offset = *reinterpret_cast<cl_uint2*>(&offsetData);

    if(TCheckSet)
    {
      while( offset.s[0] == 0 || offset.s[1] == 0)
      {
        __threadfence();
        offsetData  = offsetStorage[pos];
        offset = *reinterpret_cast<cl_uint2*>(&offsetData);
      }
      offset.s[0] -= 1;
    }
    
    readStorageElement(data, reinterpret_cast<volatile uint*>(dataAllocation) + offset.s[0], offset.s[1]);
  }*/
  __inline__ /*__device__*/ void storageFinishRead(cl_uint2 pos)
  {
     if(get_local_id(0) < pos.s[1])
    {
      uint p = (pos.s[0] + get_local_id(0)) % TQueueSize;
      OffsetData_T offsetData;
      offsetData  = offsetStorage[p];
      cl_uint2 offset = *reinterpret_cast<cl_uint2*>(&offsetData);

      TMemAlloc::freeOffset(offset.s[0], offset.s[1]);
      if(TCheckSet)
      {
        __threadfence();
        cl_uint2 o = make_cl_uint2(0, 0);
        offsetStorage[p] = *reinterpret_cast<OffsetData_T*>(&o);
      }
    }
  }
};

 
