#include "queueInterface.h"
#include "queueHelpers.h"
#include "segmentedStorage.h"

void (*SegmentedStorage::pReinitStorage)() = 0;

#ifdef OPENCL_CODE
/*__device__*/ void* storage = NULL;
#endif

void* SegmentedStorage::StoragePointer = 0;

void SegmentedStorage::destroyStorage()
{
	//if(StoragePointer != 0)
	//	CUDA_CHECKED_CALL(cudaFree(&StoragePointer));
	StoragePointer = 0;
	pReinitStorage = 0;
}

void SegmentedStorage::checkReinitStorage()
{
	if(pReinitStorage != 0)
		pReinitStorage();
}

