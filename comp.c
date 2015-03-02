// System includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "comp.h"
// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

char *mycode;
size_t source_size;
#define MAX_SOURCE_SIZE (0x1000000)

extern cl_command_queue cmdQueue;
extern cl_kernel * kernels;

//Error checking Macro
#include <assert.h>
#define clErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cl_int code, const char *file, int line)
{
	if (code != CL_SUCCESS) 
	{
		fprintf(stderr,"GPUassert: %i %s %d\n", code, file, line);
		//if (abort) exit(code);
	}
}

//Reading kernels from file
bool readKernelFromFile()
{
	FILE *fp;
	fp= fopen("test.cpp","r");
	if (!fp)
	{
		printf("Failed to load kernel/ \n");
		return(false);
	}
	mycode=(char*)malloc(MAX_SOURCE_SIZE);
	source_size=fread(mycode,1,MAX_SOURCE_SIZE,fp);
	std::cout<<source_size<<"\n";
	fclose(fp);
	return(true);
}

void compile_device_code() {
    // This code executes on the OpenCL host
    cl_int status;  
      
    //Initializing platform
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    clErrchk(clGetPlatformIDs(0, NULL, &numPlatforms));
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
	clErrchk(clGetPlatformIDs(numPlatforms, platforms,NULL));
    

    //Initialize device
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));

    //Creating context
    cl_context context = NULL;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	clErrchk(status);

    //Creating command queue
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	clErrchk(status);
	kernels=(cl_kernel*)malloc(sizeof(cl_kernel));
	//Reading and compiling program
    if(!readKernelFromFile())
		exit(1);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&mycode, &source_size, &status);
	clErrchk(status);

	
	char options[1024*1024];
	sprintf(options, "-w -x clc++ -I /home/alex/whippletree-opencl/ -I /home/alex/whippletree-opencl/examples/queuing/ -DOPENCL_CODE -DCL_HAS_NAMED_VECTOR_FIELDS");
    clErrchk(clBuildProgram(program, numDevices, devices, options, NULL, NULL));
    
	kernels[0] = clCreateKernel(program, "megakernel_1inst", &status);
	clErrchk(status);
	char *build_log;
	size_t ret_val_size;
	clErrchk(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));
	build_log = new char[ret_val_size+1];
	clErrchk(clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

	// to be carefully, terminate with \0
	// there's no information in the reference whether the string is 0 terminated or not
	build_log[ret_val_size] = '\0';

	std::cout << "BUILD LOG: '" << "'" << std::endl;
	std::cout << build_log << std::endl;
	delete[] build_log;


    // Free OpenCL resources
    clErrchk(clReleaseProgram(program));
    clErrchk(clReleaseCommandQueue(cmdQueue));
    clErrchk(clReleaseContext(context));

    // Free host resources
    free(platforms);
    free(devices);
}
