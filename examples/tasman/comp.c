// System includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// OpenCL includes
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

char *mycode;
size_t source_size;
#define MAX_SOURCE_SIZE (0x1000000)

extern cl_context context;
extern cl_device_id *devices;
extern cl_command_queue cmdQueue;
extern cl_kernel * kernels;
extern cl_uint numDevices;
extern int used_cl_device;
extern cl_uint numPlatforms;
extern cl_platform_id *platforms;
extern cl_program program;


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
	fp= fopen("./../../techniqueMegakernel.h","r");
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
      
    if(!readKernelFromFile())
		exit(1);
    program = clCreateProgramWithSource(context, 1, (const char**)&mycode, &source_size, &status);
	clErrchk(status);
	
	char options[1024*1024];
	sprintf(options, "-w -x clc++ -I /home/alex/whippletree-opencl/ -I /home/alex/whippletree-opencl/examples/tasman/ -DOPENCL_CODE -DCL_HAS_NAMED_VECTOR_FIELDS");
    (clBuildProgram(program, numDevices, devices, options, NULL, NULL));
    std::cout << "Program built \n"<< std::endl;
    
	kernels[0] = clCreateKernel(program, "megakernel1", &status);
	//clErrchk(status);
	kernels[1] = clCreateKernel(program, "init_queue1", &status);
	//clErrchk(status);
	//kernels[2] = clCreateKernel(program, "init_data1", &status);
	//clErrchk(status);
	
	char *build_log;
	size_t ret_val_size;
	clErrchk(clGetProgramBuildInfo(program, devices[used_cl_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size));
	build_log = new char[ret_val_size+1];
	clErrchk(clGetProgramBuildInfo(program, devices[used_cl_device], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL));

	// to be carefully, terminate with \0
	// there's no information in the reference whether the string is 0 terminated or not
	build_log[ret_val_size] = '\0';

	std::cout << "BUILD LOG: '" << "'" << std::endl;
	std::cout << build_log << std::endl;
	delete[] build_log;
}
