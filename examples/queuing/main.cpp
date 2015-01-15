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

//#include <cuda_runtime_api.h>
#include <CL/cl.h>

#include <iostream>
#include <tools/utils.h>

cl_context context;
cl_device_id *devices;
    

void runTest(int device);
int main(int argc, char** argv)
{
  try
  {
	cl_int status;
	int cl_device = argc > 1 ? atoi(argv[1]) : 0;

    //Initializing platform
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    clErrchk(clGetPlatformIDs(0, NULL, &numPlatforms));
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    clErrchk(clGetPlatformIDs(numPlatforms, platforms,NULL));
    

    //Initialize device
    cl_uint numDevices;
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    clErrchk(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL));
	if (!numDevices)
    {
       std::cout << "No CL devices available" << std::endl;
       return -1;
    }
	if (numDevices<=cl_device)
    {
       std::cout << "No such CL device ID" << std::endl;
       return -1;
    }

    //Creating context
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
	clErrchk(status);

    //Creating command queue
    
/*
    int cl_device = argc > 1 ? atoi(argv[1]) : 0;

    int count;
    CUDA_CHECKED_CALL(cudaGetDeviceCount(&count));
    if (!count)
    {
       std::cout << "No CUDA devices available" << std::endl;
       return -1;
    }
    cudaDeviceProp deviceProp;
    CUDA_CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, cuda_device));
    std::cout << "Using device: " << deviceProp.name << std::endl;
*/




	runTest(cl_device);


#ifdef WIN32
  if(argc < 3)
    getchar();
#endif


    clErrchk(clReleaseContext(context));
	free(platforms);
    free(devices);
	return 0;
	}
/*  catch (const Tools::CudaError& e)
  {
    std::cout << "CUDA error: " << e.what() << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -1;
  }
  catch (const std::exception& e)
  {
    std::cout << "error: " << e.what() << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -2;
  }
	*/
  catch (...)
  {
    std::cout << "unknown exception!" << std::endl;
#ifdef WIN32
    getchar();
#endif
    return -3;
  }

}
