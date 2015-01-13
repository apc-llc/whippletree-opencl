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



#ifndef TOOLS_UTILS_INCLUDED
#define TOOLS_UTILS_INCLUDED


#include <string>
//#include <sstream>
#include <stdexcept>
//#include <cuda_runtime_api.h>


//Error checking Macro
#include <assert.h>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>


#define clErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cl_int code, const char *file, int line)
{
	if (code != CL_SUCCESS) 
	{
		fprintf(stderr,"GPUassert: %i %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}



namespace Tools
{
    class clError : public std::runtime_error
    {
    private:
      static std::string genErrorString(cl_int error, const char* file, int line)
      {
        //std::ostringstream msg;
        //msg << file << '(' << line << "): error: " << cudaGetErrorString(error);
        //return msg.str();
        return std::string(file) + '(' + std::to_string(static_cast<long long>(line)) + "): error";
      }
    public:
      clError(cl_int error, const char* file, int line)
        : runtime_error(genErrorString(error, file, line))
      {
      }

      clError(cl_int error)
        : runtime_error(0)//cudaGetErrorString(error))
      {
      }

      clError(const std::string& msg)
        : runtime_error(msg)
      {
      }
    };
  inline void checkError(cl_int error, const char* file, int line)
  {
    if (error != CL_SUCCESS)
      throw clError(error, file, line);
  }

  inline void checkError()
  {
    cl_int error = 0;//
    if (error != CL_SUCCESS)
      throw clError(error);
  }
}

#define CL_CHECKED_CALL(call) Tools::checkError(call, __FILE__, __LINE__)
#define CL_CHECK_ERROR() Tools::checkError(__FILE__, __LINE__)
#define CL_IGNORE_CALL(call) call; clGetLastError();


#endif  // TOOLS_UTILS_INCLUDED
