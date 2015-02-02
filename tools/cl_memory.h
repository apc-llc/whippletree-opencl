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




#ifndef INCLUDED_CL_PTR
#define INCLUDED_CL_PTR


#include <CL/cl.h>

//global context!!!!
extern cl_context context;

#pragma once
template <typename T>
class cuda_ptr
{
private:
  cuda_ptr(const cuda_ptr& p);
  cuda_ptr& operator =(const cuda_ptr& p);

  cl_mem * ptr;

  static void release(cl_mem * ptr)
  {
    if (ptr != nullptr)
      clReleaseMemObject(*ptr);
  }

public:
  explicit cuda_ptr(cl_mem * ptr = nullptr)
    : ptr(ptr)
  {
  }

  cuda_ptr(cuda_ptr&& p)
    : ptr(p.ptr)
  {
    p.ptr = nullptr;
  }

  ~cuda_ptr()
  {
    release(ptr);
  }

  cuda_ptr& operator =(cuda_ptr&& p)
  {
    std::swap(ptr, p.ptr);
    return *this;
  }

  void release()
  {
    release(ptr);
    ptr = nullptr;
  }

  cl_mem ** bind()
  {
    release(ptr);
    return &ptr;
  }

  cl_mem * unbind()
  {
    cl_mem * temp = ptr;
    ptr = nullptr;
    return temp;
  }

  cl_mem* operator ->() const { return ptr; }

  cl_mem& operator *() const { return *ptr; }

  operator cl_mem*() const { return ptr; }

};


#include <memory>
#include "utils.h"

struct cuda_deleter
{
  void operator()(cl_mem * ptr)
  {
   cl_mem * temp = (cl_mem*)&ptr;
   //CL_CHECKED_CALL(clReleaseMemObject(*temp));
  }
};

template <typename T>
inline std::unique_ptr<cl_mem, cuda_deleter> cudaAlloc()
{
  cl_mem * ptr;
  cl_int status;
  printf("trying to allocate %.2f MB cuda buffer (%zu bytes)\n", sizeof(T) * 1.0 / (1024.0 * 1024.0), sizeof(T));
  *ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T), NULL, &status);
  CL_CHECKED_CALL(status);

    return std::unique_ptr<cl_mem, cuda_deleter>(static_cast<cl_mem*>(ptr));
}

template <typename T>
inline std::unique_ptr<cl_mem, cuda_deleter> cudaAllocArray(size_t N)
{
  cl_mem * ptr;
  cl_int status;
  printf("trying to allocate %.2f MB cuda buffer (%zu * %zu bytes)\n", N * sizeof(T) * 1.0 / (1024.0 * 1024.0), N, sizeof(T));
  *ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T)*N, NULL, &status);
  CL_CHECKED_CALL(status);
  return std::unique_ptr<cl_mem, cuda_deleter>(static_cast<cl_mem*>(ptr));
}


#endif  // INCLUDED_CUDA_PTR
