#ifndef CL_VEC_TYPE_CONST
#define CL_VEC_TYPE_CONST



#ifdef OPENCL_CODE
	int2 make_int2(int a, int b)
	{
		int2 x; 
		x.x=a; 
		x.y=b; 
		return x;
	}

	uint2 make_uint2(uint a, uint b)
	{
		uint2 x; 
		x.x=a; 
		x.y=b; 
		return x;
	}
#endif
#endif
