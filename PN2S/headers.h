///////////////////////////////////////////////////////////
//  PN2S.h
//  Contains definitions and required header includes
//
//  Created on:      26-Dec-2013 4:18:10 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A412A01E5_7D8D_4c56_A915_73B69DCFE454__INCLUDED_)
#define A412A01E5_7D8D_4c56_A915_73B69DCFE454__INCLUDED_

#include <math.h>
#include <algorithm>
#include <string>

#include <vector>
#include <map>
#include <set>
#include <deque>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "Parameters.h"
#include "HelperFunctions.h"
//#include <typeinfo> // used in Conv.h to extract compiler independent typeid
//#include <climits> // Required for g++ 4.3.2
//#include <cstring> // Required for g++ 4.3.2
//#include <cstdlib> // Required for g++ 4.3.2
using namespace std;

//	Architectures
#define ARCH_SM13       (0)
#define ARCH_SM20       (1)
#define ARCH_SM30       (2)
#define ARCH_SM35       (3)

//#define checkCudaErrors(val)    ( (val), #val, __FILE__, __LINE__ )

//#define hsc_uint uint
#define hscID_t uint

namespace pn2s{
//Setter and Getter functions
struct FIELD{
	enum  TYPE {CM_BY_DT, EM_BY_RM, RA,INIT_VM, VM, INJECT_BASAL, INJECT_VARYING, CONSTANT, EXT_CURRENT_GK,EXT_CURRENT_EKGK,
		CH_X, CH_Y, CH_Z, CH_X_POWER, CH_Y_POWER, CH_Z_POWER, CH_GBAR, CH_GK, CH_EK, CH_COMPONENT_INDEX};
};

struct Location{
	int16_t pack;
	int16_t device;
	int32_t index;

	Location():pack(0), device(0), index(0){}
	Location(int16_t _d): pack(0), device(_d), index(0){}
	Location(int16_t _d, int16_t _p): pack(_p), device(_d), index(0){}
	Location(int16_t _d,int16_t _p,int32_t _i): pack(_p), device(_d), index(_i){}
};

}


#endif // !defined(A412A01E5_7D8D_4c56_A915_73B69DCFE454__INCLUDED_)
