///////////////////////////////////////////////////////////
//  Field.h
//
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "../../headers.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/fill.h>

namespace pn2s
{
namespace models
{
#define PARAMS_A_A 0
#define PARAMS_A_B 1
#define PARAMS_A_C 2
#define PARAMS_A_D 3
#define PARAMS_A_F 4
#define PARAMS_B_A 5
#define PARAMS_B_B 6
#define PARAMS_B_C 7
#define PARAMS_B_D 8
#define PARAMS_B_F 9
#define PARAMS_DIV 10
#define PARAMS_MIN 11
#define PARAMS_MAX 12

struct ExternalCurrent{

	TYPE_ _gk;
	TYPE_ _gkek;
	friend ostream& operator<<(ostream& out, const ExternalCurrent& obj)
	{
	    out << "(" << obj._gk<< ", " << obj._gkek << ")";
	    return out;
	}
};

struct ChannelCurrent{

	TYPE_ _gk;
	TYPE_ _ek;
	friend ostream& operator<<(ostream& out, const ChannelCurrent& obj) // output
	{
	    out << "(" << obj._gk<< ", " << obj._ek << ")";
	    return out;
	}
};
struct ChannelType {
	//TODO: Constants, put them in texture memory
	unsigned char _xyz_power [3];
	TYPE_ _xyz_params[3][13];
	unsigned char _instant;
	TYPE_ _gbar;

	friend ostream& operator<<(ostream& out, const ChannelType& obj) // output
	{
//	    out << "("
//	    		<<  (int) obj._xyx_power << ", " << obj._x << ", "
//	    		<<  (int) obj._y_power << ", " << obj._y << ", "
//	    		<<  (int) obj._z_power << ", " << obj._z << ", "
//	    		<<  (int) obj._instant << ", " << obj._gbar
//	    		<< ")";
	    return out;
	}
};

template <typename T, int arch>
class PField
{
private:
public:
	size_t _size;
	enum FieldType {TYPE_IO, TYPE_INPUT, TYPE_OUTPUT} fieldType;
	T* host;
	int host_inc;
	T* device;
	int device_inc;

	int extraIndex;

	PField();
	//TODO: Use it!
	PField(FieldType t);

	virtual ~PField();

	Error_PN2S AllocateMemory(size_t size);
	Error_PN2S AllocateMemory(size_t size, TYPE_ defaultValue);

	void Fill(TYPE_ value);
	Error_PN2S Host2Device_Async(cudaStream_t stream);
	Error_PN2S Host2Device();
	Error_PN2S Device2Host_Async(cudaStream_t stream);
	Error_PN2S Device2Host();

	Error_PN2S Send2Device_Async(PField& _hostResource,cudaStream_t stream);
	Error_PN2S Send2Host_Async(PField& _hostResource,cudaStream_t stream);

	thrust::device_ptr<T> DeviceStart() {return thrust::device_ptr<T> (device); }
	thrust::device_ptr<T> DeviceEnd() {return thrust::device_ptr<T> (device+_size); }

	T operator [](int i) const {return host[i];}
	T & operator [](int i) {return host[i];}

	void print();
private:

};


}
}

