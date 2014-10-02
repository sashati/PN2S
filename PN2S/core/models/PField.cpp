///////////////////////////////////////////////////////////
//  SolverData.cpp
//  Implementation of the Class SolverData
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "PField.h"

#include <assert.h>
#include <limits>
#include <iomanip>

using namespace pn2s::models;

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

template <typename T>
__inline__ Error_PN2S sendVector(uint size, T* h, T* d, cudaStream_t stream)
{
//	cudaError_t stat = cudaMemcpyAsync(d, h, sizeof(h[0])*size, cudaMemcpyHostToDevice, stream);
	cublasStatus_t stat = cublasSetVectorAsync(size, sizeof(h[0]),h, 1,d,1,stream);
	assert(stat == CUBLAS_STATUS_SUCCESS);
	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ Error_PN2S sendVector(uint size, T* h, T* d)
{
	cudaError_t stat = cudaMemcpy(d, h, sizeof(h[0])*size, cudaMemcpyHostToDevice);
	assert(stat == cudaSuccess);
	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ Error_PN2S getVector(uint size, T* h, T* d)
{
	cublasStatus_t stat = cublasGetVector(size, sizeof(h[0]),d, 1,h,1);
	assert(stat == CUBLAS_STATUS_SUCCESS);
	return Error_PN2S::NO_ERROR;
}


template <typename T>
__inline__ Error_PN2S getVector(uint size, T* h, T* d, cudaStream_t stream)
{
//	cudaError_t stat = cudaMemcpyAsync(h, d, sizeof(h[0])*size, cudaMemcpyDeviceToHost, stream);
	cublasStatus_t stat = cublasGetVectorAsync(size, sizeof(h[0]),d, 1,h,1,stream);
	assert(stat == CUBLAS_STATUS_SUCCESS);

//	cudaError_t stat = cudaMemcpyAsync(h, d, sizeof(h[0])*size, cudaMemcpyDeviceToHost, stream);
//	if (stat != CUDA_SUCCESS) {
//		cerr << "CUDA setting Variables\n";
//		return Error_PN2S::CUDA_Error;
//	}
	return Error_PN2S::NO_ERROR;
}


template <typename T>
PField<T>::PField():
	fieldType(TYPE_IO),
	host(0),
	device(0),
	host_inc(1),
	device_inc(1),
	_size(0),
	extraIndex(0)
{

}

template <typename T>
PField<T>::~PField(){
	//Driver shutting down, so not necessary to release memory
}

template <typename T>
T PField<T>::operator [](int i) const{
	assert(i < _size);
	return host[i];
}

template <typename T>
T & PField<T>::operator [](int i) {
	assert(i < _size);
	return host[i];
}



template <typename T>
Error_PN2S PField<T>::AllocateMemory(size_t size)
{
	_size = size;
	CUDA_SAFE_CALL(cudaMallocHost((void **) &host, size * sizeof(host[0]))); //Define pinned memories
	CUDA_SAFE_CALL(cudaMalloc((void **) &device, size * sizeof(device[0])));

	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ void PField<T>::Fill(TYPE_ value)
{
	memset(host,value, _size * sizeof(host[0]));
}

template <typename T>
Error_PN2S PField<T>::AllocateMemory(size_t size, TYPE_ host_defaultValue)
{
	_size = size;
	CUDA_SAFE_CALL(cudaMallocHost((void **) &host, size * sizeof(host[0]))); //Define pinned memories
	memset(host,host_defaultValue, size * sizeof(host[0])); //Fill
	CUDA_SAFE_CALL(cudaMalloc((void **) &device, size * sizeof(device[0])));

	return Error_PN2S::NO_ERROR;
}

template <typename T>
Error_PN2S PField<T>::Host2Device_Async(cudaStream_t stream)
{
	CALL(sendVector<T>(_size, host,device, stream));
	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ Error_PN2S PField<T>::Device2Host_Async(cudaStream_t stream)
{
	CALL(getVector<T>(_size, host,device,stream));
	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ Error_PN2S PField<T>::Host2Device()
{
	CALL(sendVector<T>(_size, host,device));
	return Error_PN2S::NO_ERROR;
}

template <typename T>
__inline__ Error_PN2S PField<T>::Device2Host()
{
	CALL(getVector<T>(_size, host,device));
	return Error_PN2S::NO_ERROR;
}

template <typename T>
Error_PN2S PField<T>::Send2Device_Async(PField& _hostResource,cudaStream_t stream)
{
	size_t minSize = min(_hostResource._size, _size);
	if(minSize > 0)
		CALL(sendVector<T>(minSize, _hostResource.host,device,stream));
	return Error_PN2S::NO_ERROR;
}

template <typename T>
Error_PN2S PField<T>::Send2Host_Async(PField& _hostResource,cudaStream_t stream)
{
	size_t minSize = min(_hostResource._size, _size);
	if(minSize > 0)
		CALL(getVector<T>(minSize, _hostResource.host,device,stream));
	return Error_PN2S::NO_ERROR;
}

string convert(double v) { ostringstream ss; ss << std::setprecision (std::numeric_limits< double >::digits10) << v; return ss.str();}
string convert(float v) { ostringstream ss; ss << std::setprecision (std::numeric_limits< double >::digits10) << v; return ss.str(); }
string convert(int v) { ostringstream ss; ss << v; return ss.str(); }
string convert(GateParams v) { ostringstream ss; ss << v.p[0]; return ss.str(); }
string convert(unsigned char v) { ostringstream ss; ss << std::hex << (unsigned int)v; return ss.str();}
string convert(TYPE2_ v) {
	std::stringstream ss;
	ss << "("
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.x <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.y <<")";
	return ss.str();
}
string convert(TYPE3_ v) {
	std::stringstream ss;
	ss << "("
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.x <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.y <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.z <<")";
	return ss.str();
}
string convert(TYPE4_ v) {
	std::stringstream ss;
	ss << "("
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.x <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.y <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.z <<", "
			<< std::setprecision (std::numeric_limits< double >::digits10) << v.w <<")";
	return ss.str();
}
string convert(int2 v) { ostringstream ss; ss << "(" << v.x <<", " <<v.y <<")"; return ss.str(); }
string convert(ChannelType v) { ostringstream ss; ss << v; return ss.str(); }

template <typename T>
void PField<T>::print()
{
	for (int i = 0; i < _size; i+=host_inc) {
		cout << std::setprecision (std::numeric_limits< double >::digits10) << convert(host[i]) << "\t";
	}
	cout << endl << flush;
}

template <typename T>
void PField<T>::print(int seperator)
{
	for (int i = 0; i < _size; i+=host_inc) {
		cout << std::setprecision (std::numeric_limits< double >::digits10) << convert(host[i]) << "\t";
		if ((i+1)%seperator == 0)
			cout << endl;
	}
	cout << endl << flush;
}

template class PField<double>;
template class PField<float>;
template class PField<int>;
template class PField<int2>;
template class PField<unsigned char>;
template class PField<ChannelType>;
template class PField<GateParams>;
template class PField<TYPE2_>;
template class PField<TYPE3_>;
template class PField<TYPE4_>;

