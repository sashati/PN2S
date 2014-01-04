///////////////////////////////////////////////////////////
//  HSC_Device.cpp
//  Implementation of the Class HSC_Device
//  Created on:      26-Dec-2013 4:18:01 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#include "HSC_Device.h"
#include "Definitions.h"
#include <cuda.h>
#include <cuda_runtime.h>

HSC_Device::HSC_Device(int _id): id(_id){
	cudaDeviceReset();
}

HSC_Device::~HSC_Device(){
	cudaDeviceReset();
}


int HSC_Device::GetNumberOfActiveDevices(){
	 int device_count = 0;
	 cudaGetDeviceCount(&device_count);
	 return device_count;
}

hscError HSC_Device::SelectDevice(){
	cudaDeviceProp deviceProp;
	cudaSetDevice(id);
	cudaGetDeviceProperties(&deviceProp, id);
	//TODO: Set configurations in device
	return  NO_ERROR;
}
