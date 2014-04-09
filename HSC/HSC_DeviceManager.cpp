///////////////////////////////////////////////////////////
//  HSC_DeviceManager.cpp
//  Implementation of the Class HSC_DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_DeviceManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

HSC_DeviceManager::HSC_DeviceManager(){
	_devices.clear();
}



HSC_DeviceManager::~HSC_DeviceManager(){

}

void HSC_DeviceManager::Setup(){
	cudaDeviceReset();
}
