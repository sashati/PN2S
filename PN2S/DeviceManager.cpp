///////////////////////////////////////////////////////////
//  DeviceManager.cpp
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "DeviceManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

using namespace pn2s;

vector<Device> DeviceManager::_device;
static bool _isInitialized = false;

std::map< Id, pn2s::Location > locationMap;

int DeviceManager::CkeckAvailableDevices(){
//	cudaProfilerStop();
	_device.clear();

	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	cudaDeviceReset();

	for(int i =0; i<device_count; i++)
	{
		Device d(i);
		_device.push_back(d);
	}


	return device_count;
}

bool DeviceManager::IsInitialized(){
	return  _isInitialized;
}

Error_PN2S DeviceManager::Initialize(){
	if(!_isInitialized)
	{
		_isInitialized = true;

		DeviceManager::CkeckAvailableDevices();

//	pthread_getschedparam(pthread_self(), &policy, &param);
//	param.sched_priority = sched_get_priority_max(policy);
//	pthread_setschedparam(pthread_self(), policy, &param);
	}
	return  Error_PN2S::NO_ERROR;
}


/**
 * This function, assign the model shapes into devices
 * and assign memory for PFields
 */

void DeviceManager::AllocateMemory(vector<Id > &m, double dt){

	assert(_device.size() > 1);

	//Distribute model
	//TODO: Add Multidevice
	Location dev_address;
	int16_t device = 0;
	_device[0].AllocateMemory(dt, m,(size_t)0,(size_t)m.size(), device);

}

void DeviceManager::PrepareSolvers()
{
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
	{
		device->PrepareSolvers();
	}
//	cudaProfilerStart();
}

void DeviceManager::Process()
{

	//TODO: Each device should get its own pack
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Process();

}

void DeviceManager::Close()
{
//	cudaProfilerStop();
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Destroy();
}

