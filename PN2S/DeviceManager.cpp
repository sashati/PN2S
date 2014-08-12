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
#include <assert.h>

using namespace pn2s;

vector<Device> DeviceManager::_device;
static bool _isInitialized = false;

std::map< unsigned int, pn2s::Location > locationMap;

int DeviceManager::CkeckAvailableDevices(){
//	cudaProfilerStop();
	_device.clear();

	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	cudaDeviceReset();
	device_count = min(MAX_DEVICE_NUMBER, device_count);
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

void DeviceManager::AllocateMemory(vector<unsigned int > &ids, vector<int2 > &m, double dt){
	assert(_device.size() > 0);
	assert(ids.size() > 0);
	assert(m.size() > 0);

	if(ids.size() < _device.size() )
		for (int i = ids.size() - 1; i < _device.size(); ++i)
			_device.pop_back();

	//Distribute model
	size_t start = 0;
	size_t pack = ids.size() / _device.size();
	size_t end = pack-1;
	for (int i = 0; i < _device.size(); ++i) {
		if (i == _device.size()-1) //Last one
			end = ids.size() - 1;
		_device[i].AllocateMemory(dt, ids, m, start, end);
		start += pack;
		end += pack;
	}
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
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Process();
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Sync();
}

void DeviceManager::Close()
{
//	cudaProfilerStop();
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Destroy();
}

