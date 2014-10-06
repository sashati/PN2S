///////////////////////////////////////////////////////////
//  DeviceManager.cpp
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "DeviceManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <assert.h>

using namespace pn2s;
static bool _isInitialized = false;
extern std::map< unsigned int, pn2s::Location > locationMap;

/**
 * Multithread tasks section
 */

int DeviceManager::CkeckAvailableDevices(){
//	cudaProfilerStop();
	_device.clear();

	int device_count = 0;
	cudaDeviceReset();
	cudaGetDeviceCount(&device_count);
	device_count = min(Parameters::MAX_DEVICE_NUMBER, device_count);
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

		//Read parameters from environment
		char * val = getenv("MAX_STREAM_NUMBER");
		if (val != NULL)
			istringstream(val) >> pn2s::Parameters::MAX_STREAM_NUMBER;
		val = getenv("MAX_DEVICE_NUMBER");
		if (val != NULL)
			istringstream(val) >> pn2s::Parameters::MAX_DEVICE_NUMBER;

		DeviceManager::CkeckAvailableDevices();

	}
	return  Error_PN2S::NO_ERROR;
}


/**
 * This function, assign the model shapes into devices
 * and assign memory for PFields
 */

void DeviceManager::AllocateMemory(vector< vector <Model_pack_info> > &m, double dt){
	assert(m.size() > 0);
	for(int i = 0; i < _device.size(); i++)
	{
		cudaSetDevice(_device[i].id);
		_device[i].AllocateMemory(m[i], dt);
	}
}

void DeviceManager::PrepareSolvers()
{
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
	{
		cudaSetDevice(device->id);
		device->PrepareSolvers();
	}

	//Prepare the scheduling mechanism

}

void* process( void* a) {
	Device *d = reinterpret_cast<Device *>(a);
	cudaSetDevice(d->id);
	d->Process();
	return 0;
}

void DeviceManager::Process()
{
	for(int i = 0; i< _device.size();i++)
		_device[i].Process();

//	pthread_t threads[16];
//	for(int i = 0; i< _device.size();i++)
//		pthread_create(threads + i, NULL,process,&_device[i]);
//
//
//	for(int i = 0; i< _device.size();i++)
//		pthread_join(threads[i], NULL);
}

void DeviceManager::Close()
{
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Destroy();
}

