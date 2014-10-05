///////////////////////////////////////////////////////////
//  DeviceManager.cpp
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "DeviceManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include "tbb/flow_graph.h"
#include "tbb/atomic.h"
#include "tbb/tick_count.h"
using namespace tbb;
using namespace tbb::flow;


using namespace pn2s;
static bool _isInitialized = false;
extern std::map< unsigned int, pn2s::Location > locationMap;

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
}

/**
 * Multithread tasks section
 */

struct process_body{
	Device* operator()( Device* d) {
//		_D(std::cout<< "Process" << m<<endl<<flush);
		cudaSetDevice(d->id);
		d->Process();
        return d;
    }
};

void DeviceManager::Process()
{
	graph scheduler;
	broadcast_node<Device*> broadcast(scheduler);
	function_node< Device*, Device* > process_node(scheduler, 1, process_body());
	make_edge( broadcast, process_node );
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		broadcast.try_put(&(*device));
	scheduler.wait_for_all();
}

void DeviceManager::Close()
{
	for(vector<Device>::iterator device = _device.begin(); device != _device.end(); ++device)
		device->Destroy();
}

