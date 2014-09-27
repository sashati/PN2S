///////////////////////////////////////////////////////////
//  DeviceManager.cpp
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "ResourceManager.h"
#include <assert.h>

using namespace pn2s;

//Statically add a device manager to execute on a machine
DeviceManager ResourceManager::_device_manager;
vector<Device*> ResourceManager::_devices;
static bool _isInitialized = false;

std::map< unsigned int, pn2s::Location > locationMap;

bool ResourceManager::IsInitialized(){
	return  _isInitialized;
}

Error_PN2S ResourceManager::Initialize(){
	if(!_isInitialized)
	{
		_isInitialized = true;

		_device_manager.Initialize();

		//	pthread_getschedparam(pthread_self(), &policy, &param);
		//	param.sched_priority = sched_get_priority_max(policy);
		//	pthread_setschedparam(pthread_self(), policy, &param);

		//Get Devices
		_devices = _device_manager.GetDevices();
	}

	return  Error_PN2S::NO_ERROR;
}

TYPE_ ResourceManager::GetValue(pn2s::Location l, FIELD::CM field)
{
	return _devices[l.device]->ModelPacks()[l.pack]._compsSolver.GetValue(l.index,field);
}

void ResourceManager::SetValue(pn2s::Location l, FIELD::CM field, TYPE_ value)
{
	_devices[l.device]->ModelPacks()[l.pack]._compsSolver.SetValue(l.index,field, value);
}

TYPE_ ResourceManager::GetValue(pn2s::Location l, FIELD::CH field)
{
	return _devices[l.device]->ModelPacks()[l.pack]._chanSolver.GetValue(l.index,field);
}

void ResourceManager::SetValue(pn2s::Location l, FIELD::CH field, TYPE_ value)
{
	_devices[l.device]->ModelPacks()[l.pack]._chanSolver.SetValue(l.index,field, value);
}

void ResourceManager::AddExternalCurrent(pn2s::Location l,TYPE_ Gk, TYPE_ GkEk)
{
	return _devices[l.device]->ModelPacks()[l.pack]._compsSolver.AddExternalCurrent(l.index, Gk, GkEk);
}

void ResourceManager::AllocateMemory(vector<unsigned int > &ids, vector<int2 > &m, double dt){
	_device_manager.AllocateMemory(ids,m,dt);
}

void ResourceManager::PrepareSolvers()
{
	_device_manager.PrepareSolvers();
}

void ResourceManager::Process()
{
	_device_manager.Process();
}

void ResourceManager::Close()
{
	_device_manager.Close();
}

