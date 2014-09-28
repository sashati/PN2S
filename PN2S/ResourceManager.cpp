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


bool compareModels (pn2s::Model_info i,pn2s::Model_info j) { return (i.nChannel<j.nChannel); }

void ResourceManager::ModelDistribution(pn2s::Model_pack_info& m, double dt){

	map<unsigned int, Model_pack_info > splited;
	for(Model_pack_info::iterator i = m.begin(); i != m.end(); i++)
	{
		if(!splited.count(i->nCompt))
			splited[i->nCompt] = Model_pack_info();
		splited[i->nCompt].push_back(*i);
	}

	vector<Model_pack_info> packs;
	typedef std::map<unsigned int, Model_pack_info >::iterator it_type;
	for(it_type i = splited.begin(); i != splited.end(); i++) {
		std::sort(i->second.begin(), i->second.end(), compareModels);
		/**
		 * Create modelpacks
		 */
		// TODO: Also consider limitation on channel size. Maybe a
		// model has a few number of compartment, but many channels
		// reduce efficiency
		size_t nModel_in_pack = ceil( (double)MP_CMPT_SIZE_LIMIT / i->first);
		for(Model_pack_info::iterator start = i->second.begin();
				start < i->second.end();)
		{
			Model_pack_info::iterator end;
			if(start + nModel_in_pack > i->second.end())
				end = i->second.end();
			else
				end = start + nModel_in_pack;

			Model_pack_info pack (nModel_in_pack);
			pack.assign(start,end);
			packs.push_back(pack);
			start = end;
		}
	}

	size_t ds = _devices.size();
	vector< vector <Model_pack_info> > allocation(ds);
	int index = 0;
	for(vector<Model_pack_info>::iterator i = packs.begin(); i != packs.end(); i++)
	{
		allocation[index].push_back(*i);
		index = (index + 1 ) % ds;
//		for(Model_pack_info::iterator j = i->begin(); j != i->end(); j++)
//			cout << j->id << " ";
//		cout << endl << flush;
	}

	_device_manager.AllocateMemory(allocation,dt);
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

