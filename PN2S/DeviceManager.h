///////////////////////////////////////////////////////////
//  DeviceManager.h
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "Device.h"
//#include "../../basecode/header.h" //Moose header

/**
 * Create one thread per device and let it process tasks and when finished, add
 * and output task to Scheduler
 */
namespace pn2s
{

class DeviceManager
{
	int CkeckAvailableDevices();
	vector<Device> _device;
public:
	bool IsInitialized();
	Error_PN2S Initialize();

	// Distribute model between devices
	void AllocateMemory(vector<unsigned int > &ids, vector<int2 > &m, double dt);
	void PrepareSolvers();
	void Process();
	void Close();

	vector<Device*> GetDevices(){
		vector<Device*> res;
		for(vector<Device>::iterator d = _device.begin(); d != _device.end(); ++d)
			res.push_back(&(*d));
		return res;
	}
};
}
