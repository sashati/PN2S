///////////////////////////////////////////////////////////
//  DeviceManager.h
//  Implementation of the Class DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "Device.h"
#include "../../basecode/header.h" //Moose header

/**
 * Create one thread per device and let it process tasks and when finished, add
 * and output task to Scheduler
 */
namespace pn2s
{

class DeviceManager
{
	static int CkeckAvailableDevices();
	static vector<Device> _device; //TODO: Should be private
public:
	static bool IsInitialized();
	static Error_PN2S Initialize();

	// Distribute model between devices
	static void AllocateMemory(vector<Id > &m, double dt);
	static void PrepareSolvers();
	static void Process();
	static void Close();
	static vector<Device>& Devices(){
		return _device;
	}
};
}
