#pragma once

#include "DeviceManager.h"
namespace pn2s
{

class ResourceManager
{
	static DeviceManager _device_manager;
	static vector<Device*> _devices;
public:
	static bool IsInitialized();
	static Error_PN2S Initialize();

	// Distribute model between devices
	static void AllocateMemory(vector<unsigned int > &ids, vector<int2 > &m, double dt);
	static void PrepareSolvers();
	static void Process();
	static void Close();
	static TYPE_ GetValue(pn2s::Location address, FIELD::CM field);
	static void SetValue(pn2s::Location address, FIELD::CM field, TYPE_ value);
	static TYPE_ GetValue(pn2s::Location address, FIELD::CH field);
	static void SetValue(pn2s::Location address, FIELD::CH field, TYPE_ value);
	static void AddExternalCurrent(pn2s::Location address, TYPE_ Gk, TYPE_ GkEk);

	static vector<Device*> Devices(){
		return _devices;
	}
};
}
