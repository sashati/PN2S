#pragma once

#include "DeviceManager.h"
namespace pn2s
{

class ResourceManager
{
public:
	static bool IsInitialized();
	static Error_PN2S Initialize();

	// Distribute model between devices
	static void ModelDistribution(pn2s::Model_pack_info& m, double dt);
	static void PrepareSolvers();
	static void Process();
	static void Close();
	static TYPE_ GetValue(pn2s::Location address, FIELD::CM field);
	static void SetValue(pn2s::Location address, FIELD::CM field, TYPE_ value);
	static TYPE_ GetValue(pn2s::Location address, FIELD::CH field);
	static void SetValue(pn2s::Location address, FIELD::CH field, TYPE_ value);
	static TYPE_ GetValue(pn2s::Location address, FIELD::GATE field);
	static void SetValue(pn2s::Location address, FIELD::GATE field, TYPE_ value);

	static vector<Device*> Devices(){
		return _devices;
	}

private:

	static DeviceManager _device_manager;
	static vector<Device*> _devices;

};
}
