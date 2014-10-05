///////////////////////////////////////////////////////////
//  Device.cpp
//  Implementation of the Class Device
//  Created on:      26-Dec-2013 4:18:01 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#include "Device.h"
#include "headers.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

using namespace std;

#include "core/models/ModelStatistic.h"
#include "DeviceManager.h"
using namespace pn2s;
using namespace pn2s::models;

extern std::map< unsigned int, pn2s::Location > locationMap; //Locates in ResourceManager


Device::Device(int _id): id(_id), _dt(1), nstreams(Parameters::MAX_STREAM_NUMBER){
	_modelPacks.clear();
	cudaSetDevice(id);
	/**
	 * Check configuration
	 */
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, _id);

	if ((1 == deviceProp.major) && (deviceProp.minor < 1))
	{
		printf("%s does not have Compute Capability 1.1 or newer.  Reducing workload.\n", deviceProp.name);
	}
//	cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
//	nstreams = deviceProp.multiProcessorCount;
	nstreams = Parameters::MAX_STREAM_NUMBER;
	/**
	 * Device initialization
	 */
	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
	//Create Streams
	for (int i = 0; i < nstreams; ++i) {
		cudaStreamCreate(&streams[i]);
	}
}

Device::~Device(){
//	if(_modelPacks.size())
//		cudaDeviceReset();
}

void Device::Destroy(){
//	for (int i = 0; i < nstreams; i++)
//	{
//		cudaStreamDestroy(streams[i]);
//	}
}

Error_PN2S Device::AllocateMemory(vector<Model_pack_info> &mps, double dt ){

	if(mps.empty())
		return Error_PN2S::EMPTY;

	_dt = dt;

	size_t nPack = mps.size();

	//Distribute Modelpacks over streams
	_modelPacks.resize(nPack);
	size_t val_sum = 0;
	for (int pack = 0; pack < nPack; ++pack) {
		/**
		 * Create statistic and Allocate memory for Modelpacks
		 */
		size_t nCompt = mps[pack][0].nCompt;
		size_t nChannels = 0;
		size_t nGates = 0;
		for (Model_pack_info::iterator m = mps[pack].begin(); m != mps[pack].end(); m++)
		{
			nChannels += m->nChannel;
			nGates += m->nGates;
			_modelPacks[pack].models.push_back( m->id);
		}
		_modelPacks[pack]._device_id = id;
		ModelStatistic stat(dt, mps[pack].size(), nCompt, nChannels, nGates);
		size_t val = _modelPacks[pack].AllocateMemory(stat,streams[pack%nstreams]);
		cout << "Device" << id << "\t ModelPack " << pack << ": " <<
				/*std::fixed <<std::setprecision(2) <<*/ (double)val/1024.0/1024.0 << " MB"<<
				"(" << stat.nModels << " models with " <<stat.nCompts_per_model << " compartments)" << endl;
		val_sum += val;
	}
	cout << "Device" << id << "\t Total: " << std::fixed <<std::setprecision(2) << (double)val_sum/1024.0/1024.0 << " MB"<<endl;
	return Error_PN2S::NO_ERROR;
}

void Device::PrepareSolvers(){
	for(vector<ModelPack>::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it) {
		it->PrepareSolvers();
	}
}

void Device::Process()
{

	uint model_n = _modelPacks.size();
	if(model_n < 1)
		return;

	double time = 0;
	for (vector<ModelPack>::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		time += it->Input();
	}

	for (vector<ModelPack >::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		time += it->Process();
	}
	for (vector<ModelPack >::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		time += it->Output();
	}
//	cudaDeviceSynchronize();

//	cout << time << endl;
}

