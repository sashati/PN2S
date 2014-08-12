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
// tbb related header files
#include "tbb/flow_graph.h"
#include "tbb/atomic.h"
#include "tbb/tick_count.h"
#include "core/models/ModelStatistic.h"
#include <assert.h>
#include "DeviceManager.h"
//Moose models
//#include "../HSolve.h"

using namespace pn2s;
using namespace std;
using namespace tbb;
using namespace tbb::flow;

using namespace pn2s::models;

extern std::map< unsigned int, pn2s::Location > locationMap; //Locates in DeviceManager


Device::Device(int _id): id(_id), _dt(1), nstreams(DEFAULT_STREAM_NUMBER){
	_modelPacks.clear();

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
	nstreams = DEFAULT_STREAM_NUMBER;
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
//	cudaDeviceReset();
}

Error_PN2S Device::AllocateMemory(double dt, vector<unsigned int>& ids, vector<int2>& statistic, size_t start, size_t end){
	cudaSetDevice(id);
	_dt = dt;

	//Distribute model into packs
	int2* st_start = &statistic[start];
	unsigned int* ids_start = &ids[start];
	size_t nModel = end - start +1;

	if(nModel <= 0)
		return Error_PN2S::EMPTY;

	//TODO: Minimum size for models and change strategy for distribution of models

	//If number of models are less than streams, reduce stream number
	if (nstreams > nModel)
	{
		for (int i = nModel; i < nstreams; i++)
		{
			cudaStreamDestroy(streams[i]);
		}
		nstreams = nModel;
	}

	//Each stream for one Modelpack
	size_t nModel_in_pack = nModel/nstreams;
	_modelPacks.resize(nstreams);
	for (int pack = 0; pack < nstreams; ++pack) {
		//Check nComp for each compartments and update it's fields
		if(pack == nstreams-1) //Last one carries extra parts
			nModel_in_pack += nModel%nstreams;

		/**
		 * Create statistic and Allocate memory for Modelpacks
		 */

		size_t nCompt = 0;
		size_t numberOfChannels = 0;
		for (int i = 0; i < nModel_in_pack; ++i) {
			if(nCompt == 0)
				nCompt = st_start[i].x; //First set
			else
				assert(nCompt == st_start[i].x); //Check for others

			numberOfChannels += st_start[i].y;
		}

		ModelStatistic stat(dt, nModel_in_pack, nCompt, numberOfChannels);

		_modelPacks[pack].AllocateMemory(stat,streams[pack%nstreams]);
		_modelPacks[pack].models.assign(ids_start,ids_start+nModel_in_pack);

		st_start += nModel_in_pack;
		ids_start += nModel_in_pack;
	}
	return Error_PN2S::NO_ERROR;
}

void Device::PrepareSolvers(){
	cudaSetDevice(id);
	for(vector<ModelPack>::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it) {
		it->PrepareSolvers();
	}
}


/**
 * Multithread tasks section
 */


//#ifdef PN2S_DEBUG

struct input_body {
	ModelPack* operator()( ModelPack *m ) {
//		_D(std::cout<< "Input" << m<<endl<<flush);
		m->Input();
        return m;
    }
};

struct process_body{
	ModelPack* operator()( ModelPack* m) {
//		_D(std::cout<< "Process" << m<<endl<<flush);
		m->Process();
        return m;
    }
};

struct output_body{
	ModelPack* operator()( ModelPack* m ) {
//		_D(std::cout<< "Output" << m<<endl<<flush);
		m->Output();
        return m;
    }
};


void Device::Process()
{
	cudaSetDevice(id);

	uint model_n = _modelPacks.size();
	if(model_n < 1)
		return;

//#define USE_TBB
#ifdef USE_TBB
	graph scheduler;

	broadcast_node<ModelPack*> broadcast(scheduler);
	function_node< ModelPack*, ModelPack* > input_node(scheduler, 1, input_body());
	function_node< ModelPack*, ModelPack* > process_node(scheduler, 1, process_body() );
	function_node< ModelPack*, ModelPack* > output_node(scheduler, 1, output_body() );
	queue_node< ModelPack* > iq_node(scheduler);
	queue_node< ModelPack* > oq_node(scheduler);
	make_edge( broadcast, input_node );
	make_edge( input_node, iq_node);
	make_edge( iq_node, process_node);
	make_edge( process_node, oq_node);
	make_edge( oq_node, output_node);


	for (vector<ModelPack >::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		broadcast.try_put(&(*it));
	}
	scheduler.wait_for_all();
#else

	for (vector<ModelPack>::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		it->Input();
	}
	for (vector<ModelPack >::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		it->Process();
	}
	for (vector<ModelPack >::iterator it = _modelPacks.begin(); it != _modelPacks.end(); ++it)
	{
		it->Output();
	}

#endif
}

void Device::Sync()
{
	cudaSetDevice(id);
	cudaDeviceSynchronize();
}
