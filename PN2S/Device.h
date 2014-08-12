///////////////////////////////////////////////////////////
//  Device.h
//  Implementation of the Class Device
//  Created on:      26-Dec-2013 4:18:01 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#pragma once

#include "headers.h"
#include "core/ModelPack.h"
#include <omp.h>

//Moose models
//#include "../../basecode/header.h"

namespace pn2s
{

class Device
{
public:
	Error_PN2S SelectDevice();
	int id;
	double _dt;

	Device(int _id);
	virtual ~Device();

	vector<ModelPack>& ModelPacks(){ return _modelPacks;}
	int NumberOfModelPacks(){return nstreams;}

	//Generate model packs
	Error_PN2S AllocateMemory(double dt,
			vector<unsigned int>& ids,
			vector<int2 >& m,
			size_t start, size_t end);
	void PrepareSolvers();

	void Destroy();
	void Process();
	void Sync();
private:
	vector<ModelPack> _modelPacks;
	cudaStream_t* streams;
	int nstreams;
//	int _queue_size;
//	void task1_prepareInput(omp_lock_t& _empty_lock,omp_lock_t& _full_lock, int& state);
//	void task2_doProcess(omp_lock_t& _empty_lock_input,	omp_lock_t& _full_lock_input,omp_lock_t& _empty_lock_output, int& state);
//	void task3_prepareOutput(omp_lock_t& _empty_lock, int& state);

};
}

