///////////////////////////////////////////////////////////
//  SolverChannels.cpp
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverChannels.h"
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace pn2s::models;

SolverChannels::SolverChannels(): _stream(0)
{
}

SolverChannels::~SolverChannels()
{
}


void SolverChannels::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_m_statistic = s;
	_stream = stream;

	if(_m_statistic.nChannels_all == 0)
		return;

	_gbar.AllocateMemory(_m_statistic.nChannels_all);
	_x.AllocateMemory(_m_statistic.nChannels_all);
	_y.AllocateMemory(_m_statistic.nChannels_all);
	_z.AllocateMemory(_m_statistic.nChannels_all);
	_xPower.AllocateMemory(_m_statistic.nChannels_all);
	_yPower.AllocateMemory(_m_statistic.nChannels_all);
	_zPower.AllocateMemory(_m_statistic.nChannels_all);
	_gk.AllocateMemory(_m_statistic.nChannels_all);
	_ek.AllocateMemory(_m_statistic.nChannels_all);
}

void SolverChannels::PrepareSolver()
{
	if(_m_statistic.nChannels_all)
	{
		//Copy to GPU
		_gbar.Host2Device();
		_x.Host2Device();
		_y.Host2Device();
		_z.Host2Device();
		_xPower.Host2Device();
		_yPower.Host2Device();
		_zPower.Host2Device();
		_gk.Host2Device();
		_ek.Host2Device();
	}
}

/**
 * KERNELS
 */
__global__ void advanceChannels(TYPE_* _state, size_t size, TYPE_ dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){

    	_state[idx]++;
    }
}

void SolverChannels::Input()
{

}

void SolverChannels::Process()
{
	dim3 threads, blocks;
	threads=dim3(min((int)(_m_statistic.nChannels_all&0xFFFFFFC0)|0x20,256), 1);
	blocks=dim3(max((int)(_m_statistic.nChannels_all / threads.x),1), 1);

	advanceChannels <<<blocks, threads,0, _stream>>> (
			_x.device,
			_m_statistic.nChannels_all,
			_m_statistic.dt);
	assert(cudaSuccess == cudaGetLastError());

	_x.Device2Host();
	_x.print();

}


void SolverChannels::Output()
{
//	_Vm.Device2Host_Async(_stream);
//	cudaStreamSynchronize(_stream);
}

void SolverChannels::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::CH_GBAR:
			_gbar[index] = value;
			break;
		case FIELD::CH_X_POWER:
			_xPower[index] = value;
			break;
		case FIELD::CH_Y_POWER:
			_yPower[index] = value;
			break;
		case FIELD::CH_Z_POWER:
			_zPower[index] = value;
			break;
	}
}

TYPE_ SolverChannels::GetValue(int index, FIELD::TYPE field)
{
	switch(field)
	{
		case FIELD::CH_GBAR:
			return _gbar[index];
		case FIELD::CH_X_POWER:
			return _xPower[index];
		case FIELD::CH_Y_POWER:
			return _yPower[index];
		case FIELD::CH_Z_POWER:
			return _zPower[index];
	}
}
