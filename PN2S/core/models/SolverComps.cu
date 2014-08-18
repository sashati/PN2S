///////////////////////////////////////////////////////////
//  SolverComps.cpp
//  Implementation of the Class SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverComps.h"
#include "SolverMatrix.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace pn2s::models;

SolverComps::SolverComps(): _stream(0), _channels_current(0)
{

}

SolverComps::~SolverComps()
{
}


Error_PN2S SolverComps::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_statistic = s;
	_stream = stream;

	if(_statistic.nCompts_per_model == 0)
		return Error_PN2S::NO_ERROR;

	size_t modelSize = s.nCompts_per_model*s.nCompts_per_model;
	size_t vectorSize = s.nModels * s.nCompts_per_model;

	_hm.AllocateMemory(modelSize*s.nModels);
	_rhs.AllocateMemory(vectorSize);
	_Vm.AllocateMemory(vectorSize);
	_Constant.AllocateMemory(vectorSize);
	_Ra.AllocateMemory(vectorSize);
	_CmByDt.AllocateMemory(vectorSize);
	_EmByRm.AllocateMemory(vectorSize);
	_InjectBasal.AllocateMemory(vectorSize);
	_InjectVarying.AllocateMemory(vectorSize, 0);
	_externalCurrent.AllocateMemory(vectorSize, 0);

	//Connection to Channels
	_channelIndex.AllocateMemory(vectorSize*2, 0); //Filled with zero

	return Error_PN2S::NO_ERROR;
}

void SolverComps::PrepareSolver(PField<ChannelCurrent, ARCH_>*  channels_current)
{
	int model_dize = _statistic.nCompts_per_model * _statistic.nCompts_per_model;
	if(model_dize == 0)
		return;

	_channels_current = channels_current;

	//Copy to GPU
	_hm.Host2Device_Async(_stream);
	_EmByRm.Host2Device_Async(_stream);
	_CmByDt.Host2Device_Async(_stream);
	_channelIndex.Host2Device_Async(_stream);
	_Vm.Host2Device_Async(_stream);
	_InjectBasal.Host2Device_Async(_stream);
	_Constant.Host2Device_Async(_stream);
	_externalCurrent.Host2Device_Async(_stream);

	_threads=dim3(128);
	_blocks=dim3( model_dize / _threads.x + 1);
}

/**
 * 			UPDATE MATRIX
 *
 * RHS = Vm * Cm / ( dt / 2.0 ) + Em/Rm;
 *
 */

__global__ void update_rhs(
		TYPE_* hm,
		TYPE_* rhs,
		TYPE_* vm,
		TYPE_* constants,
		size_t nCompt,
		TYPE_* cmByDt,
		TYPE_* emByRm,
		TYPE_* inject_basal,
		TYPE_* inject_varying,
		ExternalCurrent* external_current,
		int* channelIndex,
		ChannelCurrent* channels_current,
		unsigned int size,
		TYPE_ dt)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){ //For each compartment

    	//Find location of A and C in the matrix
		unsigned int pos_a = idx * nCompt + idx % nCompt;

    	TYPE_ GkSum   = 0.0;
    	TYPE_ GkEkSum = 0.0;
    	if(channelIndex[idx << 1])
    	{
    		size_t pos = channelIndex[idx << 1 | 0x01];
    		for ( int i = 0; i < channelIndex[idx << 1]; ++i)
			{
				GkSum   += channels_current[pos+i]._gk;
				GkEkSum += channels_current[pos+i]._gk * channels_current[pos+i]._ek;
			}
    	}


    	hm[pos_a] = constants[idx] + GkSum;
    	rhs[idx] = vm[idx] * cmByDt[idx] + emByRm[idx] + GkEkSum;

    	//Injects from basal or varying resources
    	rhs[idx] += inject_basal[idx] + inject_varying[idx];

    	// add external current
//    	hm[pos_a] += external_current[idx]._gk;
//    	rhs[idx] += external_current[idx]._gkek;
    }
}

void SolverComps::Input()
{
	_InjectVarying.Host2Device_Async(_stream);
	_externalCurrent.Host2Device_Async(_stream);
}

void SolverComps::Process()
{
	uint vectorSize = _statistic.nModels * _statistic.nCompts_per_model;

//	_Vm.print();
//	_hm.print();
//	_rhs.print();
//	_Constant.print();
//	_channels_current->Device2Host();	_channels_current->print();
//	_externalCurrent.print();
	update_rhs <<<_blocks, _threads,0, _stream>>> (
			_hm.device,
			_rhs.device,
			_Vm.device,
			_Constant.device,
			_statistic.nCompts_per_model,
			_CmByDt.device,
			_EmByRm.device,
			_InjectBasal.device,
			_InjectVarying.device,
			_externalCurrent.device,
			_channelIndex.device,
			_channels_current->device,
			vectorSize,
			_statistic.dt);
	assert(cudaSuccess == cudaGetLastError());

//	_hm.Device2Host();	_hm.print();
//	_rhs.Device2Host();	_rhs.print();

	SolverMatrix<TYPE_,ARCH_>::fast_solve(
			_hm.device, _rhs.device, _Vm.device,
			_statistic.nCompts_per_model, _statistic.nModels, _stream);
	assert(cudaSuccess == cudaGetLastError());

//	_hm.Device2Host();	_hm.print();
//	_rhs.Device2Host();	_rhs.print();

//	_Vm.Device2Host();	_Vm.print();

}


void SolverComps::Output()
{
	_Vm.Device2Host_Async(_stream);
	_externalCurrent.Fill(0.0);
	_InjectVarying.Fill(0.0);
}

/**
 * 		Set/Get values
 */
void SolverComps::AddExternalCurrent( int index, TYPE_ Gk, TYPE_ GkEk)
{
	_externalCurrent[index]._gk += Gk;
	_externalCurrent[index]._gkek += GkEk;
}

void SolverComps::SetValue(int index, FIELD::TYPE field, TYPE_ value)
{
	switch(field)
	{
		case FIELD::CM_BY_DT:
			_CmByDt[index] = value;
			break;
		case FIELD::EM_BY_RM:
			_EmByRm[index] = value;
			break;
		case FIELD::RA:
			_Ra[index] = value;
			break;
		case FIELD::VM:
		case FIELD::INIT_VM:
			_Vm[index] = value;
			break;
		case FIELD::INJECT_BASAL:
			_InjectBasal[index] = value;
			break;
		case FIELD::INJECT_VARYING:
			_InjectVarying[index] = value;
			break;
		case FIELD::CONSTANT:
			_Constant[index] = value;
			break;
		case FIELD::EXT_CURRENT_GK:
			_externalCurrent[index]._gk = value;
			break;
		case FIELD::EXT_CURRENT_EKGK:
			_externalCurrent[index]._gkek = value;
			break;
	}
}

TYPE_ SolverComps::GetValue(int index, FIELD::TYPE field)
{
	switch(field)
	{
		case FIELD::CM_BY_DT:
			return _CmByDt[index];
		case FIELD::EM_BY_RM:
			return _EmByRm[index];
		case FIELD::RA:
			return _Ra[index];
		case FIELD::VM:
			return _Vm[index];
		case FIELD::INIT_VM:
			return _Vm[index];
	}
	return 0;
}

void SolverComps::SetHinesMatrix(int n, int row, int col, TYPE_ value)
{
	_hm[_statistic.nCompts_per_model*_statistic.nCompts_per_model*n + row *_statistic.nCompts_per_model + col] = value;
}

void SolverComps::ConnectChannel(int cmpt_index, int ch_index)
{
	if (_channelIndex[cmpt_index*2] == 0)
		_channelIndex[cmpt_index*2+1] = ch_index;
	_channelIndex[cmpt_index*2]++;
}
