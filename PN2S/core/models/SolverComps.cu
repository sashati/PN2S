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
	_ext_curr_gh_gkek.AllocateMemory(vectorSize, 0);

	//Connection to Channels
	_channelIndex.AllocateMemory(vectorSize, 0); //Filled with zero

	return Error_PN2S::NO_ERROR;
}

void SolverComps::PrepareSolver(PField<TYPE2_>*  channels_current)
{
	int vectorSize = _statistic.nModels * _statistic.nCompts_per_model;
	if(vectorSize == 0)
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
	_ext_curr_gh_gkek.Host2Device_Async(_stream);

	_threads=dim3(128);
	_blocks=dim3( vectorSize / _threads.x + 1);
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
		TYPE2_* ext_curr_gk_ekgk,
		int2* channelIndex,
		TYPE2_* ch_curr_gk_ek,
		unsigned int size,
		TYPE_ dt)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){ //For each compartment

    	//Find location of A and C in the matrix
		unsigned int pos_a = idx * nCompt + idx % nCompt;

    	TYPE_ GkSum   = 0.0;
    	TYPE_ GkEkSum = 0.0;
    	if(channelIndex[idx].x)
    	{
    		size_t pos = channelIndex[idx].y;
    		for ( int i = 0; i < channelIndex[idx].x; ++i)
			{
				GkSum   += ch_curr_gk_ek[pos+i].x;
				GkEkSum += ch_curr_gk_ek[pos+i].x * ch_curr_gk_ek[pos+i].y;
			}
    	}


    	hm[pos_a] = constants[idx] + GkSum;
    	rhs[idx] = vm[idx] * cmByDt[idx] + emByRm[idx] + GkEkSum;

    	//Injects from basal or varying resources
    	rhs[idx] += inject_basal[idx] + inject_varying[idx];

    	// add external current
    	hm[pos_a] += ext_curr_gk_ekgk[idx].x;
    	rhs[idx] += ext_curr_gk_ekgk[idx].y;
    }
}

void SolverComps::Input()
{
	_InjectVarying.Host2Device_Async(_stream);
	_ext_curr_gh_gkek.Host2Device_Async(_stream);
}

void SolverComps::Process()
{
	uint vectorSize = _statistic.nModels * _statistic.nCompts_per_model;

//	_hm.print(25);
//	_rhs.print(5);
//	_Vm.print(5);
//	_channelIndex.print(5);
//	_channels_current->Device2Host();
//	_channels_current->print(5);
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
			_ext_curr_gh_gkek.device,
			_channelIndex.device,
			_channels_current->device,
			vectorSize,
			_statistic.dt);
	assert(cudaSuccess == cudaGetLastError());

//	_hm.Device2Host(); _hm.print(25);
//	_rhs.Device2Host();	_rhs.print(5);
//	_channels_current->Device2Host();	_channels_current->print();

	SolverMatrix<TYPE_,ARCH_>::fast_solve(
			_hm.device, _rhs.device, _Vm.device,
			_statistic.nCompts_per_model, _statistic.nModels, _stream);
//	_Vm.Device2Host();	_Vm.print(5);
	assert(cudaSuccess == cudaGetLastError());

}


void SolverComps::Output()
{
	_Vm.Device2Host_Async(_stream);
	_ext_curr_gh_gkek.Fill(0.0);
	_InjectVarying.Fill(0.0);
}

/**
 * 		Set/Get values
 */

void SolverComps::SetValue(int index, FIELD::CM field, TYPE_ value)
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
			_ext_curr_gh_gkek[index].x = value;
			break;
		case FIELD::EXT_CURRENT_EKGK:
			_ext_curr_gh_gkek[index].y = value;
			break;
	}
}

TYPE_ SolverComps::GetValue(int index, FIELD::CM field)
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
	if (_channelIndex[cmpt_index].x == 0)
		_channelIndex[cmpt_index].y = ch_index;
	_channelIndex[cmpt_index].x++;
}
