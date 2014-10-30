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


size_t SolverComps::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_statistic = s;
	_stream = stream;

	if(_statistic.nCompts_per_model == 0)
		return Error_PN2S::NO_ERROR;

	size_t modelSize = s.nCompts_per_model*s.nCompts_per_model;
	size_t vectorSize = s.nModels * s.nCompts_per_model;

	size_t val = 0;
	val += _hm.AllocateMemory(modelSize*s.nModels);
	val += _rhs.AllocateMemory(vectorSize);
	val += _Vm.AllocateMemory(vectorSize);
	val += _Constant.AllocateMemory(vectorSize);
	val += _Ra.AllocateMemory(vectorSize);
	val += _CmByDt.AllocateMemory(vectorSize);
	val += _EmByRm.AllocateMemory(vectorSize);
	val += _InjectBasal.AllocateMemory(vectorSize);
	val += _InjectVarying.AllocateMemory(vectorSize, 0);
	val += _ext_curr_gh_gkek.AllocateMemory(vectorSize, 0);

	//Connection to Channels
	val += _channelIndex.AllocateMemory(vectorSize, 0); //Filled with zero

	return val;
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
	_threads=dim3(256);
	_blocks=dim3( ceil(vectorSize / (double)_threads.x));
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
    	TYPE_ hm_local;
    	TYPE_ rhs_local;

    	//Find location of A and C in the matrix
		unsigned int pos_a = idx * nCompt + idx % nCompt;

    	//Injects from basal or varying resources
    	// add external current
		rhs_local = inject_basal[idx] + inject_varying[idx] +
				ext_curr_gk_ekgk[idx].y +
				vm[idx] * cmByDt[idx] + emByRm[idx];
    	hm_local = ext_curr_gk_ekgk[idx].x + constants[idx];

		size_t pos = channelIndex[idx].y;
		for ( int i = 0; i < channelIndex[idx].x; ++i)
		{
			hm_local   += ch_curr_gk_ek[pos].x;
			rhs_local += ch_curr_gk_ek[pos].x * ch_curr_gk_ek[pos].y;
			pos++;
		}
    	__syncthreads();
    	hm[pos_a] = hm_local;
    	rhs[idx] = rhs_local;
    }
}

double SolverComps::Input()
{
	clock_t	start_time = clock();
	_InjectVarying.Host2Device_Async(_stream);
	_ext_curr_gh_gkek.Host2Device_Async(_stream);
	return ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
}


double SolverComps::Process()
{

	clock_t	start_time = clock();

	uint vectorSize = _statistic.nModels * _statistic.nCompts_per_model;

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

	SolverMatrix<TYPE_,ARCH_>::fast_solve(
			_hm.device, _rhs.device, _Vm.device,
			_statistic.nCompts_per_model, _statistic.nModels, _stream);
	assert(cudaSuccess == cudaGetLastError());

	return ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
}


double SolverComps::Output()
{
	clock_t	start_time = clock();

	_Vm.Device2Host_Async(_stream);
	_ext_curr_gh_gkek.Fill(0.0);
	_InjectVarying.Fill(0.0);

	return ( std::clock() - start_time ) / (double) CLOCKS_PER_SEC;
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
//	cudaStreamSynchronize(_stream);
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
