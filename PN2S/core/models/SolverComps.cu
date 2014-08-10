///////////////////////////////////////////////////////////
//  SolverComps.cpp
//  Implementation of the Class SolverComps
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverComps.h"
#include "solve.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace pn2s::models;
//CuBLAS variables
//cublasHandle_t _handle;

SolverComps::SolverComps(): _stream(0)
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
	_VMid.AllocateMemory(vectorSize);
	_Ra.AllocateMemory(vectorSize);
	_CmByDt.AllocateMemory(vectorSize);
	_EmByRm.AllocateMemory(vectorSize);
	_InjectBasal.AllocateMemory(vectorSize);
	_InjectVarying.AllocateMemory(vectorSize, 0);

	//Connection to Channels
	_channelIndex.AllocateMemory(vectorSize*2, 0); //Filled with zero

	return Error_PN2S::NO_ERROR;
}

void SolverComps::PrepareSolver(PField<ChannelCurrent, ARCH_>*  channels_current, PField<TYPE_, ARCH_> * Vchannel)
{
	if(_statistic.nCompts_per_model == 0)
		return;

	_channels_current = channels_current;
	_channels_voltage = Vchannel;

	//Copy to GPU
	_hm.Host2Device_Async(_stream);
	_EmByRm.Host2Device_Async(_stream);
	_CmByDt.Host2Device_Async(_stream);
	_channelIndex.Host2Device_Async(_stream);
	_Vm.Host2Device_Async(_stream);
	_InjectBasal.Host2Device_Async(_stream);
	_Constant.Host2Device_Async(_stream);

//	//Create Cublas
//	if ( cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
//	{
//		return Error_PN2S(Error_PN2S::CuBLASError,
//				"CUBLAS initialization failed");
//	}
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
		int* channelIndex,
		ChannelCurrent* channels_current,
		unsigned int size,
		TYPE_ dt)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){ //For each compartment

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
//    	// diagonal (a) = below (c) + GkSum
//
//    	unsigned int pos_c = pos_a - !!(pos_localIndex);//C is one back of the A,
//    	    											//if A is the first item, then C is same as A
//    	    											//With this trick we eliminate wrap division
    	//Find location of A and C in the matrix
		unsigned int pos_matrix = (unsigned int)(idx / nCompt) * nCompt * nCompt;
		unsigned int pos_localIndex = idx % nCompt;

		unsigned int pos_a = pos_matrix + pos_localIndex * (nCompt+1) ;

    	hm[pos_a] = constants[idx] + GkSum; //TODO: Check this value for more complex models
    	rhs[idx] = vm[idx] * cmByDt[idx] + emByRm[idx] + GkEkSum;

    	//Injects from basal or varying resources
    	rhs[idx] += inject_basal[idx] + inject_varying[idx];
    }
}


__global__ void update_vm(TYPE_* vm, TYPE_* vmid, int* channelIndex, TYPE_* channels_voltage, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
    	vm[idx] = 2.0 * vmid[idx]- vm[idx];
    	if(channelIndex[idx << 1])
		{
			size_t pos = channelIndex[idx << 1 | 0x01];
			for ( int i = 0; i < channelIndex[idx << 1]; ++i)
				channels_voltage[pos+i] = vm[idx];
		}
    }
}

void SolverComps::Input()
{
	_InjectVarying.Host2Device();
}

void SolverComps::Process()
{
	uint vectorSize = _statistic.nModels * _statistic.nCompts_per_model;

	dim3 threads, blocks;
	threads=dim3(min((vectorSize&0xFFFFFFC0)|0x20,256), 1); //TODO: Check
	blocks=dim3(max(vectorSize / threads.x,1), 1);

//	_hm.print();
//	_rhs.print();
//	_Constant.print();
//	_channels_current->Device2Host();
//	_channels_current->print();

	update_rhs <<<blocks, threads,0, _stream>>> (
			_hm.device,
			_rhs.device,
			_Vm.device,
			_Constant.device,
			_statistic.nCompts_per_model,
			_CmByDt.device,
			_EmByRm.device,
			_InjectBasal.device,
			_InjectVarying.device,
			_channelIndex.device,
			_channels_current->device,
			vectorSize,
			_statistic.dt);
	assert(cudaSuccess == cudaGetLastError());

	cudaStreamSynchronize(_stream);

//	_hm.Device2Host();
//	_hm.print();
//	_rhs.Device2Host();
//	_rhs.print();
	assert(!dsolve_batch (_hm.device, _rhs.device, _VMid.device, _statistic.nCompts_per_model, _statistic.nModels, _stream));

//	_Vm.Device2Host();
//	_Vm.print();
//	_VMid.Device2Host();
//	_VMid.print();

	update_vm <<<blocks, threads,0, _stream>>> (
				_Vm.device,
				_VMid.device,
				_channelIndex.device,
				_channels_voltage->device,
				vectorSize);

//	_Vm.Device2Host_Async(_stream);
//	_Vm.print();
//	_VMid.Device2Host();
//	_VMid.print();


	assert(cudaSuccess == cudaGetLastError());
//	cudaStreamSynchronize(_stream);
}


void SolverComps::Output()
{
	_Vm.Device2Host_Async(_stream);
	cudaStreamSynchronize(_stream);
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
			_Vm[index] = value;
			break;
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
