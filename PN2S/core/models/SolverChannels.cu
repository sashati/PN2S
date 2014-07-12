///////////////////////////////////////////////////////////
//  SolverChannels.cpp
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "SolverChannels.h"
#include "solve.h"

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


Error_PN2S SolverChannels::AllocateMemory(models::ModelStatistic& s, cudaStream_t stream)
{
	_stat = s;
	_stream = stream;

	if(_stat.nCompts == 0)
		return Error_PN2S::NO_ERROR;

	size_t modelSize = s.nCompts*s.nCompts;
	size_t vectorSize = s.nModels * s.nCompts;

	_hm.AllocateMemory(modelSize*s.nModels);
	_rhs.AllocateMemory(vectorSize);
	_Vm.AllocateMemory(vectorSize);
	_VMid.AllocateMemory(vectorSize);
	_Ra.AllocateMemory(vectorSize);
	_CmByDt.AllocateMemory(vectorSize);
	_EmByRm.AllocateMemory(vectorSize);

	_currentIndex.AllocateMemory(vectorSize*2,0);
	_current.AllocateMemory(_stat.nChannels*2);

	return Error_PN2S::NO_ERROR;
}

void SolverChannels::PrepareSolver()
{
	if(_stat.nCompts == 0)
		return;

	//Copy to GPU
	_hm.Host2Device_Async(_stream);
	_EmByRm.Host2Device_Async(_stream);


}

void SolverChannels::Input()
{
//	//Copy to GPU
//	_rhs.Send2Device_Async(_Em,_stream); // Em -> rhs
//	_Rm.Host2Device_Async(_stream);
//	_Vm.Host2Device_Async(_stream);
//	_Cm.Host2Device_Async(_stream);
}

void SolverChannels::Process()
{
//	uint vectorSize = _stat.nModels * _stat.nCompts;
//
//	dim3 threads, blocks;
//	threads=dim3(min((vectorSize&0xFFFFFFC0)|0x20,256), 1); //TODO: Check
//	blocks=dim3(max(vectorSize / threads.x,1), 1);
//
//	update_rhs <<<blocks, threads,0, _stream>>> (
//			_rhs.device,
//			_Vm.device,
//			_CmByDt.device,
//			_EmByRm.device,
//			vectorSize,
//			_stat.dt);
//	assert(cudaSuccess == cudaGetLastError());
//
////	cudaStreamSynchronize(_stream);
//
//	_hm.Device2Host();
//	_hm.print();
//	_rhs.Device2Host();
//	_rhs.print();
//	assert(!dsolve_batch (_hm.device, _rhs.device, _VMid.device, _stat.nCompts, _stat.nModels, _stream));
//
//	update_vm <<<blocks, threads,0, _stream>>> (
//				_Vm.device,
//				_VMid.device,
//				vectorSize);
//
//	assert(cudaSuccess == cudaGetLastError());
////	cudaStreamSynchronize(_stream);
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
	}
}

TYPE_ SolverChannels::GetValue(int index, FIELD::TYPE field)
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


void SolverChannels::AddChannelCurrent(int index, TYPE_ gk, TYPE_ ek)
{
	if (_currentIndex[index*2] == 0)
		_currentIndex[index*2+1] = _current.extraIndex;
	_currentIndex[index*2]++; //Number of Channels

	_current[_current.extraIndex++] = gk;
	_current[_current.extraIndex++] = ek;
}
