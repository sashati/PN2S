///////////////////////////////////////////////////////////
//  SolverChannels.h
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "../../headers.h"
#include "PField.h"
#include "ModelStatistic.h"

namespace pn2s
{
namespace models
{
class SolverChannels
{
private:

	ModelStatistic _m_statistic;
	cudaStream_t _stream;

	PField<TYPE_, ARCH_>*  _Vm;

	PField<TYPE_, ARCH_>  _state;
	PField<int, ARCH_>  _comptIndex;
	PField<ChannelType, ARCH_>  _channel_base;
	PField<ChannelCurrent, ARCH_>  _channel_currents;


public:
	dim3 _threads, _blocks;
	SolverChannels();
	~SolverChannels();
	void AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver(PField<TYPE_, ARCH_>*  Vm);
	void Input();
	void Process();
	void Output();

	PField<ChannelCurrent, ARCH_> * GetFieldChannelCurrents(){return & _channel_currents;}

	double GetDt(){ return _m_statistic.dt;}
	void SetDt(double dt){ _m_statistic.dt = dt;}

	void 	SetGateXParams(int index, vector<double>& params);
	void 	SetGateYParams(int index, vector<double>& params);
	void 	SetGateZParams(int index, vector<double>& params);

	void 	SetValue(int index, FIELD::CH field, TYPE_ value);
	TYPE_ 	GetValue(int index, FIELD::CH field);
};

}
}

