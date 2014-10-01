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

	PField<TYPE_>*  _Vm;
	PField<TYPE_>  _state;
	PField<int>  _comptIndex;
	PField<ChannelType>  _channel_base;
	PField<TYPE2_>  _ch_currents_gk_ek;

public:
	dim3 _threads, _blocks;
	SolverChannels();
	~SolverChannels();
	void AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver(PField<TYPE_>*  Vm);
	double Input();
	double Process();
	double Output();

	PField<TYPE2_> * GetFieldChannelCurrents(){return & _ch_currents_gk_ek;}

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

