///////////////////////////////////////////////////////////
//  SolverChannels.h
//  Implementation of the Class SolverChannels
//  Created on:      27-Dec-2013 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "../../headers.h"
#include "../models/Model.h"
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

	//Connection Fields
//	PField<TYPE_, ARCH_>  _gbar;
	PField<ChannelType, ARCH_>  _channels;

//	PField<TYPE_, ARCH_>  _gk;
//	PField<TYPE_, ARCH_>  _ek;
//
//	PField<uint, ARCH_>  _instant;



public:
	SolverChannels();
	~SolverChannels();
	void AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver();
	void Input();
	void Process(PField<TYPE_, ARCH_>* _Vm);
	void Output();

	double GetDt(){ return _m_statistic.dt;}
	void SetDt(double dt){ _m_statistic.dt = dt;}

	void 	SetGateXParams(int index, vector<double> params);
	void 	SetGateYParams(int index, vector<double> params);
	void 	SetGateZParams(int index, vector<double> params);

	void 	SetValue(int index, FIELD::TYPE field, TYPE_ value);
	TYPE_ 	GetValue(int index, FIELD::TYPE field);
};

}
}

