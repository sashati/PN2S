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
	PField<TYPE_, ARCH_>  _state;
	PField<TYPE_, ARCH_>  _gbar;
	PField<TYPE_, ARCH_>  _xPower;
	PField<TYPE_, ARCH_>  _yPower;
	PField<TYPE_, ARCH_>  _zPower;
	PField<TYPE_, ARCH_>  _gk;
	PField<TYPE_, ARCH_>  _ek;


public:
	SolverChannels();
	~SolverChannels();

	void AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver();
	void Input();
	void Process();
	void Output();

	double GetDt(){ return _m_statistic.dt;}
	void SetDt(double dt){ _m_statistic.dt = dt;}

	void 	SetValue(int index, FIELD::TYPE field, TYPE_ value);
	TYPE_ 	GetValue(int index, FIELD::TYPE field);
};

}
}

