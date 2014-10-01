///////////////////////////////////////////////////////////
//  SolverComps.h
//  Implementation of the Class SolverComps
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

class SolverComps
{
private:
	ModelStatistic _statistic;
	cudaStream_t _stream;

	//Connection Fields
	PField<TYPE_>  _Vm;	// Vm of the compartments
	PField<TYPE_>  _Constant;
	PField<TYPE_>  _hm;	// Hines Matrices
	PField<TYPE_>  _rhs;	// Right hand side of the equation
	PField<TYPE_>  _Ra;	// Ra of the compartments
	PField<TYPE_>  _CmByDt;	// Cm of the compartments
	PField<TYPE_>  _EmByRm;	// Em of the compartments
	PField<TYPE_>  _InjectBasal;
	PField<TYPE_>  _InjectVarying;
	PField<TYPE2_>  _ext_curr_gh_gkek;

	//Channel currents
	PField<int2>  _channelIndex; 	// (NumberOfChannels, Index in _current)
	PField<TYPE2_>*  _channels_current;	// Refer to channel kernel

	void getValues();

	void updateMatrix();
	void updateVm();

public:
	dim3 _threads, _blocks;

	SolverComps();
	~SolverComps();
	Error_PN2S AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver(PField<TYPE2_>*  channels_current);
	double Input();
	double Process();
	double Output();

	double GetDt(){ return _statistic.dt;}
	void SetDt(double dt){ _statistic.dt = dt;}

	void 	SetHinesMatrix(int n, int row, int col, TYPE_ value);
	TYPE_ GetA(int n,int i, int j){return _hm[n*_statistic.nCompts_per_model*_statistic.nCompts_per_model+i*_statistic.nCompts_per_model+j];}
	void 	SetValue(int cmpt_index, FIELD::CM field, TYPE_ value);
	TYPE_ 	GetValue(int cmpt_index, FIELD::CM field);
	void ConnectChannel(int cmpt_index,  int ch_index);
	PField<TYPE_> * GetFieldVm(){return & _Vm;}
	void AddExternalCurrent( int index, TYPE_ Gk, TYPE_ GkEk);
};

}
}

