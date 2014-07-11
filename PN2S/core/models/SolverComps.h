///////////////////////////////////////////////////////////
//  SolverComps.h
//  Implementation of the Class SolverComps
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

class SolverComps
{
private:
	ModelStatistic _stat;
	cudaStream_t _stream;

	//Connection Fields
	PField<TYPE_, ARCH_>  _hm;	// Hines Matrices
	PField<TYPE_, ARCH_>  _rhs;	// Right hand side of the equation
	PField<TYPE_, ARCH_>  _Vm;	// Vm of the compartments
	PField<TYPE_, ARCH_>  _VMid;	// Vm of the compartments
	PField<TYPE_, ARCH_>  _Ra;	// Ra of the compartments
	PField<TYPE_, ARCH_>  _CmByDt;	// Cm of the compartments
	PField<TYPE_, ARCH_>  _EmByRm;	// Em of the compartments

	//Channel currents
	PField<int, ARCH_>  _currentIndex; // (NumberOfChannels, Index in _current)
	PField<TYPE_, ARCH_>  _current; // (Gk,Ek)


	void  makeHinesMatrix(models::Model *model, TYPE_ * matrix);// float** matrix, uint nCompt);
	void getValues();

	void updateMatrix();
	void updateVm();

public:
	SolverComps();
	~SolverComps();
	Error_PN2S AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver();
	void Input();
	void Process();
	void Output();

	double GetDt(){ return _stat.dt;}
	void SetDt(double dt){ _stat.dt = dt;}

	void 	SetA(int index, int row, int col, TYPE_ value);
	TYPE_ GetA(int n,int i, int j){return _hm[n*_stat.nCompts*_stat.nCompts+i*_stat.nCompts+j];}
//	TYPE_ GetRHS(int n,int i){return _rhs[n*nComp+i];}
	void 	SetValue(int index, FIELD::TYPE field, TYPE_ value);
	TYPE_ 	GetValue(int index, FIELD::TYPE field);
	void AddChannelCurrent(int index, TYPE_ gk, TYPE_ ek);
};

}
}

