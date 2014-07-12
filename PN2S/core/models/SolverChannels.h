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
	ModelStatistic _stat;
	cudaStream_t _stream;

	//Connection Fields
	PField<TYPE_, ARCH_>  _gbar; //Select between 1,2,3,4,N power function
	PField<TYPE_, ARCH_>  _current; // (Gk,Ek)
	PField<TYPE_, ARCH_>  _hm;	// Hines Matrices

	//Channel currents
	PField<int, ARCH_>  _currentIndex; // (NumberOfChannels, Index in _current)


	void  makeHinesMatrix(models::Model *model, TYPE_ * matrix);// float** matrix, uint nCompt);
	void getValues();

	void updateMatrix();
	void updateVm();

public:
	SolverChannels();
	~SolverChannels();
	Error_PN2S AllocateMemory(models::ModelStatistic& s, cudaStream_t stream);
	void PrepareSolver();
	void Input();
	void Process();
	void Output();

	double GetDt(){ return _stat.dt;}
	void SetDt(double dt){ _stat.dt = dt;}

	void 	SetA(int index, int row, int col, TYPE_ value);
	void 	SetValue(int index, FIELD::TYPE field, TYPE_ value);
	TYPE_ 	GetValue(int index, FIELD::TYPE field);
	void AddChannelCurrent(int index, TYPE_ gk, TYPE_ ek);
};

}
}

