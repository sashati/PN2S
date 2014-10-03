///////////////////////////////////////////////////////////
//  SolverData.cpp
//  Implementation of the Class SolverData
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "ModelPack.h"


#include <assert.h>

using namespace pn2s;


ModelPack::ModelPack(): _dt(1), _st(0){

}

ModelPack::~ModelPack(){

}

void ModelPack::AllocateMemory(models::ModelStatistic s, cudaStream_t st){
	stat = s;
	_st = st;
	_compsSolver.AllocateMemory(stat, st);
//	_chanSolver.AllocateMemory(stat, st);
	_gateSolver.AllocateMemory(stat, st);
}

void ModelPack::PrepareSolvers(){
	_compsSolver.PrepareSolver(_gateSolver.GetFieldChannelCurrents());
//	_chanSolver.PrepareSolver(_compsSolver.GetFieldVm());
	_gateSolver.PrepareSolver(_compsSolver.GetFieldVm());
	cudaDeviceSynchronize();
}


double ModelPack::Input()
{
	double t = 0;
	t += _compsSolver.Input();
//	t += _chanSolver.Input();
	return t;
}

double ModelPack::Process()
{
	cudaStreamSynchronize(_st);
	double t = 0;
//	t += _chanSolver.Process();
	t += _gateSolver.Process();
	t += _compsSolver.Process();
	return t;
}

double ModelPack::Output()
{
	double t = 0;
//	t +=_chanSolver.Output();
	t += _compsSolver.Output();
	return t;
}



