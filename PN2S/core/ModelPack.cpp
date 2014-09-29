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
	_chanSolver.AllocateMemory(stat, st);
}

void ModelPack::PrepareSolvers(){
	_compsSolver.PrepareSolver(_chanSolver.GetFieldChannelCurrents());
	_chanSolver.PrepareSolver(_compsSolver.GetFieldVm());
	cudaDeviceSynchronize();
}


void ModelPack::Input()
{
//	cudaStreamSynchronize(_st);
	_compsSolver.Input();
	_chanSolver.Input();
}

void ModelPack::Process()
{
	_chanSolver.Process();
	_compsSolver.Process();
}

void ModelPack::Output()
{
	_chanSolver.Output();
	_compsSolver.Output();
}



