///////////////////////////////////////////////////////////
//  SolverData.cpp
//  Implementation of the Class SolverData
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "ModelPack.h"


#include <assert.h>

using namespace pn2s;


ModelPack::ModelPack(): _dt(1){

}

ModelPack::~ModelPack(){

}

void ModelPack::AllocateMemory(models::ModelStatistic s, cudaStream_t st){
	stat = s;
	_compsSolver.AllocateMemory(stat, st);
	_chanSolver.AllocateMemory(stat, st);
}

//void ModelPack::AddModel(HSolve* h){
//	/**
//	 * Make Indices and copy data
//	 */
////	int idx = 0;
////	for (int i = 0; i < nModel_in_pack; ++i, m_start++) {
////		h =	reinterpret_cast< HSolve* >( m_start->eref().data());
////		assert(h->HinesMatrix::nCompt_ == nCompt);
////
////		for (int c = 0; c < nCompt; ++c) {
////			Id& compt =	h->HSolvePassive::compartmentId_[c];
////
////			//Assign address
////			Location l = device;
////			l.pack = pack;
////			l.index = idx++; //Assign index of each object in a modelPack
////			compartmentMap[compt] = l;
////		}
////	}
//}

void ModelPack::PrepareSolvers(){
	_compsSolver.PrepareSolver();
	_chanSolver.PrepareSolver();

//	res = _channelsSolver.PrepareSolver(net, _analyzer);

	cudaDeviceSynchronize();
}


void ModelPack::Input()
{
	//For each model in the pack, executes the Process()
	_chanSolver.Input();
	_compsSolver.Input();
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



