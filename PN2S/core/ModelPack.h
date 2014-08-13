///////////////////////////////////////////////////////////
//  SolverData.h
//  Implementation of the Classes related to Solver
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once

#include "models/ModelStatistic.h"
#include <cuda.h>
#include "../headers.h"
#include "./models/SolverChannels.h"
#include "./models/SolverComps.h"

//#include "../../../basecode/header.h" //For seed vector

namespace pn2s
{

class ModelPack
{
	cudaStream_t _st;
public:
	models::SolverComps _compsSolver;
	models::SolverChannels _chanSolver;

	double _dt;
	vector<unsigned int> models;
	models::ModelStatistic stat;

	ModelPack();
	virtual ~ModelPack();


	void AllocateMemory( models::ModelStatistic s, cudaStream_t st);
	void PrepareSolvers();

	void Process();
	void Output();
	void Input();

	double GetDt(){ return _dt;}
	models::SolverComps& ComptSolver(){return _compsSolver;}
	models::SolverChannels& ChannelSolver(){return _chanSolver;}

};

}

