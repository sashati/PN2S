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

#include "../../../basecode/header.h" //For seed vector

namespace pn2s
{

class ModelPack
{
private:
public:
	double _dt;
	vector<Id> models;

	models::ModelStatistic stat;

	ModelPack();
	virtual ~ModelPack();

	double GetDt(){ return _dt;}
//	void SetDt(double dt){ _dt = dt;
//		_compsSolver.SetDt(dt);
//	}

	Error_PN2S AllocateMemory( models::ModelStatistic s, cudaStream_t st);
//	void AddModel(HSolve* h);
	Error_PN2S PrepareSolvers();

	void Process();
	void Output();
	void Input();

	models::SolverComps _compsSolver; //TODO Encapsulation
	//	solvers::SolverChannels<TYPE_,CURRENT_ARCH> _channelsSolver; //TODO Encapsulation

//	friend ostream& operator<<(ostream& out, ModelPack<T,arch>& dt)
//	{
//	    out << "ModelPakc";
//	    return out;
//	}

};

}

