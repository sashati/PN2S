///////////////////////////////////////////////////////////
//  HSC_SolverData.cpp
//  Implementation of the Class HSC_SolverData
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverData.h"
#include "../Definitions.h"
#include <assert.h>


HSC_SolverData::HSC_SolverData(double _dt): 	dt(_dt), _compsSolver(_dt)
{

}


HSC_SolverData::~HSC_SolverData(){

}

hscError HSC_SolverData::PrepareSolver(vector<HSCModel > &net){
	hscError res = NO_ERROR;

	analyzer.ImportNetwork(net);

	res = _compsSolver.PrepareSolver(net, analyzer);
	assert(res==NO_ERROR);
	res = _channelSolver.PrepareSolver(net, analyzer);
	assert(res==NO_ERROR);
	return  res;
}

hscError HSC_SolverData::Process()
{
	_compsSolver.Process();
	return NO_ERROR;
}
