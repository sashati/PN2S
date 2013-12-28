///////////////////////////////////////////////////////////
//  HSC_SolverData.h
//  Implementation of the Classes related to HSC_Solver
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A73EC9AE3_8C2D_45e3_AE47_0F3CC8B2E661__INCLUDED_)
#define A73EC9AE3_8C2D_45e3_AE47_0F3CC8B2E661__INCLUDED_
#include "HSCModel_Base.h"
#include "HSC_SolverChannels.h"
#include "HSC_SolverComps.h"

class HSC_SolverData
{
public:
	HSC_SolverData();
	virtual ~HSC_SolverData();
	hscError PrepareSolver(map<hscID_t,vector<HSCModel_Base> > &models);

//	vector<vector<HSCModel_Base> > models; //TODO: Encapsulation

private:
	HSC_SolverChannels channels;
	HSC_SolverComps comps;

};
#endif // !defined(A73EC9AE3_8C2D_45e3_AE47_0F3CC8B2E661__INCLUDED_)
