///////////////////////////////////////////////////////////
//  HSC_SolverData.cpp
//  Implementation of the Class HSC_SolverData
//  Created on:      27-Dec-2013 7:29:01 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_SolverData.h"
#include "../Definitions.h"
#include <assert.h>


HSC_SolverData::HSC_SolverData(){

}


HSC_SolverData::~HSC_SolverData(){

}

hscError HSC_SolverData::PrepareSolver(map<hscID_t, vector<HSCModel_Base> > &models){
	hscError res = NO_ERROR;

	//Get model's statistic
	HSCModelStatistic st;
	uint modelNumeber = models.size();
	st.resize(modelNumeber);

	for (uint i = 0; i< modelNumeber; i++) {
		uint modelSize = models[i].size();
		for (uint j = 0; j < modelSize ; j++) {
			//TODO: Pars trees
			switch(models[i][j].type)
			{
			case HHCHannel:
				st.numHHChannels[i]++;
				break;
			}
		}
	}
	res = channels.PrepareSolver(models,st);
//	assert(1);
	res = comps.PrepareSolver(models, st);
//	assert(res);
	return  res;
}
