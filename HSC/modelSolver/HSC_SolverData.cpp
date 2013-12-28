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

hscError HSC_SolverData::PrepareSolver(vector< vector<HSCModel_Base> > &models){
	hscError res = NO_ERROR;

	//Get model's statistic
	hsc_uint modelNumeber = models.size();
	vector<hsc_uint> numHHChannels(modelNumeber);
	vector<hsc_uint> numCaChannels;
	vector<hsc_uint> numSynChannels;
	vector<hsc_uint> numCompartments;

	cout << numHHChannels.size() << endl;
	cout << numCaChannels.size() << endl;

	for (hsc_uint i = 0; i< modelNumeber; i++) {
		hsc_uint modelSize = models[i].size();
		hsc_uint numHHChannels = 0;
		hsc_uint numCaChannels = 0;
		hsc_uint numSynChannels = 0;
		hsc_uint numCompartments = 0;
		for (hsc_uint j = 0; j < modelSize ; j++) {
//			switch(models[i][j].type)
//			{
//			case HSCModel_Base::Type::HHCHannel:
//				num
//				break;
//			}
		}
	}
	res = channels.PrepareSolver(models);
	assert(res);
	res = comps.PrepareSolver(models);
	assert(res);
	return  res;
}
