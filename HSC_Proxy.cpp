///////////////////////////////////////////////////////////
//  HSC_Proxy.cpp
//  Implementation of the Class HSC_Proxy
//  Created on:      26-Dec-2013 4:08:07 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSC_Proxy.h"
#include "HSC/Definitions.h"
#include "HSC/modelSolver/HSCModel.h"
#include "HSC/HSC_Manager.h"

/**
 * This method is responsible to create model and get pertinent information from
 * the Shell and send it to Manager
 */
void HSC_Proxy::InsertCompartmentModel(Id seed, const vector< TreeNodeStruct >& tree, double dt){
	uint nCompt = tree.size();
	HSCModel neutral(seed.value());

	neutral.compts.resize(nCompt);
	for (uint i = 0; i < nCompt; ++i) {
		neutral.compts[i].Ra = tree[i].Ra;
		neutral.compts[i].Rm = tree[i].Rm;
		neutral.compts[i].Cm = tree[i].Cm;
		neutral.compts[i].Em = tree[i].Em;
		neutral.compts[i].initVm = tree[i].initVm;
		//TODO: Check it
		neutral.compts[i].children.assign(tree[i].children.begin(), tree[i].children.end());
//		_printVector(tree[i].children.size(), &(tree[i].children[0]));
	}

	HSC_Manager::InsertModel(neutral);
}


void HSC_Proxy::Reinit(){

}


/**
 * If it's the first time to execute, prepare solver
 */
void HSC_Proxy::Process(int id){

}
