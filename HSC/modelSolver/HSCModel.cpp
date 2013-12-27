///////////////////////////////////////////////////////////
//  HSCModel.cpp
//  Implementation of the Class HSCModel
//  Created on:      26-Dec-2013 4:20:44 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSCModel.h"


HSCModel::HSCModel(){

}



HSCModel::~HSCModel(){

}


void HSCModel::AddElement(IHSCModel_Element &e){
	elements.push_back(e);
}
