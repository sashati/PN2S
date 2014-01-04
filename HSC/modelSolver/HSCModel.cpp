///////////////////////////////////////////////////////////
//  HSCModel.cpp
//  Implementation of the Class HSCModel
//  Created on:      26-Dec-2013 9:48:38 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "HSCModel.h"
#include "HSCModel_HHChannel.h"
#include "HSCModel_Compartment.h"

#include <typeinfo>

HSCModel::HSCModel(uint _id) :
		id(_id) {

	compts.clear();
	hhChannels.clear();
}

HSCModel::~HSCModel() {

}
