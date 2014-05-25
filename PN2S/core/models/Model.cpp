///////////////////////////////////////////////////////////
//  Model.cpp
//  Implementation of the Class Model
//  Created on:      26-Dec-2013 9:48:38 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "Model.h"
#include "HHChannel.h"
#include "Compartment.h"

#include <typeinfo>

using namespace pn2s::models;

template <typename T, int arch>
Model<T,arch>::Model(uint _id) :
		id(_id) {

	compts.clear();
	hhChannels.clear();
}

template <typename T, int arch>
Model<T,arch>::~Model() {

}

template class Model<double, ARCH_SM30>;
