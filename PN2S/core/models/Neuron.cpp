///////////////////////////////////////////////////////////
//  Model.cpp
//  Implementation of the Class Model
//  Created on:      26-Dec-2013 9:48:38 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "Neuron.h"
//#include "HHChannel.h"
//#include "Compartment.h"

#include <typeinfo>

using namespace pn2s::models;

Neuron::Neuron(unsigned int _gid, vector<Compartment > * _vec, int idx):
	gid(_gid), _compt(_vec), _compt_base(idx), _compt_size(0)
{

}

Neuron::~Neuron() {

}

//Copy constractor
Neuron::Neuron( const Neuron& other )
{

}

typename Compartment::itr Neuron::RegisterCompartment(int gid)
{
	Compartment c(gid);	//, _compt, _compt_base+_compt_size);
	_compt_size++;

//	_compt->push_back(c); ?????????????????????????????? Use external object!
	return _compt->end();
}

