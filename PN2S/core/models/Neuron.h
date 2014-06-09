#pragma once

#include "../../headers.h"
#include "Compartment.h"
//#include "HHChannel.h"
//#include "Compartment.h"
//#include "Matrix.h"

namespace pn2s
{
namespace models
{

class Neuron
{
	vector<Compartment > * _compt;
	int _compt_base;
	int _compt_size;

public:
	unsigned int gid;

	typedef std::vector<Neuron> Vec;
	typedef typename Vec::iterator itr;

	Neuron(unsigned int _gid, vector<Compartment> *, int idx);
	Neuron( const Neuron& other );
	virtual ~Neuron();

	typename Compartment::itr RegisterCompartment(int gid);
};

}
}
