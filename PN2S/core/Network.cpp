#include "Network.h"
#include <assert.h>

using namespace pn2s;

Network::Network()
{
	_dt = 1; //1ms
	_neurons = new vector<Neuron >();
	_comps = new vector<Compartment >();
}

Network::~Network()
{

}

typename Neuron::itr Network::RegisterNeuron(int gid)
{
	int size = _comps->size();
	Neuron n(gid, _comps, size-1);
	_neurons->push_back(n);
	return _neurons->end();
}

//template <typename T, int arch>
//typename Compartment<T>::itr Network<T, arch>::RegisterCompartment(int gid)
//{
//	int size = _comps.size();
//	_comps.resize(size+1);
//	_comps.end()->gid = gid;
//	return _comps.end();
//}


