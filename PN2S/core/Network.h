#pragma once

#include "../headers.h"
#include "models/Neuron.h"
#include "models/Compartment.h"
//#include "ModelPack.h"
//#include "solvers/SolverChannels.h"
//#include "solvers/SolverComps.h"
//#include "models/Model.h"
using namespace pn2s::models;

namespace pn2s
{

class Network
{
private:
	double _dt;
	uint _gid;

public:

	Network();
	virtual ~Network();

	vector<Neuron >* _neurons;
	vector<Compartment >* _comps;

	typename Neuron::itr 		RegisterNeuron(int gid);
//	typename Compartment<T>::itr 	RegisterCompartment(int gid);
//	double GetDt(){ return _dt;}
//	void SetDt(double dt){ _dt = dt;}
//
//
//	Error_PN2S Setup(double dt);
//	Error_PN2S PrepareSolver( vector<models::Model<T> > & _models,  double dt);
//	ModelPack<T,arch>* FindModelPack(hscID_t id);
//	void Process(ModelPack<T,arch>* data);
};

}

