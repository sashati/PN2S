///////////////////////////////////////////////////////////
//  HSC_Solver.h
//  Implementation of the Class HSC_Solver
//  Created on:      26-Dec-2013 4:21:51 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(FCD33A96_9E58_4bec_BA66_91CF4FD383BD__INCLUDED_)
#define FCD33A96_9E58_4bec_BA66_91CF4FD383BD__INCLUDED_

#include "HSC_ModelPack.h"
#include "HSCModel.h"
#include "../Definitions.h"

class HSC_Solver
{
public:
	static vector<HSC_ModelPack> modelPacks;
	static hscError Setup();
	static hscError CreateModelPack(vector<HSCModel> _models);
	static hscError SendDataToGPU();
//	void DoProcess(HSC_Device* d);

};

#endif // !defined(FCD33A96_9E58_4bec_BA66_91CF4FD383BD__INCLUDED_)
