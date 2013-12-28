///////////////////////////////////////////////////////////
//  HSCModel.h
//  Implementation of the Class HSCModel
//	It's a container that holds elements of a model
//  Created on:      26-Dec-2013 4:20:44 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(ADB51230C_974E_4161_BFBD_474969301884__INCLUDED_)
#define ADB51230C_974E_4161_BFBD_474969301884__INCLUDED_
#include "../Definitions.h"

enum HSCModelType{
	HHCHannel,
	CaChannel,
	CustomChannel,
	SynapticChannel,
	Compartment
};

class HSCModelStatistic{
private:
	uint getSum(uint* v, uint size){
		uint sum = 0;
		for(uint i = 0; i< size;i++)
			sum+=v[i];
		return sum;
	}

public:
	vector<uint> numHHChannels;
	vector<uint> numCaChannels;
	vector<uint> numSynChannels;
	vector<uint> numCustomChannels;
	vector<uint> numCompartments;
	void resize(uint numOfModels){
		numHHChannels.resize(numOfModels,0);
		numCaChannels.resize(numOfModels,0);
		numSynChannels.resize(numOfModels,0);
		numCustomChannels.resize(numOfModels,0);
		numCompartments.resize(numOfModels,0);
	}
	uint GetSumHHChannels(){
		return getSum(&numHHChannels[0], numHHChannels.size());
	}
};


class HSCModel_Base
{
public:
	HSCModelType type;
	unsigned int id;
};
#endif // !defined(ADB51230C_974E_4161_BFBD_474969301884__INCLUDED_)
