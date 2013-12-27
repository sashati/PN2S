///////////////////////////////////////////////////////////
//  HSC_ModelPack.h
//  Implementation of the Class HSC_ModelPack
//  Created on:      26-Dec-2013 4:20:59 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A6E7F5F2E_5B3D_424f_B594_59714CF7F8D1__INCLUDED_)
#define A6E7F5F2E_5B3D_424f_B594_59714CF7F8D1__INCLUDED_

#include "../Definitions.h"
#include "HSCModel.h"

class HSC_ModelPack
{

public:
	HSC_ModelPack();
	virtual ~HSC_ModelPack();

private:
	vector<HSCModel> modelPack;

};
#endif // !defined(A6E7F5F2E_5B3D_424f_B594_59714CF7F8D1__INCLUDED_)
