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
#include "IHSCModel_Element.h"
#include "HSCModel_HHChannel.h"

class HSCModel
{

public:
	HSCModel();
	virtual ~HSCModel();
	void AddElement(IHSCModel_Element& e);

	vector<IHSCModel_Element> elements;
};
#endif // !defined(ADB51230C_974E_4161_BFBD_474969301884__INCLUDED_)
