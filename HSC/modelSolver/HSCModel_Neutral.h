///////////////////////////////////////////////////////////
//  HSCModel_Neutral.h
//  Implementation of the Class HSCModel_Neutral
//  Created on:      26-Dec-2013 9:48:37 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
#define A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_

#include "Definitions.h"
#include "HSCModel_HHChannel.h"
#include "IHSCModel_Element.h"

class HSCModel_Neutral
{

public:
	vector<IHSCModel_Element> elements;

	HSCModel_Neutral();
	virtual ~HSCModel_Neutral();
	void AddElement();

};
#endif // !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
