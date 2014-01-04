///////////////////////////////////////////////////////////
//  HSCModel_Neutral.h
//  Implementation of the Class HSCModel_Neutral
//  Created on:      26-Dec-2013 9:48:37 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
#define A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_

#include "../Definitions.h"
#include "HSCModel_HHChannel.h"
#include "HSCModel_Compartment.h"

/*
 * This class inherits from vector, there is no destructor call for this function.
 */
class HSCModel
{
public:
	uint id;

	vector<HSCModel_HHChannel> hhChannels;
	vector<HSCModel_Compartment> compts;

	HSCModel(uint _id);
	virtual ~HSCModel();
};
#endif // !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
