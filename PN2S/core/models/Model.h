///////////////////////////////////////////////////////////
//  Neutral.h
//  Implementation of the Class Neutral
//  Created on:      26-Dec-2013 9:48:37 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
#define A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_

#include "../../headers.h"
#include "HHChannel.h"
#include "Compartment.h"

namespace pn2s
{
namespace models
{

template <typename T, int arch>
class Model
{
public:
	uint id;

	vector<HHChannel<T,arch> > hhChannels;
	vector<Compartment<T,arch> > compts;

	Model(uint _id);
	virtual ~Model();
};

}
}
#endif // !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
