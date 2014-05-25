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

// workaround issue between gcc >= 4.6 and cuda 6.0
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=6)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#include <Eigen/Core>

namespace pn2s
{
namespace models
{

template <typename T>
class Model
{
public:
	uint id;

	vector<HHChannel<T> > hhChannels;
	vector<Compartment<T> > compts;

	Eigen::MatrixXd matrix;

	Model(uint _id);
	Model(Eigen::MatrixXd m, uint _id);
	virtual ~Model();
};

}
}
#endif // !defined(A33409E93_3943_416a_AEEF_DDB798B0DF86__INCLUDED_)
