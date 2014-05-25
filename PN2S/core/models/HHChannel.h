///////////////////////////////////////////////////////////
//  HHChannel.h
//  Implementation of the Class HHChannel
//  Created on:      26-Dec-2013 4:20:54 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A904F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_)
#define A904F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_

namespace pn2s
{
namespace models
{

template <typename T, int arch>
class HHChannel
{
public:
	HHChannel();
	virtual ~HHChannel();

	double Gbar;
	double Ek;
	double Gk;
	double Ik;
	double Vm;
};

}
}
#endif // !defined(A904F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_)
