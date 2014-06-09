///////////////////////////////////////////////////////////
//  Compartment.h
//  Implementation of the Class Compartment
//  Created on:      27-Dec-2013 9:35:18 PM
//  Original author: saeed
///////////////////////////////////////////////////////////

#if !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
#define EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_

#include "../../headers.h"
#include "CaChannel.h"
#include "CustomChannel.h"
#include "HHChannel.h"
#include "SynapticChannel.h"

namespace pn2s
{
namespace models
{

class Compartment
{
//	vector<HHChannel<T> > * _compt;
//	int _compt_base;
//	int _compt_size;

public:
	uint gid;

	typedef std::vector<Compartment> Vec;
	typedef typename Vec::iterator itr;

	vector< unsigned int > children;	///< Hines indices of child compts

//	vector<HHChannel<T> > hhchannels;

	Compartment(unsigned int _gid);
	virtual ~Compartment();

	//Copy constractor is necessary because at Vector assign, information will copy through it.
	Compartment( const Compartment& other );
private:

};

}
}
#endif // !defined(EA_F11DAA80_6F7A_4ad9_B555_12F0C681E799__INCLUDED_)
