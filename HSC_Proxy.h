///////////////////////////////////////////////////////////
//  HSC_Proxy.h
//  Implementation of the Class HSC_Proxy
//  Created on:      26-Dec-2013 4:08:07 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(EA_27150678_5056_4754_82F6_A77DCEB1BC1C__INCLUDED_)
#define EA_27150678_5056_4754_82F6_A77DCEB1BC1C__INCLUDED_

/**
 * The class that use HSC package and deal with moose constrains.
 */
class HSC_Proxy
{

public:
	HSC_Proxy();
	virtual ~HSC_Proxy();

	void InsertModel(int id);
	void Reinit();
	void createModel();
	void Process(int id);

};
#endif // !defined(EA_27150678_5056_4754_82F6_A77DCEB1BC1C__INCLUDED_)
