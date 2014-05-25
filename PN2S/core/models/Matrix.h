///////////////////////////////////////////////////////////
//  Matrix.h
//
//  Created on:      26-Dec-2013 4:20:54 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(A905F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_)
#define A905F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_

#include <vector>

namespace pn2s
{
namespace models
{

template <typename T>
class Matrix
{
	std::vector< std::vector<T> > _data;
	int _n;
	int _m;
public:
	Matrix();
	Matrix(int n, int m);
	virtual ~Matrix();

	std::vector<T> operator [](int i) const {return _data[i];}
	std::vector<T> & operator [](int i) {return _data[i];}
};

}
}
#endif // !defined(A905F55B9_7DDF_45c6_81E6_3396EFC0EED4__INCLUDED_)
