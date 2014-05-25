///////////////////////////////////////////////////////////
//  Matrix.cpp
//
//  Created on:      26-Dec-2013 4:20:54 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#include "Matrix.h"
#include "../../headers.h"

using namespace pn2s::models;

template <typename T>
Matrix<T>::Matrix():
	_n(0), _m(0)
{
}

template <typename T>
Matrix<T>::Matrix(int n, int m):
	_n(n), _m(m)
{
	_data.resize(n);
	for (int var = 0; var < m; ++var) {
		_data[var].resize(m);
	}
}

template <typename T>
Matrix<T>::~Matrix(){
}

template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;
template class Matrix<long>;
template class Matrix<uint>;
