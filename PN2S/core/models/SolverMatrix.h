///////////////////////////////////////////////////////////
//  SolverMatrix.h
//  Implementation of the Class SolverMatrix
//  Created on:      15-Agu-2014 7:57:50 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#pragma once
#include "../../headers.h"

namespace pn2s
{
namespace models
{

template <typename T, int arch>
class SolverMatrix
{
public:
	static int fast_solve  (const T *A_d, T *b_d, T *x_d, int n, int batch, cudaStream_t stream);
};

}
}

