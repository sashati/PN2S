///////////////////////////////////////////////////////////
//  PN2S_Utils.h
//
//  Created on:      26-Dec-2013 4:19:23 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(PN2S_TaskQueue__INCLUDED_)
#define PN2S_TaskQueue__INCLUDED_
#include "PN2S_TaskInfo.h"

class PN2S_TaskQueue
{
public:
	PN2S_TaskQueue();
	virtual ~PN2S_TaskQueue();
	hscError Add(PN2S_TaskInfo &task);
	PN2S_TaskInfo* Get();
//	void Process();
//	hscError AddOutputTask();
//	PN2S_TaskInfo* GetOutputTask();
//
	int _size;
	int _limit;
private:
	std::deque<PN2S_TaskInfo> _list;
//	hscError getValuesFromShell();

};
#endif // !defined(PN2S_TaskQueue__INCLUDED_)
