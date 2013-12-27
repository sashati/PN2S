///////////////////////////////////////////////////////////
//  HSC_DeviceManager.h
//  Implementation of the Class HSC_DeviceManager
//  Created on:      26-Dec-2013 4:18:15 PM
//  Original author: Saeed Shariati
///////////////////////////////////////////////////////////

#if !defined(EA_8204B80E_EF46_47df_8AB8_FC787EF1223C__INCLUDED_)
#define EA_8204B80E_EF46_47df_8AB8_FC787EF1223C__INCLUDED_

#include "HSC_Device.h"
#include "HSC_Scheduler.h"

/**
 * Create one thread per device and let it process tasks and when finished, add
 * and output task to Scheduler
 */
class HSC_DeviceManager
{

public:
	HSC_DeviceManager();
	virtual ~HSC_DeviceManager();
	HSC_Device *m_HSC_Device;
	HSC_Scheduler *m_HSC_Scheduler;

};
#endif // !defined(EA_8204B80E_EF46_47df_8AB8_FC787EF1223C__INCLUDED_)
