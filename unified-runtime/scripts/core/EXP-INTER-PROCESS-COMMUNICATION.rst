<%
    OneApi=tags['$OneApi']
    x=tags['$x']
    X=x.upper()
%>

.. _experimental-inter-process-communication:

================================================================================
Inter Process Communication
================================================================================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions over
        time.
    *   Do not require conformance testing of their own additions.


Motivation
--------------------------------------------------------------------------------
This extension introduces functionality for allowing processes to share device
USM memory allocations and events. Doing so lets processes actively
communicate with each other through the devices, by explicitly managing handles
that represent shareable objects for inter-process communication.

API
--------------------------------------------------------------------------------

Enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_device_info_t
    * ${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP
    * ${X}_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP

Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ${x}_exp_ipc_mem_handle_t
* ${x}_exp_ipc_event_handle_t

Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Inter-Process Memory
   * ${x}IPCGetMemHandleExp
   * ${x}IPCPutMemHandleExp
   * ${x}IPCOpenMemHandleExp
   * ${x}IPCCloseMemHandleExp
   * ${x}IPCGetMemHandleDataExp
* Inter-Process Event
   * ${x}IPCEnqueueEventsWait
   * ${x}IPCGetEventHandleExp
   * ${x}IPCOpenEventHandleExp
   * ${x}IPCGetEventHandleDataExp

Changelog
--------------------------------------------------------------------------------

+-----------+------------------------+
| Revision  | Changes                |
+===========+========================+
| 1.0       | Initial Draft          |
+-----------+------------------------+

Support
--------------------------------------------------------------------------------

Adapters which support the inter-process memory experimental functions *must*
return true for the new ``${X}_DEVICE_INFO_IPC_MEMORY_SUPPORT_EXP`` device info
query.
Adapters which support the inter-process event experimental functions *must*
return true for the new ``${X}_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP`` device info
query.

Contributors
--------------------------------------------------------------------------------

* Larsen, Steffen `steffen.larsen@intel.com <steffen.larsen@intel.com>`_
