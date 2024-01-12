/*
 * Generated with the FPGA Interface C API Generator 19.0
 * for NI-RIO 19.0 or later.
 */
#ifndef __NiFpga_FPGA_CANBus_IMU_4module_IRQ_h__
#define __NiFpga_FPGA_CANBus_IMU_4module_IRQ_h__

#ifndef NiFpga_Version
#define NiFpga_Version 190
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_FPGA_CANBus_IMU_4module_IRQ_Bitfile;
 */
#define NiFpga_FPGA_CANBus_IMU_4module_IRQ_Bitfile "/home/admin/fpga_driver/fpga_bitfile/NiFpga_FPGA_CANBus_IMU_4module_IRQ_0530.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char *const NiFpga_FPGA_CANBus_IMU_4module_IRQ_Signature = "C6668CE3477B80E76E583EFB8BCEB744";

#if NiFpga_Cpp
extern "C"
{
#endif

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_ChecksumOK = 0x1803A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Conn9_4r = 0x180AE,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Conn9_5r = 0x180AA,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0Complete = 0x18012,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0ID1RXTimedout = 0x180C2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0ID1TXTimedout = 0x180D6,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0ID2RXTimedout = 0x180DE,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0ID2TXTimedout = 0x180EE,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0TXFault = 0x180F6,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN0success = 0x1800A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1Complete = 0x1804E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1ID1RXTimedout = 0x1812A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1ID1TXTimedout = 0x18132,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1ID2RXTimedout = 0x18146,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1ID2TXTimedout = 0x18156,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1TXFault = 0x1816A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod1CAN1success = 0x18056,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0Complete = 0x18062,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0ID1RXTimedout = 0x1818E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0ID1TXTimedout = 0x18196,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0ID2RXTimedout = 0x181AA,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0ID2TXTimedout = 0x181BA,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0TXFault = 0x181C6,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN0success = 0x1806A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1Complete = 0x18076,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1ID1RXTimedout = 0x181F2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1ID1TXTimedout = 0x181FA,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1ID2RXTimedout = 0x1820E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1ID2TXTimedout = 0x1821E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1TXFault = 0x1822A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_Mod2CAN1success = 0x1807E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool_RXfinish = 0x1802A,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorBool;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN0CommState = 0x1810A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN0ECCRegister = 0x18102,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN0ReceiveErrorCounter = 0x18106,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN1CommState = 0x1816E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN1ECCRegister = 0x18176,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod1CAN1ReceiveErrorCounter = 0x18172,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN0CommState = 0x181CA,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN0ECCRegister = 0x181D2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN0ReceiveErrorCounter = 0x181CE,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN1CommState = 0x1822E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN1ECCRegister = 0x18236,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8_Mod2CAN1ReceiveErrorCounter = 0x18232,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU8;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI16_Mod1CAN0CompleteCounter = 0x1800E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI16_Mod1CAN1CompleteCounter = 0x18052,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI16_Mod2CAN0CompleteCounter = 0x18066,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI16_Mod2CAN1CompleteCounter = 0x1807A,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI16;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI32_IRQ0_cnt = 0x1808C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI32_IRQ1_cnt = 0x18098,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorI32;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN0BusError = 0x180F8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN0ID1TX2RXTime = 0x180CC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN0ID2TX2RXTime = 0x180E8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN0RXCounter = 0x1810C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN0TXError = 0x180FC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN1BusError = 0x1817C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN1ID1TX2RXTime = 0x1812C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN1ID2TX2RXTime = 0x18148,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN1RXCounter = 0x18164,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod1CAN1TXError = 0x18178,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN0BusError = 0x181D8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN0ID1TX2RXTime = 0x18190,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN0ID2TX2RXTime = 0x181AC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN0RXCounter = 0x181C0,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN0TXError = 0x181D4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN1BusError = 0x1823C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN1ID1TX2RXTime = 0x181F4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN1ID2TX2RXTime = 0x18210,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN1RXCounter = 0x18224,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32_Mod2CAN1TXError = 0x18238,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorU32;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Conn9_1w = 0x180B6,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Conn9_2w = 0x180B2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Conn9_3w = 0x180A6,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Digital = 0x18022,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_IRQ0_wait_until_cleared = 0x18096,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_IRQ1_wait_until_cleared = 0x180A2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_MOD1CAN0Transmit = 0x18002,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_MOD1CAN1Transmit = 0x1805A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_MOD2CAN0Transmit = 0x1806E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_MOD2CAN1Transmit = 0x18082,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Power = 0x1801A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_Signal = 0x1801E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_VICONtrigger = 0x18016,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool_stop = 0x18006,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlBool;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_IOrwuSec = 0x180B8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_IRQ0_period_us = 0x18090,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_IRQ1_period_us = 0x1809C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN0 = 0x18040,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN0ID1 = 0x180C8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN0ID2 = 0x180E4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN0RXTimoutus = 0x18118,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN1 = 0x18044,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN1ID1 = 0x18134,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN1ID2 = 0x1814C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod1CAN1RXTimoutus = 0x18158,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN0 = 0x18084,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN0ID1 = 0x18198,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN0ID2 = 0x181B0,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN0RXTimoutus = 0x181E0,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN1 = 0x18088,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN1ID1 = 0x181FC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN1ID2 = 0x18214,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32_Mod2CAN1RXTimoutus = 0x18244,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlU32;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_DataArray = 0x18034,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_DataRX = 0x1802C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod1CAN0ID1RXData = 0x180BC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod1CAN0ID2RXData = 0x180D8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod1CAN1ID1RXData = 0x18124,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod1CAN1ID2RXData = 0x18140,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod2CAN0ID1RXData = 0x18188,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod2CAN0ID2RXData = 0x181A4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod2CAN1ID1RXData = 0x181EC,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8_Mod2CAN1ID2RXData = 0x18208,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_DataArray = 49,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_DataRX = 64,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod1CAN0ID1RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod1CAN0ID2RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod1CAN1ID1RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod1CAN1ID2RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod2CAN0ID1RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod2CAN0ID2RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod2CAN1ID1RXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size_Mod2CAN1ID2RXData = 8,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU8Size;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU16_Data = 0x18030,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU16;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU16Size_Data = 24,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorArrayU16Size;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBool_Mod1CAN0Select = 0x1803E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBool_Mod1CAN1Select = 0x1804A,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBool_Mod2CAN0Select = 0x1805E,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBool_Mod2CAN1Select = 0x18072,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBool;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBoolSize_Mod1CAN0Select = 2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBoolSize_Mod1CAN1Select = 2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBoolSize_Mod2CAN0Select = 2,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBoolSize_Mod2CAN1Select = 2,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayBoolSize;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_DataTx = 0x18024,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod1CAN0ID1TXData = 0x180C4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod1CAN0ID2TXData = 0x180E0,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod1CAN1ID1TXData = 0x18138,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod1CAN1ID2TXData = 0x18150,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod2CAN0ID1TXData = 0x1819C,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod2CAN0ID2TXData = 0x181B4,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod2CAN1ID1TXData = 0x18200,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8_Mod2CAN1ID2TXData = 0x18218,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8;

    typedef enum
    {
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_DataTx = 12,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod1CAN0ID1TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod1CAN0ID2TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod1CAN1ID1TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod1CAN1ID2TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod2CAN0ID1TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod2CAN0ID2TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod2CAN1ID1TXData = 8,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size_Mod2CAN1ID2TXData = 8,
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_ControlArrayU8Size;

#if !NiFpga_VxWorks

    /* Indicator: Mod1CAN0ID1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN0ID1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Resource = 0x180D0;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type *const source);

    /* Indicator: Mod1CAN0ID2RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN0ID2RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Resource = 0x180F0;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type *const source);

    /* Indicator: Mod1CAN0RXError */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN0RXError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Resource = 0x18114;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type *const source);

    /* Indicator: Mod1CAN0RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN0RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Resource = 0x18110;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type *const source);

    /* Indicator: Mod1CAN0TransmitError */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN0TransmitError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Resource = 0x1811C;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type *const source);

    /* Indicator: Mod1CAN1ID1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN1ID1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Resource = 0x18120;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type *const source);

    /* Indicator: Mod1CAN1ID2RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN1ID2RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Resource = 0x1813C;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type *const source);

    /* Indicator: Mod1CAN1RXError */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN1RXError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Resource = 0x18180;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type *const source);

    /* Indicator: Mod1CAN1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Resource = 0x18160;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type *const source);

    /* Indicator: Mod1CAN1TransmitError */
    /* Use NiFpga_ReadArrayU8() to access Mod1CAN1TransmitError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Resource = 0x1815C;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type *const source);

    /* Indicator: Mod2CAN0ID1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN0ID1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Resource = 0x18184;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type *const source);

    /* Indicator: Mod2CAN0ID2RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN0ID2RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Resource = 0x181A0;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type *const source);

    /* Indicator: Mod2CAN0RXError */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN0RXError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Resource = 0x181DC;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type *const source);

    /* Indicator: Mod2CAN0RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN0RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Resource = 0x181BC;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type *const source);

    /* Indicator: Mod2CAN0TransmitError */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN0TransmitError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Resource = 0x181E4;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type *const source);

    /* Indicator: Mod2CAN1ID1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN1ID1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Resource = 0x181E8;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type *const source);

    /* Indicator: Mod2CAN1ID2RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN1ID2RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Resource = 0x18204;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type *const source);

    /* Indicator: Mod2CAN1RXError */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN1RXError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Resource = 0x18240;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type *const source);

    /* Indicator: Mod2CAN1RXFrame */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN1RXFrame */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Resource = 0x18220;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_PackedSizeInBytes = 24;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type
    {
        uint32_t timestamphigh;
        uint32_t timestamplow;
        uint32_t identifier;
        uint8_t type;
        uint8_t infoA;
        uint8_t infoB;
        uint8_t datalength;
        uint8_t data[8];
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type *const source);

    /* Indicator: Mod2CAN1TransmitError */
    /* Use NiFpga_ReadArrayU8() to access Mod2CAN1TransmitError */
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Resource = 0x18248;
    const uint32_t NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_PackedSizeInBytes = 5;

    typedef struct NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type
    {
        NiFpga_Bool status;
        int32_t code;
    } NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type;

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_UnpackCluster(
        const uint8_t *const packedData,
        NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type *const destination);

    void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_PackCluster(
        uint8_t *const packedData,
        const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type *const source);

#endif /* !NiFpga_VxWorks */

#if NiFpga_Cpp
}
#endif

#endif
