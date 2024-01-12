#include "NiFpga_FPGA_CANBus_IMU_4module_IRQ.h"

#if !NiFpga_VxWorks

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0ID2RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN0TransmitError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1ID2RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod1CAN1TransmitError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0ID2RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN0TransmitError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1ID2RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type *const destination)
{
   (*destination).timestamphigh = 0;
   (*destination).timestamphigh |= (packedData[0] & 0xFFULL) << 24;
   (*destination).timestamphigh |= (packedData[1] & 0xFF) << 16;
   (*destination).timestamphigh |= (packedData[2] & 0xFF) << 8;
   (*destination).timestamphigh |= (packedData[3] & 0xFF);
   (*destination).timestamplow = 0;
   (*destination).timestamplow |= (packedData[4] & 0xFFULL) << 24;
   (*destination).timestamplow |= (packedData[5] & 0xFF) << 16;
   (*destination).timestamplow |= (packedData[6] & 0xFF) << 8;
   (*destination).timestamplow |= (packedData[7] & 0xFF);
   (*destination).identifier = 0;
   (*destination).identifier |= (packedData[8] & 0xFFULL) << 24;
   (*destination).identifier |= (packedData[9] & 0xFF) << 16;
   (*destination).identifier |= (packedData[10] & 0xFF) << 8;
   (*destination).identifier |= (packedData[11] & 0xFF);
   (*destination).type = 0;
   (*destination).type |= (packedData[12] & 0xFF);
   (*destination).infoA = 0;
   (*destination).infoA |= (packedData[13] & 0xFF);
   (*destination).infoB = 0;
   (*destination).infoB |= (packedData[14] & 0xFF);
   (*destination).datalength = 0;
   (*destination).datalength |= (packedData[15] & 0xFF);
   (*destination).data[0] = 0;
   (*destination).data[0] |= (packedData[16] & 0xFF);
   (*destination).data[1] = 0;
   (*destination).data[1] |= (packedData[17] & 0xFF);
   (*destination).data[2] = 0;
   (*destination).data[2] |= (packedData[18] & 0xFF);
   (*destination).data[3] = 0;
   (*destination).data[3] |= (packedData[19] & 0xFF);
   (*destination).data[4] = 0;
   (*destination).data[4] |= (packedData[20] & 0xFF);
   (*destination).data[5] = 0;
   (*destination).data[5] |= (packedData[21] & 0xFF);
   (*destination).data[6] = 0;
   (*destination).data[6] |= (packedData[22] & 0xFF);
   (*destination).data[7] = 0;
   (*destination).data[7] |= (packedData[23] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1RXFrame_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).timestamphigh >> 24) & 0xFF);
   packedData[1] = (uint8_t)(((*source).timestamphigh >> 16) & 0xFF);
   packedData[2] = (uint8_t)(((*source).timestamphigh >> 8) & 0xFF);
   packedData[3] = (uint8_t)((*source).timestamphigh & 0xFF);
   packedData[4] = (uint8_t)(((*source).timestamplow >> 24) & 0xFF);
   packedData[5] = (uint8_t)(((*source).timestamplow >> 16) & 0xFF);
   packedData[6] = (uint8_t)(((*source).timestamplow >> 8) & 0xFF);
   packedData[7] = (uint8_t)((*source).timestamplow & 0xFF);
   packedData[8] = (uint8_t)(((*source).identifier >> 24) & 0xFF);
   packedData[9] = (uint8_t)(((*source).identifier >> 16) & 0xFF);
   packedData[10] = (uint8_t)(((*source).identifier >> 8) & 0xFF);
   packedData[11] = (uint8_t)((*source).identifier & 0xFF);
   packedData[12] = (uint8_t)((*source).type & 0xFF);
   packedData[13] = (uint8_t)((*source).infoA & 0xFF);
   packedData[14] = (uint8_t)((*source).infoB & 0xFF);
   packedData[15] = (uint8_t)((*source).datalength & 0xFF);
   packedData[16] = (uint8_t)((*source).data[0] & 0xFF);
   packedData[17] = (uint8_t)((*source).data[1] & 0xFF);
   packedData[18] = (uint8_t)((*source).data[2] & 0xFF);
   packedData[19] = (uint8_t)((*source).data[3] & 0xFF);
   packedData[20] = (uint8_t)((*source).data[4] & 0xFF);
   packedData[21] = (uint8_t)((*source).data[5] & 0xFF);
   packedData[22] = (uint8_t)((*source).data[6] & 0xFF);
   packedData[23] = (uint8_t)((*source).data[7] & 0xFF);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_UnpackCluster(
    const uint8_t *const packedData,
    NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type *const destination)
{
   (*destination).status = 0;
   (*destination).status |= ((packedData[0] >> 7) & 0x1);
   (*destination).code = 0;
   (*destination).code |= (packedData[0] & 0x7FULL) << 25;
   (*destination).code |= (packedData[1] & 0xFF) << 17;
   (*destination).code |= (packedData[2] & 0xFF) << 9;
   (*destination).code |= (packedData[3] & 0xFF) << 1;
   (*destination).code |= ((packedData[4] >> 7) & 0x1);
}

void NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_PackCluster(
    uint8_t *const packedData,
    const NiFpga_FPGA_CANBus_IMU_4module_IRQ_IndicatorCluster_Mod2CAN1TransmitError_Type *const source)
{
   packedData[0] = (uint8_t)(((*source).status & 0x1) << 7);
   packedData[0] |= (uint8_t)(((*source).code >> 25) & 0x7F);
   packedData[1] = (uint8_t)(((*source).code >> 17) & 0xFF);
   packedData[2] = (uint8_t)(((*source).code >> 9) & 0xFF);
   packedData[3] = (uint8_t)(((*source).code >> 1) & 0xFF);
   packedData[4] = (uint8_t)(((*source).code & 0x1) << 7);
}

#endif /* !NiFpga_VxWorks */
