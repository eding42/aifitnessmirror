#include <stdio.h>
#include "NuMicro.h"
#include <stdbool.h>

#define RX_BUF_SIZE 128
volatile uint8_t g_u8RecData[RX_BUF_SIZE];
volatile uint32_t g_u32DataIdx = 0;
volatile bool g_bMsgReceived = false;

void SYS_Init(void)
{
    /* Enable HIRC (Internal 12MHz) */
    CLK_EnableXtalRC(CLK_SRCCTL_HIRCEN_Msk);
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HIRC, FREQ_220MHZ);
    SystemCoreClockUpdate();

    /* UART1 (Bluetooth) Clock */
    CLK_SetModuleClock(UART1_MODULE, CLK_UARTSEL0_UART1SEL_HIRC, CLK_UARTDIV0_UART1DIV(1));
    CLK_EnableModuleClock(UART1_MODULE);
    
    /* UART1 Pins */
    SET_UART1_RXD_PA2();
    SET_UART1_TXD_PA3();
    
		/* Enale printing to terminal (115200) */
    SetDebugUartCLK();
    SetDebugUartMFP();
}

void UART1_Init(void)
{
    SYS_ResetModule(SYS_UART1RST);
    UART_Open(UART1, 9600);

    /* Enable Receive Data Available Interrupt */
    UART_EnableInt(UART1, UART_INTEN_RDAIEN_Msk);
    NVIC_EnableIRQ(UART1_IRQn);
}

void UART1_IRQHandler(void)
{
    uint32_t u32IntSts = UART1->INTSTS;

    if (u32IntSts & UART_INTSTS_RDAINT_Msk)
    {
        /* Read data until FIFO is empty */
        while (!UART_GET_RX_EMPTY(UART1))
        {
            uint8_t u8Char = UART_READ(UART1);

            // Store char if there is room (leave 1 byte for null terminator)
            if (g_u32DataIdx < (RX_BUF_SIZE - 1))
            {
                g_u8RecData[g_u32DataIdx++] = u8Char;

                // Check for end of line (Newline or Carriage Return)
                if (u8Char == '\n' || u8Char == '\r')
                {
                    g_u8RecData[g_u32DataIdx] = '\0'; // Null terminate string
                    g_bMsgReceived = true;
                }
            }
            else
            {
                // Buffer overflow protection: reset index
                g_u32DataIdx = 0;
            }
        }
    }

    /* Handle Errors */
    if (u32IntSts & (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk))
    {
        UART_ClearIntFlag(UART1, (UART_INTSTS_RLSINT_Msk | UART_INTSTS_BUFERRINT_Msk));
    }
}

int main(void)
{
    SYS_UnlockReg();
    SYS_Init();
    InitDebugUart(); // UART0 115200
    UART1_Init();    // UART1 9600
    SYS_LockReg();

    printf("\nBridge Ready: Bluetooth(9600) -> Terminal(115200)\n");
    
    while (1) 
    {
        if (g_bMsgReceived)
        {
            // Print the entire captured string at once
            printf("Message: %s", g_u8RecData);

            // Reset for next message
            g_u32DataIdx = 0;
            g_bMsgReceived = false;
        }
    }
}
