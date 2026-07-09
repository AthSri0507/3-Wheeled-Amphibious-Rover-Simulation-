#include "hal_interrupt.h"

void hal_interrupt_attach(
    uint8_t pin,
    hal_interrupt_callback_t callback,
    int mode
)
{
    attachInterrupt(
        digitalPinToInterrupt(pin),
        callback,
        mode
    );
}
