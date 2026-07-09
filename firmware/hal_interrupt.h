#pragma once

#include <Arduino.h>

typedef void (*hal_interrupt_callback_t)(void);

void hal_interrupt_attach(
    uint8_t pin,
    hal_interrupt_callback_t callback,
    int mode
);
