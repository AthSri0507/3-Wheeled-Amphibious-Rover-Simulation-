#pragma once

#include <Arduino.h>

/*
    Encoder object
*/

typedef struct
{
    uint8_t pin_a;
    uint8_t pin_b;

    volatile long ticks;

    volatile int last_encoded;

} encoder_t;

/*
    Initialize encoder
*/

void encoder_init(
    encoder_t* encoder
);

/*
    Get encoder ticks
*/

long encoder_get_ticks(
    encoder_t* encoder
);

/*
    Reset encoder ticks
*/

void encoder_reset(
    encoder_t* encoder
);
