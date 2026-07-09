#pragma once

#include <Arduino.h>

/*
    Motor direction
*/

typedef enum
{
    L298_FORWARD,
    L298_REVERSE,
    L298_BRAKE,
    L298_COAST
} l298_direction_t;

/*
    L298 motor object
*/

typedef struct
{
    uint8_t in1;
    uint8_t in2;
    uint8_t en;

} l298_motor_t;

/*
    Initialize motor driver
*/

void l298_init(
    l298_motor_t* motor
);

/*
    Set motor output
*/

void l298_drive(
    l298_motor_t* motor,
    l298_direction_t direction,
    uint8_t speed
);

/*
    Stop motor
*/

void l298_stop(
    l298_motor_t* motor
);
