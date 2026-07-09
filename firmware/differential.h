#pragma once

#ifndef DIFFERENTIAL_H
#define DIFFERENTIAL_H


#include <Arduino.h>

//Differential mixer output
typedef struct
{
    float left;
    float right;
} diff_output_t;

/*
    Inputs:
        throttle -> forward/reverse motion
        steering -> turning amount
    Range:
        throttle : -1.0 to +1.0
        steering : -1.0 to +1.0
    Outputs:
        left/right normalized wheel commands
*/


void differential_mixer_update(
    float throttle,
    float steering,
    diff_output_t* output
);

// Time (ms) to rotate angle_rad at the given steering fraction (open-loop)
uint32_t differential_turn_duration_ms(float angle_rad, float steering);
#endif
