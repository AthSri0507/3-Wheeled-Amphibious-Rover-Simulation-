#pragma once

#include <Arduino.h>
#include "encoder.h"

//Odometry data
typedef struct
{
    //Raw RPM
    float left_motor_rpm;
    float right_motor_rpm;

    //Actual wheel RPM
    float left_wheel_rpm;
    float right_wheel_rpm;

    //Linear wheel velocities
    float left_velocity;
    float right_velocity;

    //Robot velocities
    float linear_velocity;
    float angular_velocity;

    //Robot pose
    float x;
    float y;
    float heading;

    float distance;
} odometry_data_t;

//Odometry object


typedef struct
{
    encoder_t* left_encoder;
    encoder_t* right_encoder;
    //Encoder configuration
    float ticks_per_motor_rev;
    
    //Gear ratio
    float gear_ratio;

    //Wheel radius (meters)
    float wheel_radius;

    //Distance between wheels
    float wheel_base;

    //Previous encoder ticks
    long prev_left_ticks;
    long prev_right_ticks;

    //Previous update time
    uint32_t prev_time_ms;

    //Output data
    odometry_data_t data;

} odometry_t;

/*
    Initialize odometry
*/

void odometry_init(odometry_t* odom);

/*
    Update odometry
*/

void odometry_update(odometry_t* odom);
