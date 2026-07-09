#include "odometry.h"
#include <math.h>

/*
    PI constant
*/

#define PI_F 3.14159265359f

/*
    Initialize odometry
*/

void odometry_init(odometry_t* odom)
{
    /*
        Null pointer protection
    */

    if(odom == nullptr)
    {
        return;
    }

    /*
        Initialize previous encoder ticks safely
    */

    if(odom->left_encoder != nullptr)
    {
        odom->prev_left_ticks =
            encoder_get_ticks(
                odom->left_encoder
            );
    }
    else
    {
        odom->prev_left_ticks = 0;
    }

    if(odom->right_encoder != nullptr)
    {
        odom->prev_right_ticks =
            encoder_get_ticks(
                odom->right_encoder
            );
    }
    else
    {
        odom->prev_right_ticks = 0;
    }

    /*
        Initialize timing
    */

    odom->prev_time_ms = millis();

    /*
        Reset pose
    */

    odom->data.x = 0.0f;
    odom->data.y = 0.0f;
    odom->data.heading = 0.0f;
    odom->data.distance = 0.0f;

    /*
        Reset velocities
    */

    odom->data.left_motor_rpm = 0.0f;
    odom->data.right_motor_rpm = 0.0f;

    odom->data.left_wheel_rpm = 0.0f;
    odom->data.right_wheel_rpm = 0.0f;

    odom->data.left_velocity = 0.0f;
    odom->data.right_velocity = 0.0f;

    odom->data.linear_velocity = 0.0f;
    odom->data.angular_velocity = 0.0f;
}

/*
    Update odometry
*/

void odometry_update(odometry_t* odom)
{
    /*
        Null protection
    */

    if(odom == nullptr)
    {
        return;
    }

    /*
        Validate configuration
    */

    if(odom->ticks_per_motor_rev <= 0.0f)
    {
        return;
    }

    if(odom->gear_ratio <= 0.0f)
    {
        return;
    }

    if(odom->wheel_radius <= 0.0f)
    {
        return;
    }

    /*
        Current time
    */

    uint32_t current_time = millis();

    float dt =
        (current_time - odom->prev_time_ms)
        / 1000.0f;

    /*
        Prevent divide by zero
    */

    if(dt <= 0.0f)
    {
        return;
    }

    /*
        Read encoder ticks safely
    */

    long left_ticks = 0;
    long right_ticks = 0;

    if(odom->left_encoder != nullptr)
    {
        left_ticks =
            encoder_get_ticks(
                odom->left_encoder
            );
    }

    if(odom->right_encoder != nullptr)
    {
        right_ticks =
            encoder_get_ticks(
                odom->right_encoder
            );
    }

    /*
        Tick deltas
    */

    long delta_left =
        left_ticks -
        odom->prev_left_ticks;

    long delta_right =
        right_ticks -
        odom->prev_right_ticks;

    /*
        Save current state
    */

    odom->prev_left_ticks =
        left_ticks;

    odom->prev_right_ticks =
        right_ticks;

    odom->prev_time_ms =
        current_time;

    /*
        Motor RPM
    */

    odom->data.left_motor_rpm =
        ((float)delta_left /
        odom->ticks_per_motor_rev)
        * (60.0f / dt);

    odom->data.right_motor_rpm =
        ((float)delta_right /
        odom->ticks_per_motor_rev)
        * (60.0f / dt);

    /*
        Wheel RPM
    */

    odom->data.left_wheel_rpm =
        odom->data.left_motor_rpm /
        odom->gear_ratio;

    odom->data.right_wheel_rpm =
        odom->data.right_motor_rpm /
        odom->gear_ratio;

    /*
        Wheel linear velocity
    */

    float wheel_circumference =
        2.0f * PI_F *
        odom->wheel_radius;

    odom->data.left_velocity =
        (odom->data.left_wheel_rpm *
        wheel_circumference)
        / 60.0f;

    odom->data.right_velocity =
        (odom->data.right_wheel_rpm *
        wheel_circumference)
        / 60.0f;

    /*
        Single encoder mode
    */

    bool left_exists =
        (odom->left_encoder != nullptr);

    bool right_exists =
        (odom->right_encoder != nullptr);

    /*
        Differential drive mode
    */

    if(
        left_exists &&
        right_exists &&
        odom->wheel_base > 0.0f
    )
    {
        odom->data.linear_velocity =
            (
                odom->data.left_velocity +
                odom->data.right_velocity
            ) * 0.5f;

        odom->data.angular_velocity =
            (
                odom->data.right_velocity -
                odom->data.left_velocity
            ) / odom->wheel_base;

        /*
            Pose update
        */

        odom->data.heading +=
            odom->data.angular_velocity * dt;

        odom->data.x +=
            odom->data.linear_velocity *
            cos(odom->data.heading) *
            dt;

        odom->data.y +=
            odom->data.linear_velocity *
            sin(odom->data.heading) *
            dt;

        /*
            Distance update
        */

        odom->data.distance +=
            fabs(
                odom->data.linear_velocity
            ) * dt;
    }

    /*
        Single wheel mode
    */

    else
    {
        /*
            Use whichever wheel exists
        */

        if(right_exists)
        {
            odom->data.linear_velocity =
                odom->data.right_velocity;

            odom->data.distance +=
                fabs(
                    odom->data.right_velocity
                ) * dt;
        }
        else if(left_exists)
        {
            odom->data.linear_velocity =
                odom->data.left_velocity;

            odom->data.distance +=
                fabs(
                    odom->data.left_velocity
                ) * dt;
        }

        /*
            No angular estimation
        */

        odom->data.angular_velocity = 0.0f;
    }
}
