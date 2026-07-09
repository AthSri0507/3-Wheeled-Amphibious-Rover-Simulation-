#include "l289.h"

#include "hal_gpio.h"
#include "hal_pwm.h"

/*
    Initialize 
*/

void l298_init(
    l298_motor_t* motor
)
{
    hal_gpio_init(
        motor->in1,
        HAL_GPIO_OUTPUT
    );

    hal_gpio_init(
        motor->in2,
        HAL_GPIO_OUTPUT
    );

    hal_pwm_init(
        motor->en
    );

    l298_stop(motor);
}

/*
    Drive 
*/

void l298_drive(
    l298_motor_t* motor,
    l298_direction_t direction,
    uint8_t speed
)
{
    switch(direction)
    {
        case L298_FORWARD:

            hal_gpio_write(
                motor->in1,
                HAL_GPIO_HIGH
            );

            hal_gpio_write(
                motor->in2,
                HAL_GPIO_LOW
            );

            break;

        case L298_REVERSE:

            hal_gpio_write(
                motor->in1,
                HAL_GPIO_LOW
            );

            hal_gpio_write(
                motor->in2,
                HAL_GPIO_HIGH
            );

            break;

        case L298_BRAKE:

            hal_gpio_write(
                motor->in1,
                HAL_GPIO_HIGH
            );

            hal_gpio_write(
                motor->in2,
                HAL_GPIO_HIGH
            );

            break;

        case L298_COAST:

            hal_gpio_write(
                motor->in1,
                HAL_GPIO_LOW
            );

            hal_gpio_write(
                motor->in2,
                HAL_GPIO_LOW
            );

            break;

        default:
            break;
    }

    hal_pwm_write(
        motor->en,
        speed
    );
}

/*
    Stop motor
*/

void l298_stop(
    l298_motor_t* motor
)
{
    hal_gpio_write(
        motor->in1,
        HAL_GPIO_LOW
    );

    hal_gpio_write(
        motor->in2,
        HAL_GPIO_LOW
    );

    hal_pwm_write(
        motor->en,
        0
    );
}
