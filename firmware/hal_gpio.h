#pragma once

#include <Arduino.h>

/*
    GPIO modes
*/

typedef enum
{
    HAL_GPIO_INPUT,
    HAL_GPIO_OUTPUT,
    HAL_GPIO_INPUT_PULLUP
} hal_gpio_mode_t;

/*
    GPIO state values
*/

typedef enum
{
    HAL_GPIO_LOW = 0,
    HAL_GPIO_HIGH = 1
} hal_gpio_state_t;

/*
    Initialize GPIO pin
*/

void hal_gpio_init(
    uint8_t pin,
    hal_gpio_mode_t mode
);

/*
    Write value to pin
*/

void hal_gpio_write(
    uint8_t pin,
    hal_gpio_state_t state
);

/*
    Read value from pin
*/

hal_gpio_state_t hal_gpio_read(
    uint8_t pin
);

/*
    Toggle GPIO state
*/

void hal_gpio_toggle(
    uint8_t pin
);
