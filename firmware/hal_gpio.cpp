#include "hal_gpio.h"

/*
    Initialize GPIO pin
*/

void hal_gpio_init(
    uint8_t pin,
    hal_gpio_mode_t mode
)
{
    switch(mode)
    {
        case HAL_GPIO_INPUT:
            pinMode(pin, INPUT);
            break;

        case HAL_GPIO_OUTPUT:
            pinMode(pin, OUTPUT);
            break;

        case HAL_GPIO_INPUT_PULLUP:
            pinMode(pin, INPUT_PULLUP);
            break;

        default:
            break;
    }
}

/*
    Write digital state
*/

void hal_gpio_write(
    uint8_t pin,
    hal_gpio_state_t state
)
{
    digitalWrite(pin, state);
}

/*
    Read digital state
*/

hal_gpio_state_t hal_gpio_read(
    uint8_t pin
)
{
    return (hal_gpio_state_t)digitalRead(pin);
}

/*
    Toggle pin state
*/

void hal_gpio_toggle(
    uint8_t pin
)
{
    hal_gpio_state_t current;

    current = hal_gpio_read(pin);

    if(current == HAL_GPIO_HIGH)
    {
        hal_gpio_write(pin, HAL_GPIO_LOW);
    }
    else
    {
        hal_gpio_write(pin, HAL_GPIO_HIGH);
    }
}
