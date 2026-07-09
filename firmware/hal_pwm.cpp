#include "hal_pwm.h"

#ifdef ESP32

/*
    ESP32 implementation
*/

void hal_pwm_init(uint8_t pin)
{
    pinMode(pin, OUTPUT);
}

void hal_pwm_write(uint8_t pin, uint16_t value)
{
    // value 0–255
    analogWrite(pin, value);
}

#elif defined(ARDUINO_ARCH_RP2350)

/*
     Pico implementation
*/

void hal_pwm_init(uint8_t pin)
{
    pinMode(pin, OUTPUT);
}

void hal_pwm_write(uint8_t pin, uint16_t value)
{
    analogWrite(pin, value);
}

#else

#error "Unsupported platform"

#endif
