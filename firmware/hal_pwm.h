#ifndef HAL_PWM_H
#define HAL_PWM_H

#pragma once
#include <Arduino.h>

void hal_pwm_init(uint8_t pin);
void hal_pwm_write(uint8_t pin, uint16_t value);


#endif
