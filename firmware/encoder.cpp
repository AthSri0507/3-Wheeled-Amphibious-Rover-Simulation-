#include "encoder.h"

#include "hal_gpio.h"
#include "hal_interrupt.h"


static encoder_t* encoder0 = nullptr;
static encoder_t* encoder1 = nullptr;

/*
    Quadrature decoder
*/

static void encoder_update(
    encoder_t* encoder
)
{
    int msb =
        hal_gpio_read(encoder->pin_a);

    int lsb =
        hal_gpio_read(encoder->pin_b);

    int encoded =
        (msb << 1) | lsb;

    int sum =
        (encoder->last_encoded << 2)
        | encoded;

    /*
        Direction detection
    */

    if(
        sum == 0b1101 ||
        sum == 0b0100 ||
        sum == 0b0010 ||
        sum == 0b1011
    )
    {
        encoder->ticks--;
    }

    if(
        sum == 0b1110 ||
        sum == 0b0111 ||
        sum == 0b0001 ||
        sum == 0b1000
    )
    {
        encoder->ticks++;
    }

    encoder->last_encoded =
        encoded;
}

/*
    ISR wrappers
*/

static void encoder0_isr()
{
    if(encoder0 != nullptr)
    {
        encoder_update(encoder0);
    }
}

static void encoder1_isr()
{
    if(encoder1 != nullptr)
    {
        encoder_update(encoder1);
    }
}

/*
    Initialize encoder
*/

void encoder_init(
    encoder_t* encoder
)
{
    hal_gpio_init(
        encoder->pin_a,
        HAL_GPIO_INPUT_PULLUP
    );

    hal_gpio_init(
        encoder->pin_b,
        HAL_GPIO_INPUT_PULLUP
    );

    encoder->ticks = 0;
    encoder->last_encoded = 0;

    /*
        Register encoder instances
    */

    if(encoder0 == nullptr)
    {
        encoder0 = encoder;

        hal_interrupt_attach(
            encoder->pin_a,
            encoder0_isr,
            CHANGE
        );

        hal_interrupt_attach(
            encoder->pin_b,
            encoder0_isr,
            CHANGE
        );
    }
    else if(encoder1 == nullptr)
    {
        encoder1 = encoder;

        hal_interrupt_attach(
            encoder->pin_a,
            encoder1_isr,
            CHANGE
        );

        hal_interrupt_attach(
            encoder->pin_b,
            encoder1_isr,
            CHANGE
        );
    }
}

/*
    Get tick count
*/

long encoder_get_ticks(
    encoder_t* encoder
)
{
    noInterrupts();

    long ticks =
        encoder->ticks;

    interrupts();

    return ticks;
}

/*
    Reset encoder
*/

void encoder_reset(
    encoder_t* encoder
)
{
    noInterrupts();

    encoder->ticks = 0;

    interrupts();
}
