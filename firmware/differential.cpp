#include "differential.h"
#include <math.h>

/*
    =====================================
    ROBOT GEOMETRY / DRIVE CONFIG
    Measure these on YOUR robot.
    =====================================
*/

static const float WHEEL_BASE_M   = 0.12f;     // track width (m)
static const float WHEEL_RADIUS_M = 0.025f;   // wheel radius (m)

// Per-wheel max RPM at full PWM, SEPARATELY for each direction.
// Measure all four — motors are rarely symmetric fwd vs rev.
static const float MAX_RPM_LEFT_FWD  = 140.0f;
static const float MAX_RPM_LEFT_REV  = 140.0f;
static const float MAX_RPM_RIGHT_FWD = 100.0f;
static const float MAX_RPM_RIGHT_REV = 100.0f;

// Turn rate commanded at steering = +/-1.0 (rad/s). Tuning knob.
static const float STEER_MAX_RAD_S = 1.5f;

// --- derived: max linear wheel speed (m/s) per side per direction ---
#define RPM_TO_MPS(rpm) (((rpm) / 60.0f) * 2.0f * (float)M_PI * WHEEL_RADIUS_M)

static const float MAX_L_FWD = RPM_TO_MPS(MAX_RPM_LEFT_FWD);
static const float MAX_L_REV = RPM_TO_MPS(MAX_RPM_LEFT_REV);
static const float MAX_R_FWD = RPM_TO_MPS(MAX_RPM_RIGHT_FWD);
static const float MAX_R_REV = RPM_TO_MPS(MAX_RPM_RIGHT_REV);

static float clamp(float value, float min_value, float max_value)
{
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

// Pick the right max for a wheel given its commanded velocity sign.
static float wheel_max(float v, float max_fwd, float max_rev)
{
    return (v >= 0.0f) ? max_fwd : max_rev;
}

/*
    Core kinematics + direction-aware per-wheel normalization.

      v_ref limited by the slower wheel IN THE TRAVEL DIRECTION,
      so throttle = +/-1.0 is achievable without saturating either side.

      v     = throttle * v_ref
      omega = steering * STEER_MAX_RAD_S
      v_left  = v - omega*(L/2)
      v_right = v + omega*(L/2)
      left  = v_left  / (that wheel's max for its direction)
      right = v_right / (that wheel's max for its direction)
*/
static void mix(float throttle, float steering,
                float* left, float* right, float* scale_out)
{
    // Reference uses the slower side for the intended travel direction.
    float v_ref = (throttle >= 0.0f)
                  ? fminf(MAX_L_FWD, MAX_R_FWD)
                  : fminf(MAX_L_REV, MAX_R_REV);

    float v     = throttle * v_ref;
    float omega = steering * STEER_MAX_RAD_S;

    float v_left  = v - omega * (WHEEL_BASE_M * 0.5f);
    float v_right = v + omega * (WHEEL_BASE_M * 0.5f);

    // Each wheel normalized by its own max FOR ITS OWN direction.
    float l = v_left  / wheel_max(v_left,  MAX_L_FWD, MAX_L_REV);
    float r = v_right / wheel_max(v_right, MAX_R_FWD, MAX_R_REV);

    float max_mag = max(fabsf(l), fabsf(r));
    float scale = (max_mag > 1.0f) ? (1.0f / max_mag) : 1.0f;

    *left      = l * scale;
    *right     = r * scale;
    *scale_out = scale;
}

void differential_mixer_update(float throttle, float steering, diff_output_t* output)
{
    float l, r, scale;
    mix(throttle, steering, &l, &r, &scale);

    output->left  = clamp(l, -1.0f, 1.0f);
    output->right = clamp(r, -1.0f, 1.0f);
}

uint32_t differential_turn_duration_ms(float angle_rad, float steering)
{
    float l, r, scale;
    mix(0.0f, steering, &l, &r, &scale);   // pure turn

    float omega = fabsf(steering) * STEER_MAX_RAD_S * scale;
    if (omega < 1e-3f) return 0;

    return (uint32_t)((fabsf(angle_rad) / omega) * 1000.0f);
}
