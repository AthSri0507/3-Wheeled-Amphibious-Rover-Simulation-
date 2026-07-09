/*
    DIFFERENTIAL DRIVE — API-ONLY FIRMWARE
    =======================================

    This firmware runs on the ESP32 and exposes a REST API for
    controlling the differential drive rover. NO HTML is served —
    the frontend is a separate static web app that talks to these
    endpoints over WiFi.

    API Endpoints:
        POST /api/drive?throttle=F&steering=F
            Drive using differential mixer. Both -1.0 to +1.0.

        GET  /api/status
            Returns full telemetry JSON (read-only).

        POST /api/config?speed=N&lscale=F&rscale=F&lrev=0|1&rrev=0|1
            Set configuration parameters. All optional.

        POST /api/estop
            Emergency stop. Kills motors immediately.

    Safety:
        - 500ms command timeout: if no /api/drive received within
          500ms, motors auto-stop.
        - Motors start in STOPPED state on boot.
        - CORS headers on all responses (Access-Control-Allow-Origin: *)
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "encoder.h"
#include "odometry.h"
#include "l289.h"
#include "differential.h"

/*
    =========================
    WI-FI CONFIGURATION
    =========================
*/

#define WIFI_SSID       "AMDEAAAR_2.4G"
#define WIFI_PASSWORD   "amdeaaar77"
#define MDNS_HOSTNAME   "diffdrive"

WebServer web_server(80);

/*
    =========================
    HARDWARE CONFIGURATION
    =========================
*/

#define HAS_LEFT_MOTOR     1
#define HAS_RIGHT_MOTOR    1

#define HAS_LEFT_ENCODER   1
#define HAS_RIGHT_ENCODER  1

/*
    =========================
    MOTOR CONFIGURATION
    =========================
*/

#if HAS_LEFT_MOTOR
l298_motor_t left_motor =
{
    .in1 = 32,
    .in2 = 26,
    .en  = 13
};
#endif

#if HAS_RIGHT_MOTOR
l298_motor_t right_motor =
{
    .in1 = 25,
    .in2 = 14,
    .en  = 33
};
#endif

/*
    =========================
    ENCODER CONFIGURATION
    =========================
*/

#if HAS_LEFT_ENCODER
encoder_t left_encoder =
{
    .pin_a = 22,
    .pin_b = 23
};
#endif

#if HAS_RIGHT_ENCODER
encoder_t right_encoder =
{
    .pin_a = 18,
    .pin_b = 21
};
#endif

/*
    =========================
    ODOMETRY CONFIGURATION
    =========================
*/

odometry_t odom =
{
    #if HAS_LEFT_ENCODER
    .left_encoder = &left_encoder,
    #else
    .left_encoder = nullptr,
    #endif

    #if HAS_RIGHT_ENCODER
    .right_encoder = &right_encoder,
    #else
    .right_encoder = nullptr,
    #endif

    .ticks_per_motor_rev = 28.0f,
    .gear_ratio          = 200.0f,
    .wheel_radius        = 0.031f,
    .wheel_base          = 0.18f
};

/*
    =========================
    RUNTIME CONFIGURATION
    =========================
*/

uint8_t base_speed   = 150;
float   left_scale   = 1.0f;
float   right_scale  = 1.0f;
bool    left_invert  = false;
bool    right_invert = false;

/*
    =========================
    DRIVE STATE
    =========================
*/

float   current_throttle = 0.0f;
float   current_steering = 0.0f;
bool    motor_running    = false;

/*
    =========================
    COMMAND TIMEOUT
    500ms without a /api/drive → auto-stop
    =========================
*/

#define COMMAND_TIMEOUT_MS 500
uint32_t last_drive_time   = 0;
bool     drive_active      = false;

/*
    =========================
    TIMING
    =========================
*/

unsigned long last_update_time = 0;
String serial_line = "";

/*
    =========================
    MOTOR HELPERS
    =========================
*/

uint8_t scaled_speed(uint8_t base, float scale)
{
    float result = (float)base * scale;

    if (result < 0.0f)   result = 0.0f;
    if (result > 255.0f) result = 255.0f;

    return (uint8_t)result;
}

l298_direction_t effective_direction(l298_direction_t direction, bool invert)
{
    if (!invert)
    {
        return direction;
    }

    if (direction == L298_FORWARD)
    {
        return L298_REVERSE;
    }

    if (direction == L298_REVERSE)
    {
        return L298_FORWARD;
    }

    return direction;
}

/*
    Apply differential mixer output to physical motors.
    mixer_val is -1.0 to +1.0 per wheel.
*/
void apply_mixer_output(float left_mix, float right_mix)
{
    // Convert mixer output to direction + speed
    l298_direction_t left_dir  = (left_mix >= 0.0f) ? L298_FORWARD : L298_REVERSE;
    l298_direction_t right_dir = (right_mix >= 0.0f) ? L298_FORWARD : L298_REVERSE;

    uint8_t left_spd  = scaled_speed(base_speed, fabsf(left_mix)  * left_scale);
    uint8_t right_spd = scaled_speed(base_speed, fabsf(right_mix) * right_scale);

    // Apply inversion
    left_dir  = effective_direction(left_dir, left_invert);
    right_dir = effective_direction(right_dir, right_invert);

#if HAS_LEFT_MOTOR
    l298_drive(&left_motor, left_dir, left_spd);
#endif

#if HAS_RIGHT_MOTOR
    l298_drive(&right_motor, right_dir, right_spd);
#endif
}

void stop_motors()
{
#if HAS_LEFT_MOTOR
    l298_stop(&left_motor);
#endif

#if HAS_RIGHT_MOTOR
    l298_stop(&right_motor);
#endif
}

/*
    =========================
    CORS HELPER
    =========================
*/

void send_cors_headers()
{
    web_server.sendHeader("Access-Control-Allow-Origin", "*");
    web_server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    web_server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

/*
    =========================
    JSON STATUS BUILDER
    =========================
*/

String build_status_json()
{
    String json = "{";

    json += "\"base_speed\":" + String(base_speed) + ",";
    json += "\"left_scale\":" + String(left_scale, 3) + ",";
    json += "\"right_scale\":" + String(right_scale, 3) + ",";
    json += "\"left_invert\":" + String(left_invert  ? 1 : 0) + ",";
    json += "\"right_invert\":" + String(right_invert ? 1 : 0) + ",";
    json += "\"running\":" + String(motor_running ? 1 : 0) + ",";
    json += "\"throttle\":" + String(current_throttle, 3) + ",";
    json += "\"steering\":" + String(current_steering, 3) + ",";
    json += "\"left_applied_speed\":" + String(scaled_speed(base_speed, left_scale)) + ",";
    json += "\"right_applied_speed\":" + String(scaled_speed(base_speed, right_scale)) + ",";

#if HAS_LEFT_ENCODER
    json += "\"left_ticks\":" + String(encoder_get_ticks(&left_encoder)) + ",";
#else
    json += "\"left_ticks\":0,";
#endif

#if HAS_RIGHT_ENCODER
    json += "\"right_ticks\":" + String(encoder_get_ticks(&right_encoder)) + ",";
#else
    json += "\"right_ticks\":0,";
#endif

    json += "\"left_motor_rpm\":"   + String(odom.data.left_motor_rpm, 2) + ",";
    json += "\"right_motor_rpm\":"  + String(odom.data.right_motor_rpm, 2) + ",";
    json += "\"left_wheel_rpm\":"   + String(odom.data.left_wheel_rpm, 2) + ",";
    json += "\"right_wheel_rpm\":"  + String(odom.data.right_wheel_rpm, 2) + ",";
    json += "\"left_velocity\":"    + String(odom.data.left_velocity, 3) + ",";
    json += "\"right_velocity\":"   + String(odom.data.right_velocity, 3) + ",";
    json += "\"linear_velocity\":"  + String(odom.data.linear_velocity, 3) + ",";
    json += "\"angular_velocity\":" + String(odom.data.angular_velocity, 3) + ",";
    json += "\"heading_deg\":"      + String(degrees(odom.data.heading), 2) + ",";
    json += "\"x\":"                + String(odom.data.x, 3) + ",";
    json += "\"y\":"                + String(odom.data.y, 3) + ",";
    json += "\"distance\":"         + String(odom.data.distance, 3);

    json += "}";

    return json;
}

/*
    =========================
    API HANDLERS
    =========================
*/

// POST /api/drive?throttle=F&steering=F
void handle_api_drive()
{
    send_cors_headers();

    float throttle = 0.0f;
    float steering = 0.0f;

    if (web_server.hasArg("throttle"))
    {
        throttle = web_server.arg("throttle").toFloat();
        if (throttle < -1.0f) throttle = -1.0f;
        if (throttle >  1.0f) throttle =  1.0f;
    }

    if (web_server.hasArg("steering"))
    {
        steering = web_server.arg("steering").toFloat();
        if (steering < -1.0f) steering = -1.0f;
        if (steering >  1.0f) steering =  1.0f;
    }

    current_throttle = throttle;
    current_steering = steering;

    // Feed into differential mixer
    diff_output_t mix_out;
    differential_mixer_update(throttle, steering, &mix_out);

    // Check if we should be moving
    bool should_move = (fabsf(throttle) > 0.01f || fabsf(steering) > 0.01f);

    if (should_move)
    {
        motor_running = true;
        drive_active  = true;
        last_drive_time = millis();

        apply_mixer_output(mix_out.left, mix_out.right);
    }
    else
    {
        motor_running = false;
        drive_active  = false;
        stop_motors();
    }

    web_server.send(200, "application/json", build_status_json());
}

// GET /api/status
void handle_api_status()
{
    send_cors_headers();
    web_server.send(200, "application/json", build_status_json());
}

// POST /api/config?speed=N&lscale=F&rscale=F&lrev=0|1&rrev=0|1
void handle_api_config()
{
    send_cors_headers();

    if (web_server.hasArg("speed"))
    {
        int val = web_server.arg("speed").toInt();
        if (val < 0)   val = 0;
        if (val > 255) val = 255;
        base_speed = (uint8_t)val;
    }

    if (web_server.hasArg("lscale"))
    {
        left_scale = web_server.arg("lscale").toFloat();
    }

    if (web_server.hasArg("rscale"))
    {
        right_scale = web_server.arg("rscale").toFloat();
    }

    if (web_server.hasArg("lrev"))
    {
        left_invert = (web_server.arg("lrev").toInt() != 0);
    }

    if (web_server.hasArg("rrev"))
    {
        right_invert = (web_server.arg("rrev").toInt() != 0);
    }

    web_server.send(200, "application/json", build_status_json());
}

// POST /api/estop
void handle_api_estop()
{
    send_cors_headers();

    current_throttle = 0.0f;
    current_steering = 0.0f;
    motor_running    = false;
    drive_active     = false;

    stop_motors();

    Serial.println("!!! EMERGENCY STOP !!!");

    web_server.send(200, "application/json", build_status_json());
}

// OPTIONS preflight handler (for CORS)
void handle_options()
{
    send_cors_headers();
    web_server.send(204);
}

void handle_not_found()
{
    send_cors_headers();
    web_server.send(404, "text/plain", "Not found");
}

/*
    =========================
    SERIAL COMMAND PARSER
    (kept for debugging — same commands as before)
    =========================
*/

void print_status()
{
    Serial.println("---- CURRENT CONFIG ----");
    Serial.print("base_speed:  "); Serial.println(base_speed);
    Serial.print("left_scale:  "); Serial.println(left_scale, 3);
    Serial.print("right_scale: "); Serial.println(right_scale, 3);
    Serial.print("left_invert:  "); Serial.println(left_invert  ? "YES (reversed)" : "no");
    Serial.print("right_invert: "); Serial.println(right_invert ? "YES (reversed)" : "no");
    Serial.print("left speed (applied):  "); Serial.println(scaled_speed(base_speed, left_scale));
    Serial.print("right speed (applied): "); Serial.println(scaled_speed(base_speed, right_scale));
    Serial.print("motor_running: "); Serial.println(motor_running ? "YES" : "NO");
    Serial.print("throttle: "); Serial.println(current_throttle, 3);
    Serial.print("steering: "); Serial.println(current_steering, 3);
    Serial.println("------------------------");
}

void print_help()
{
    Serial.println("---- SERIAL COMMANDS ----");
    Serial.println("SPEED <0-255>   set base speed for both motors");
    Serial.println("LSCALE <float>  set left motor speed multiplier");
    Serial.println("RSCALE <float>  set right motor speed multiplier");
    Serial.println("LREV <0|1>      reverse left motor direction");
    Serial.println("RREV <0|1>      reverse right motor direction");
    Serial.println("DRIVE <T> <S>   set throttle and steering (-1.0 to 1.0)");
    Serial.println("STOP            stop both motors");
    Serial.println("ESTOP           emergency stop");
    Serial.println("STATUS          print current configuration");
    Serial.println("HELP            print this message");
    Serial.println("-------------------------");
}

void handle_command(String line)
{
    line.trim();

    if (line.length() == 0)
    {
        return;
    }

    // Parse first word as command
    int space_index = line.indexOf(' ');

    String cmd = (space_index == -1) ? line : line.substring(0, space_index);
    String arg = (space_index == -1) ? ""   : line.substring(space_index + 1);

    cmd.trim();
    arg.trim();
    cmd.toUpperCase();

    if (cmd == "SPEED")
    {
        int val = arg.toInt();

        if (val < 0)   val = 0;
        if (val > 255) val = 255;

        base_speed = (uint8_t)val;

        Serial.print("OK base_speed = ");
        Serial.println(base_speed);
    }
    else if (cmd == "LSCALE")
    {
        left_scale = arg.toFloat();

        Serial.print("OK left_scale = ");
        Serial.println(left_scale, 3);
    }
    else if (cmd == "RSCALE")
    {
        right_scale = arg.toFloat();

        Serial.print("OK right_scale = ");
        Serial.println(right_scale, 3);
    }
    else if (cmd == "LREV")
    {
        left_invert = (arg.toInt() != 0);

        Serial.print("OK left_invert = ");
        Serial.println(left_invert ? "1 (reversed)" : "0 (normal)");
    }
    else if (cmd == "RREV")
    {
        right_invert = (arg.toInt() != 0);

        Serial.print("OK right_invert = ");
        Serial.println(right_invert ? "1 (reversed)" : "0 (normal)");
    }
    else if (cmd == "DRIVE")
    {
        // Parse "DRIVE <throttle> <steering>"
        int sep = arg.indexOf(' ');
        if (sep != -1)
        {
            float t = arg.substring(0, sep).toFloat();
            float s = arg.substring(sep + 1).toFloat();

            if (t < -1.0f) t = -1.0f;
            if (t >  1.0f) t =  1.0f;
            if (s < -1.0f) s = -1.0f;
            if (s >  1.0f) s =  1.0f;

            current_throttle = t;
            current_steering = s;

            diff_output_t mix_out;
            differential_mixer_update(t, s, &mix_out);

            bool should_move = (fabsf(t) > 0.01f || fabsf(s) > 0.01f);

            if (should_move)
            {
                motor_running = true;
                drive_active  = true;
                last_drive_time = millis();
                apply_mixer_output(mix_out.left, mix_out.right);
            }
            else
            {
                motor_running = false;
                drive_active  = false;
                stop_motors();
            }

            Serial.print("OK throttle=");
            Serial.print(t, 3);
            Serial.print(" steering=");
            Serial.println(s, 3);
        }
        else
        {
            Serial.println("Usage: DRIVE <throttle> <steering>");
        }
    }
    else if (cmd == "STOP")
    {
        current_throttle = 0.0f;
        current_steering = 0.0f;
        motor_running = false;
        drive_active  = false;
        stop_motors();

        Serial.println("OK motors stopped");
    }
    else if (cmd == "ESTOP")
    {
        current_throttle = 0.0f;
        current_steering = 0.0f;
        motor_running = false;
        drive_active  = false;
        stop_motors();

        Serial.println("!!! EMERGENCY STOP !!!");
    }
    else if (cmd == "STATUS")
    {
        print_status();
    }
    else if (cmd == "HELP")
    {
        print_help();
    }
    else
    {
        Serial.print("Unknown command: ");
        Serial.println(cmd);
        Serial.println("Type HELP for the list of commands.");
    }
}

void process_serial_input()
{
    while (Serial.available() > 0)
    {
        char c = (char)Serial.read();

        if (c == '\n')
        {
            handle_command(serial_line);
            serial_line = "";
        }
        else if (c != '\r')
        {
            serial_line += c;
        }
    }
}

/*
    =========================
    SETUP
    =========================
*/

void setup()
{
    Serial.begin(115200);

#if HAS_LEFT_MOTOR
    l298_init(&left_motor);
#endif

#if HAS_RIGHT_MOTOR
    l298_init(&right_motor);
#endif

#if HAS_LEFT_ENCODER
    encoder_init(&left_encoder);
#endif

#if HAS_RIGHT_ENCODER
    encoder_init(&right_encoder);
#endif

    odometry_init(&odom);

    Serial.println();
    Serial.println("================================");
    Serial.println("DIFF-DRIVE API SERVER");
    Serial.println("================================");

    Serial.print("Left motor:    ");
    Serial.println(HAS_LEFT_MOTOR ? "ENABLED" : "disabled");

    Serial.print("Right motor:   ");
    Serial.println(HAS_RIGHT_MOTOR ? "ENABLED" : "disabled");

    Serial.print("Left encoder:  ");
    Serial.println(HAS_LEFT_ENCODER ? "ENABLED" : "disabled");

    Serial.print("Right encoder: ");
    Serial.println(HAS_RIGHT_ENCODER ? "ENABLED" : "disabled");

    Serial.println();
    print_help();
    Serial.println();
    print_status();

    /*
        Connect to Wi-Fi
    */

    Serial.println();
    Serial.print("Connecting to Wi-Fi: ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint32_t wifi_start = millis();

    while (WiFi.status() != WL_CONNECTED && (millis() - wifi_start) < 15000)
    {
        delay(300);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.print("Wi-Fi connected. IP address: ");
        Serial.println(WiFi.localIP());

        if (MDNS.begin(MDNS_HOSTNAME))
        {
            Serial.print("API also reachable at: http://");
            Serial.print(MDNS_HOSTNAME);
            Serial.println(".local/");
        }
        else
        {
            Serial.println("mDNS failed — use the IP address above.");
        }

        /*
            Register API routes
        */

        // Drive endpoint (POST)
        web_server.on("/api/drive", HTTP_POST, handle_api_drive);
        web_server.on("/api/drive", HTTP_GET,  handle_api_drive);  // Allow GET too for easy testing

        // Status endpoint (GET)
        web_server.on("/api/status", HTTP_GET, handle_api_status);

        // Config endpoint (POST)
        web_server.on("/api/config", HTTP_POST, handle_api_config);
        web_server.on("/api/config", HTTP_GET,  handle_api_config);  // Allow GET for easy testing

        // Emergency stop (POST)
        web_server.on("/api/estop", HTTP_POST, handle_api_estop);
        web_server.on("/api/estop", HTTP_GET,  handle_api_estop);  // Allow GET for quick browser testing

        // CORS preflight
        web_server.on("/api/drive",  HTTP_OPTIONS, handle_options);
        web_server.on("/api/status", HTTP_OPTIONS, handle_options);
        web_server.on("/api/config", HTTP_OPTIONS, handle_options);
        web_server.on("/api/estop",  HTTP_OPTIONS, handle_options);

        web_server.onNotFound(handle_not_found);

        web_server.begin();

        Serial.println("API server started. No HTML served — use the external frontend.");
    }
    else
    {
        Serial.println("Wi-Fi FAILED — check WIFI_SSID/WIFI_PASSWORD.");
        Serial.println("Falling back to serial-only control.");
    }

#if HAS_LEFT_ENCODER
    encoder_reset(&left_encoder);
#endif

#if HAS_RIGHT_ENCODER
    encoder_reset(&right_encoder);
#endif

    /*
        Motors start STOPPED — wait for a drive command
    */

    motor_running = false;
    drive_active  = false;
}

/*
    =========================
    LOOP
    =========================
*/

void loop()
{
    /*
        Process serial & HTTP
    */

    process_serial_input();
    web_server.handleClient();

    /*
        Command timeout: auto-stop if no drive command for 500ms
    */

    if (drive_active && (millis() - last_drive_time >= COMMAND_TIMEOUT_MS))
    {
        current_throttle = 0.0f;
        current_steering = 0.0f;
        motor_running    = false;
        drive_active     = false;

        stop_motors();

        Serial.println("Command timeout — motors stopped.");
    }

    /*
        Update odometry every 100ms
    */

    if (millis() - last_update_time >= 100)
    {
        last_update_time = millis();

        odometry_update(&odom);

        // Print diagnostics
        #if HAS_LEFT_ENCODER
            Serial.print("L ticks: ");
            Serial.print(encoder_get_ticks(&left_encoder));
            Serial.print("  ");
        #endif

        #if HAS_RIGHT_ENCODER
            Serial.print("R ticks: ");
            Serial.print(encoder_get_ticks(&right_encoder));
            Serial.print("  ");
        #endif

        Serial.print("v=");
        Serial.print(odom.data.linear_velocity, 3);
        Serial.print(" w=");
        Serial.print(odom.data.angular_velocity, 3);
        Serial.print(" hdg=");
        Serial.print(degrees(odom.data.heading), 1);
        Serial.print(" x=");
        Serial.print(odom.data.x, 3);
        Serial.print(" y=");
        Serial.println(odom.data.y, 3);
    }
}
