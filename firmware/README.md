# Diff-Drive Rover — Separated Architecture

ESP32-based differential drive rover with a **standalone web controller**.

```
┌─────────────────┐         WiFi (HTTP)         ┌─────────────────────┐
│   ESP32 Board   │ ◄──────────────────────────► │  Browser Frontend   │
│  firmware/      │   POST /api/drive            │  frontend/          │
│  (API server)   │   POST /api/config           │  (WASD controller)  │
│                 │   POST /api/estop            │                     │
│                 │   GET  /api/status           │                     │
└─────────────────┘                              └─────────────────────┘
```

---

## Quick Start

### 1. Flash the Firmware

**Requirements:**
- [Arduino IDE](https://www.arduino.cc/en/software) (2.x recommended)
- ESP32 Board Package — install via Arduino IDE Board Manager:
  - Go to **File → Preferences**, add this to "Additional Board Manager URLs":
    ```
    https://espressif.github.io/arduino-esp32/package_esp32_index.json
    ```
  - Go to **Tools → Board → Board Manager**, search "esp32", install **esp32 by Espressif Systems**

**Libraries used** (all bundled with the ESP32 board package — no extra installs needed):
- `WiFi.h`
- `WebServer.h`
- `ESPmDNS.h`

**Steps:**
1. Open `firmware/esp_server.ino` in Arduino IDE
2. Edit the WiFi credentials at the top of the file:
   ```cpp
   #define WIFI_SSID     "YOUR_WIFI_SSID"
   #define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
   ```
3. Select your board: **Tools → Board → ESP32 Dev Module** (or your specific board)
4. Select the correct COM port: **Tools → Port**
5. Click **Upload**
6. Open **Serial Monitor** at 115200 baud — note the IP address printed on boot

### 2. Open the Frontend

No build tools or installation required.

1. Open `frontend/index.html` in any modern browser (Chrome, Firefox, Edge)
2. Enter the ESP32's IP address (from Serial Monitor) in the connection field
3. Start driving with **WASD keys**!

> **Tip:** You can also double-click `index.html` to open it directly. Alternatively, use `diffdrive.local` as the hostname if your network supports mDNS.

---

## Controls

| Key | Action |
|-----|--------|
| **W** | Drive forward (throttle = +1.0) |
| **S** | Drive backward (throttle = -1.0) |
| **A** | Turn left (steering = -1.0) |
| **D** | Turn right (steering = +1.0) |
| **W+A** | Forward-left arc |
| **W+D** | Forward-right arc |
| **Spacebar** | Emergency stop |
| **Release all** | Stop |

On-screen buttons also work for mobile/touch control.

---

## API Reference

All state-changing endpoints accept **POST** requests. Parameters are passed as query strings.

### `POST /api/drive`

Drive using the differential mixer.

| Param | Type | Range | Description |
|-------|------|-------|-------------|
| `throttle` | float | -1.0 to +1.0 | Forward/reverse |
| `steering` | float | -1.0 to +1.0 | Left/right turn |

### `GET /api/status`

Returns telemetry JSON with all sensor data, configuration, and motor state.

### `POST /api/config`

Set configuration parameters (all optional):

| Param | Type | Description |
|-------|------|-------------|
| `speed` | int (0–255) | Base PWM speed |
| `lscale` | float | Left motor scale multiplier |
| `rscale` | float | Right motor scale multiplier |
| `lrev` | 0 or 1 | Reverse left motor direction |
| `rrev` | 0 or 1 | Reverse right motor direction |

### `POST /api/estop`

Emergency stop. Immediately kills both motors and clears all drive state. No parameters.

---

## Safety Features

- **500ms command timeout** — if the firmware receives no `/api/drive` within 500ms, motors auto-stop (prevents runaways on WiFi drop)
- **E-Stop** — dedicated emergency stop endpoint + keyboard shortcut (spacebar)
- **Motors start stopped** — must explicitly send a drive command to move

---

## Pin Wiring Reference

| Signal | Pin | Notes |
|--------|-----|-------|
| Left Motor IN1 | 32 | |
| Left Motor IN2 | 26 | |
| Left Motor EN | 13 | PWM |
| Right Motor IN1 | 25 | |
| Right Motor IN2 | 14 | |
| Right Motor EN | 33 | PWM |
| Left Encoder A | 22 | |
| Left Encoder B | 23 | |
| Right Encoder A | 18 | |
| Right Encoder B | 21 | |

Edit `HAS_LEFT_MOTOR`, `HAS_RIGHT_MOTOR`, etc. flags in `esp_server.ino` to disable any unconnected hardware.

---

## Serial Commands (debugging)

Connect via Serial Monitor at 115200 baud. Same commands still work alongside the web API:

```
SPEED <0-255>     Set base speed
LSCALE <float>    Set left motor scale
RSCALE <float>    Set right motor scale
LREV <0|1>        Reverse left motor
RREV <0|1>        Reverse right motor
DRIVE <T> <S>     Set throttle and steering (-1.0 to 1.0)
STOP              Stop motors
ESTOP             Emergency stop
STATUS            Print current config
HELP              List commands
```
