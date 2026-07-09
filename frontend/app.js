/*
    DIFF-DRIVE CONTROLLER — Frontend Logic
    =======================================

    Keyboard state machine:
      - Tracks which WASD keys are currently pressed
      - Computes throttle + steering from key combo
      - Only sends a POST /api/drive when state CHANGES (no spam)

    Command keepalive:
      - While keys are held, resends the drive command every 300ms
        to prevent the firmware's 500ms command timeout from triggering

    Polling:
      - GET /api/status every 300ms for live telemetry
*/

// ─────────────────────────────────────────
// STATE
// ─────────────────────────────────────────

let espHost = 'diffdrive.local';
let connected = false;
let pollTimer = null;
let keepaliveTimer = null;

// Currently pressed keys
const keys = { w: false, a: false, s: false, d: false };

// Last sent values (to avoid duplicate sends)
let lastThrottle = null;
let lastSteering = null;

// ─────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────

const $ = id => document.getElementById(id);

function init() {
    const hostInput   = $('host-input');
    const statusDot   = $('status-dot');
    const statusText  = $('status-text');
    const speedSlider = $('speed-slider');
    const speedVal    = $('speed-val');
    const estopBtn    = $('estop-btn');

    // Load saved host
    const saved = localStorage.getItem('diffdrive-host');
    if (saved) {
        espHost = saved;
        hostInput.value = saved;
    } else {
        hostInput.value = espHost;
    }

    // Host input
    hostInput.addEventListener('change', () => {
        espHost = hostInput.value.trim();
        localStorage.setItem('diffdrive-host', espHost);
        setConnected(false);
    });

    // Speed slider
    speedSlider.addEventListener('input', () => {
        speedVal.textContent = speedSlider.value;
    });

    speedSlider.addEventListener('change', () => {
        sendConfig({ speed: speedSlider.value });
    });

    // E-Stop
    estopBtn.addEventListener('click', sendEstop);

    // WASD keyboard
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    // WASD touch/click buttons
    setupTouchButton('key-w', 'w');
    setupTouchButton('key-a', 'a');
    setupTouchButton('key-s', 's');
    setupTouchButton('key-d', 'd');

    // Advanced config toggle
    const advToggle = $('adv-toggle');
    const advBody   = $('adv-body');
    advToggle.addEventListener('click', () => {
        advToggle.classList.toggle('open');
        advBody.classList.toggle('show');
    });

    // Config inputs
    $('cfg-lscale').addEventListener('change', () => {
        sendConfig({ lscale: $('cfg-lscale').value });
    });

    $('cfg-rscale').addEventListener('change', () => {
        sendConfig({ rscale: $('cfg-rscale').value });
    });

    $('cfg-lrev').addEventListener('change', () => {
        sendConfig({ lrev: $('cfg-lrev').checked ? 1 : 0 });
    });

    $('cfg-rrev').addEventListener('change', () => {
        sendConfig({ rrev: $('cfg-rrev').checked ? 1 : 0 });
    });

    // Start polling
    poll();
    pollTimer = setInterval(poll, 300);
}

// ─────────────────────────────────────────
// CONNECTION
// ─────────────────────────────────────────

function setConnected(ok) {
    connected = ok;
    const dot  = $('status-dot');
    const text = $('status-text');
    dot.classList.toggle('ok', ok);
    text.textContent = ok ? 'connected' : 'disconnected';
}

function apiUrl(path) {
    // Ensure no double slashes
    const host = espHost.replace(/\/+$/, '');
    return `http://${host}${path}`;
}

// ─────────────────────────────────────────
// KEYBOARD HANDLING
// ─────────────────────────────────────────

function onKeyDown(e) {
    const key = e.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key) && !keys[key]) {
        e.preventDefault();
        keys[key] = true;
        updateKeyVisuals();
        sendDriveFromKeys();
    }

    // Spacebar = E-Stop
    if (e.key === ' ') {
        e.preventDefault();
        sendEstop();
    }
}

function onKeyUp(e) {
    const key = e.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key) && keys[key]) {
        e.preventDefault();
        keys[key] = false;
        updateKeyVisuals();
        sendDriveFromKeys();
    }
}

function updateKeyVisuals() {
    ['w', 'a', 's', 'd'].forEach(k => {
        const el = $('key-' + k);
        if (el) el.classList.toggle('active', keys[k]);
    });
}

// ─────────────────────────────────────────
// TOUCH SUPPORT FOR ON-SCREEN BUTTONS
// ─────────────────────────────────────────

function setupTouchButton(elementId, key) {
    const el = $(elementId);
    if (!el) return;

    const press = (e) => {
        e.preventDefault();
        if (!keys[key]) {
            keys[key] = true;
            updateKeyVisuals();
            sendDriveFromKeys();
        }
    };

    const release = (e) => {
        e.preventDefault();
        if (keys[key]) {
            keys[key] = false;
            updateKeyVisuals();
            sendDriveFromKeys();
        }
    };

    el.addEventListener('mousedown', press);
    el.addEventListener('mouseup', release);
    el.addEventListener('mouseleave', release);
    el.addEventListener('touchstart', press, { passive: false });
    el.addEventListener('touchend', release, { passive: false });
    el.addEventListener('touchcancel', release, { passive: false });
}

// ─────────────────────────────────────────
// DRIVE COMMAND
// ─────────────────────────────────────────

function sendDriveFromKeys() {
    let throttle = 0;
    let steering = 0;

    if (keys.w) throttle += 1.0;
    if (keys.s) throttle -= 1.0;
    if (keys.a) steering -= 1.0;
    if (keys.d) steering += 1.0;

    // Only send if changed
    if (throttle === lastThrottle && steering === lastSteering) {
        return;
    }

    lastThrottle = throttle;
    lastSteering = steering;

    sendDrive(throttle, steering);

    // Set up keepalive while moving
    clearInterval(keepaliveTimer);
    if (throttle !== 0 || steering !== 0) {
        keepaliveTimer = setInterval(() => {
            sendDrive(lastThrottle, lastSteering);
        }, 300);
    }
}

function sendDrive(throttle, steering) {
    const params = new URLSearchParams({
        throttle: throttle.toFixed(2),
        steering: steering.toFixed(2)
    });

    fetch(apiUrl('/api/drive?' + params.toString()), {
        method: 'POST'
    })
    .then(r => r.json())
    .then(d => {
        setConnected(true);
        updateTelemetry(d);
    })
    .catch(() => setConnected(false));

    // Update drive indicator
    $('ind-throttle').textContent = throttle.toFixed(2);
    $('ind-steering').textContent = steering.toFixed(2);
}

// ─────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────

function sendConfig(params) {
    const qs = new URLSearchParams(params).toString();

    fetch(apiUrl('/api/config?' + qs), {
        method: 'POST'
    })
    .then(r => r.json())
    .then(d => {
        setConnected(true);
        updateTelemetry(d);
    })
    .catch(() => setConnected(false));
}

// ─────────────────────────────────────────
// E-STOP
// ─────────────────────────────────────────

function sendEstop() {
    // Immediately clear local state
    keys.w = keys.a = keys.s = keys.d = false;
    lastThrottle = 0;
    lastSteering = 0;
    clearInterval(keepaliveTimer);
    updateKeyVisuals();

    $('ind-throttle').textContent = '0.00';
    $('ind-steering').textContent = '0.00';

    fetch(apiUrl('/api/estop'), {
        method: 'POST'
    })
    .then(r => r.json())
    .then(d => {
        setConnected(true);
        updateTelemetry(d);
    })
    .catch(() => setConnected(false));
}

// ─────────────────────────────────────────
// POLLING
// ─────────────────────────────────────────

function poll() {
    fetch(apiUrl('/api/status'))
        .then(r => r.json())
        .then(d => {
            setConnected(true);
            updateTelemetry(d);
        })
        .catch(() => setConnected(false));
}

// ─────────────────────────────────────────
// UI UPDATE
// ─────────────────────────────────────────

function updateTelemetry(d) {
    // Speed slider sync (only if not focused)
    const slider = $('speed-slider');
    if (document.activeElement !== slider) {
        slider.value = d.base_speed;
        $('speed-val').textContent = d.base_speed;
    }

    // Left channel
    $('l-bar').style.width = (d.left_applied_speed / 255 * 100) + '%';
    $('l-output').textContent = d.left_applied_speed + ' / 255';
    $('l-rpm').textContent = d.left_wheel_rpm;
    $('l-vel').textContent = d.left_velocity;
    $('l-ticks').textContent = d.left_ticks;

    // Right channel
    $('r-bar').style.width = (d.right_applied_speed / 255 * 100) + '%';
    $('r-output').textContent = d.right_applied_speed + ' / 255';
    $('r-rpm').textContent = d.right_wheel_rpm;
    $('r-vel').textContent = d.right_velocity;
    $('r-ticks').textContent = d.right_ticks;

    // Pose
    $('pose-heading').textContent = d.heading_deg + '°';
    $('pose-dist').textContent = d.distance + ' m';
    $('pose-x').textContent = d.x + ' m';
    $('pose-y').textContent = d.y + ' m';

    // Compass needle
    const needle = $('needle');
    if (needle) {
        needle.style.transform = 'rotate(' + d.heading_deg + 'deg)';
    }

    // Running state
    const runDot = $('run-indicator');
    if (runDot) {
        runDot.style.background = d.running ? 'var(--good)' : 'var(--danger)';
    }

    // Advanced config sync (only if not focused)
    syncIfIdle('cfg-lscale', d.left_scale);
    syncIfIdle('cfg-rscale', d.right_scale);

    const lrev = $('cfg-lrev');
    const rrev = $('cfg-rrev');
    if (document.activeElement !== lrev) lrev.checked = !!d.left_invert;
    if (document.activeElement !== rrev) rrev.checked = !!d.right_invert;
}

function syncIfIdle(id, value) {
    const el = $(id);
    if (el && document.activeElement !== el) {
        el.value = value;
    }
}

// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
