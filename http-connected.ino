#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <VL53L0X.h>

// --------------------
// WIFI CONFIG
// --------------------
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";

// Your Mac LAN IP (NOT 127.0.0.1)
const char* CONTROLLER_DISTANCE_URL = "http://:8766/distance";

// --------------------
// TIMING
// --------------------
static const uint32_t SEND_INTERVAL_MS   = 500;
static const uint32_t SERIAL_INTERVAL_MS = 500;
static const uint32_t WIFI_RETRY_MS      = 5000;

// --------------------
// I2C / SENSOR
// --------------------
static const int SDA_PIN = 21;
static const int SCL_PIN = 22;

VL53L0X sensor;

// --------------------
// STATE
// --------------------
uint32_t lastSendMs   = 0;
uint32_t lastSerialMs = 0;
uint32_t lastWifiTry  = 0;
uint32_t sequence     = 0;

volatile bool wifiEverConnected = false;
volatile uint8_t lastDiscReason = 0;

static const char* wifiStatusToStr(wl_status_t s) {
  switch (s) {
    case WL_NO_SHIELD: return "WL_NO_SHIELD";
    case WL_IDLE_STATUS: return "WL_IDLE_STATUS";
    case WL_NO_SSID_AVAIL: return "WL_NO_SSID_AVAIL";
    case WL_SCAN_COMPLETED: return "WL_SCAN_COMPLETED";
    case WL_CONNECTED: return "WL_CONNECTED";
    case WL_CONNECT_FAILED: return "WL_CONNECT_FAILED";
    case WL_CONNECTION_LOST: return "WL_CONNECTION_LOST";
    case WL_DISCONNECTED: return "WL_DISCONNECTED";
    default: return "WL_UNKNOWN";
  }
}

// ESP-IDF reason codes vary by core version, but these are common/most useful.
static const char* wifiReasonToStr(uint8_t r) {
  switch (r) {
    case 1:  return "UNSPECIFIED";
    case 2:  return "AUTH_EXPIRE";
    case 3:  return "AUTH_LEAVE";
    case 4:  return "ASSOC_EXPIRE";
    case 5:  return "ASSOC_TOOMANY";
    case 6:  return "NOT_AUTHED";
    case 7:  return "NOT_ASSOCED";
    case 8:  return "ASSOC_LEAVE";
    case 9:  return "ASSOC_NOT_AUTHED";
    case 10: return "DISASSOC_PWRCAP_BAD";
    case 11: return "DISASSOC_SUPCHAN_BAD";
    case 13: return "IE_INVALID";
    case 14: return "MIC_FAILURE";
    case 15: return "4WAY_HANDSHAKE_TIMEOUT";
    case 16: return "GROUP_KEY_UPDATE_TIMEOUT";
    case 17: return "IE_IN_4WAY_DIFFERS";
    case 18: return "GROUP_CIPHER_INVALID";
    case 19: return "PAIRWISE_CIPHER_INVALID";
    case 20: return "AKMP_INVALID";
    case 21: return "UNSUPP_RSN_IE_VERSION";
    case 22: return "INVALID_RSN_IE_CAP";
    case 23: return "802_1X_AUTH_FAILED";
    case 24: return "CIPHER_SUITE_REJECTED";
    case 200: return "BEACON_TIMEOUT";
    case 201: return "NO_AP_FOUND";
    case 202: return "AUTH_FAIL";
    case 203: return "ASSOC_FAIL";
    case 204: return "HANDSHAKE_TIMEOUT";
    default: return "UNKNOWN_REASON";
  }
}

static void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_CONNECTED:
      Serial.println("[WiFi] STA_CONNECTED");
      break;

    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      wifiEverConnected = true;
      Serial.print("[WiFi] GOT_IP: ");
      Serial.println(WiFi.localIP());
      break;

    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      lastDiscReason = info.wifi_sta_disconnected.reason;
      Serial.print("[WiFi] DISCONNECTED. reason=");
      Serial.print(lastDiscReason);
      Serial.print(" (");
      Serial.print(wifiReasonToStr(lastDiscReason));
      Serial.println(")");
      break;

    default:
      // silence other events
      break;
  }
}

static void wifiStart() {
  WiFi.mode(WIFI_STA);

  // These help stability on some networks
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  Serial.print("[WiFi] begin SSID=");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

static void wifiEnsureConnected() {
  if (WiFi.status() == WL_CONNECTED) return;

  uint32_t now = millis();
  if (now - lastWifiTry < WIFI_RETRY_MS) return;
  lastWifiTry = now;

  Serial.print("[WiFi] status=");
  Serial.print(wifiStatusToStr(WiFi.status()));
  Serial.println(" -> retry begin()");
  wifiStart();
}

static bool httpPostDistance(int distanceCm, bool valid, uint32_t seq) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  http.begin(CONTROLLER_DISTANCE_URL);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"distance_cm\":" + String(distanceCm) + ",";
  payload += "\"valid\":" + String(valid ? "true" : "false") + ",";
  payload += "\"sequence\":" + String(seq);
  payload += "}";

  int code = http.POST(payload);
  http.end();

  return (code >= 200 && code < 300);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("ESP32: VL53L0X + WiFi diagnostics + POST /distance @ 500ms");

  WiFi.onEvent(onWiFiEvent);
  wifiStart();

  Wire.begin(SDA_PIN, SCL_PIN);

  sensor.setTimeout(500);
  if (!sensor.init()) {
    Serial.println("ERROR: Failed to detect VL53L0X. Check VCC/GND/SDA/SCL + solder.");
    while (true) { delay(1000); }
  }

  sensor.startContinuous();
  Serial.println("VL53L0X initialized (continuous mode).");
}

void loop() {
  delay(1);

  wifiEnsureConnected();

  uint16_t distanceMm = sensor.readRangeContinuousMillimeters();
  bool timeout = sensor.timeoutOccurred();
  bool valid = (!timeout) && (distanceMm > 30) && (distanceMm < 3000);

  uint32_t now = millis();

  if (now - lastSerialMs >= SERIAL_INTERVAL_MS) {
    lastSerialMs = now;

    Serial.print("WiFi=");
    Serial.print(wifiStatusToStr(WiFi.status()));

    if (WiFi.status() == WL_CONNECTED) {
      Serial.print(" IP=");
      Serial.print(WiFi.localIP());
      Serial.print(" RSSI=");
      Serial.print(WiFi.RSSI());
    } else {
      Serial.print(" last_reason=");
      Serial.print(lastDiscReason);
      Serial.print("(");
      Serial.print(wifiReasonToStr(lastDiscReason));
      Serial.print(")");
    }

    Serial.print("  dist=");
    Serial.print(distanceMm);
    Serial.print("mm valid=");
    Serial.println(valid ? "true" : "false");
  }

  if (now - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = now;

    int distanceCm = (int)(distanceMm / 10);

    bool ok = httpPostDistance(distanceCm, valid, sequence++);
    if (!ok) {
      // Do NOT reconnect WiFi just because POST failed
      // We already handle wifiEnsureConnected() separately
      Serial.println("[HTTP] POST /distance failed");
    }
  }
}