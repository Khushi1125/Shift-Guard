# Code file to bridge the gap between heart rate and temperature data streamed in from Arduino
# and the AI model determining stress presence based on the data.

import serial
import serial.tools.list_ports
import requests
import time
import csv
import os
from datetime import datetime
from model import predict

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Your dashboard backend URL — change this to wherever your backend lives
DASHBOARD_URL = "http://localhost:3000/api/results"

# Baud rate must match Serial.begin(9600) in your Arduino sketch
BAUD_RATE = 9600

# Optional: log all readings to a CSV file
ENABLE_CSV_LOG = True
CSV_FILE = "readings_log.csv"

# ── AUTO-DETECT ARDUINO PORT ──────────────────────────────────────────────────

def find_arduino_port():
    """Automatically find the Arduino's serial port on Mac."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # On Mac, Arduino usually shows up as /dev/tty.usbmodem... or /dev/tty.usbserial...
        if "usbmodem" in port.device or "usbserial" in port.device:
            print(f"Found Arduino on {port.device}")
            return port.device
    # Fallback: list available ports to help debug
    print("Could not auto-detect Arduino. Available ports:")
    for port in ports:
        print(f"  {port.device} — {port.description}")
    raise Exception("Arduino not found. Plug it in and try again.")

# ── CSV LOGGING (optional) ────────────────────────────────────────────────────

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "bpm", "temp_c", "risk_score"])

def log_to_csv(bpm, temp_c, risk):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), bpm, temp_c, risk])

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    port = "/dev/cu.usbserial-0001"
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)  # wait for Arduino to reset after serial connection

    print("Bridge running. Ctrl+C to stop.\n")

    if ENABLE_CSV_LOG:
        init_csv()

    while True:
        try:
            # Read a line from Arduino e.g. "72,36.50"
            raw = ser.readline().decode("utf-8").strip()

            if not raw or "," not in raw:
                continue  # skip empty or malformed lines

            bpm_str, temp_str = raw.split(",")
            bpm = float(bpm_str)
            temp_c = float(temp_str)

            # Run the Random Forest model
            risk = predict(bpm, temp_c)

            print(f"BPM: {bpm}  Temp: {temp_c}°C  →  Risk: {risk:.2f}")

            # Send risk score back to Arduino (for LCD display)
            ser.write(f"{risk:.2f}\n".encode("utf-8"))

            # Send to dashboard backend
            try:
                requests.post(DASHBOARD_URL, json={
                    "bpm": bpm,
                    "temp_c": temp_c,
                    "risk_score": risk,
                    "timestamp": datetime.now().isoformat()
                }, timeout=2)
            except requests.exceptions.RequestException as e:
                print(f"  Dashboard send failed: {e}")

            # Log to CSV
            if ENABLE_CSV_LOG:
                log_to_csv(bpm, temp_c, risk)

        except ValueError:
            print(f"  Bad data received: '{raw}' — skipping")
        except KeyboardInterrupt:
            print("\nStopped.")
            ser.close()
            break

if __name__ == "__main__":
    main()


