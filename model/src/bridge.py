# Code file to bridge the gap between heart rate and temperature data streamed in from Arduino
# and the AI model determining stress presence based on the data.

from collections import deque
import numpy as np
import serial
import serial.tools.list_ports
import requests
import time
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from final_scoring import on_sensor_update
from arize import ArizeClient
from arize.ml.types import ModelTypes, Environments

load_dotenv()

# ARIZE CONFIG + initialize client
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
arize_client = ArizeClient(api_key=ARIZE_API_KEY)

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Your dashboard backend URL — change this to wherever your backend lives
DASHBOARD_URL = "http://localhost:8000/"

# Baud rate must match Serial.begin(115200) in your Arduino sketch
BAUD_RATE = 115200

# Optional: log all readings to a CSV file
ENABLE_CSV_LOG = True
CSV_FILE = "readings_log.csv"

# buffers for 10 max readings
WINDOW = 10
bpm_buffer = deque(maxlen=WINDOW)
temp_buffer = deque(maxlen=WINDOW)

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


# ARIZE LOGGING
def log_to_arize(bpm, temp_c, risk):
    try:
        arize_client.ml.log_stream(
            space_id=ARIZE_SPACE_ID,
            model_name="shiftguard-rf",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment="production",
            prediction_id=f"pred_{int(time.time())}",
            prediction_label=risk,
            features={
                "bpm": float(bpm),
                "temp_c": float(temp_c)
            }
        )
    except Exception as e:
        print(f"  Arize log failed: {e}")

# ── CSV LOGGING ────────────────────────────────────────────────────

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "bpm", "temp_c", "final_score"])

def log_to_csv(bpm, temp_c, risk):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), bpm, temp_c, risk])

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    # port = "/dev/cu.usbserial-0001" # /dev/cu.usbserial-0001 Serial Port (USB)
    port = find_arduino_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)  # wait for Arduino to reset after serial connection

    print("Bridge running. Ctrl+C to stop.\n")

    if ENABLE_CSV_LOG:
        init_csv()

    raw = ""
    while True:
        try:
            print("Waiting for data from Arduino...")
            # Read a line from Arduino e.g. "72,36.50"
            raw = ser.readline().decode("utf-8", errors="ignore").strip()

            if not raw or "," not in raw:
                print("Skipping malformed data:", raw)
                continue  # skip empty or malformed lines

            bpm_str, temp_str = raw.split(",")
            bpm = float(bpm_str)
            temp_c = float(temp_str)

            bpm_buffer.append(bpm)
            temp_buffer.append(temp_c)

            if len(bpm_buffer) < WINDOW:
                print(f"  Buffering... {len(bpm_buffer)}/{WINDOW}")
                continue

            # calculate buffer statistics
            bpm_mean  = float(np.mean(bpm_buffer))
            bpm_std   = float(np.std(bpm_buffer))
            temp_mean = float(np.mean(temp_buffer))
            temp_slope = float(np.polyfit(range(WINDOW), list(temp_buffer), 1)[0])
        
            # Run the model
            result = on_sensor_update(bpm_mean, temp_mean, bpm_std, temp_slope)
            risk = result["final_score"]

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

            # Log to Arize
            log_to_arize(bpm, temp_c, risk)

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


