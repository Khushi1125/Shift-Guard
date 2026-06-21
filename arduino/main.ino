#define TEMP_POWER 33
#define TEMP_PIN 35
#define PULSE_PIN 34

int tempRawCold = 440;
float tempRealCold = 22.0;
int tempRawHot = 300;
float tempRealHot = 35.0;

unsigned long lastBeatTime = 0;
unsigned long lastValidBeatTime = 0;
int signalMin = 4095;
int signalMax = 0;
bool aboveThreshold = false;

float tempSmoothed = 440;
int lastBPM = 0;

unsigned long lastPrintTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(TEMP_POWER, OUTPUT);
  digitalWrite(TEMP_POWER, HIGH);
}

void loop() {
  int rawTemp = analogRead(TEMP_PIN);
  tempSmoothed = tempSmoothed * 0.95 + rawTemp * 0.05;
  float temperature = tempRealCold + (tempSmoothed - tempRawCold) * (tempRealHot - tempRealCold) / (tempRawHot - tempRawCold);

  int pulseValue = analogRead(PULSE_PIN);

  if (pulseValue < signalMin) signalMin = pulseValue;
  if (pulseValue > signalMax) signalMax = pulseValue;

  int dynamicThreshold = signalMin + (signalMax - signalMin) * 0.6;

  if (pulseValue > dynamicThreshold && !aboveThreshold) {
    aboveThreshold = true;
    unsigned long now = millis();
    unsigned long interval = now - lastBeatTime;

    if (interval > 300 && interval < 2000 && lastBeatTime > 0) {
      lastBPM = 60000 / interval;
      lastValidBeatTime = millis();
    }
    lastBeatTime = now;

    signalMin = pulseValue;
    signalMax = pulseValue;
  } else if (pulseValue < dynamicThreshold) {
    aboveThreshold = false;
  }

  if (millis() - lastValidBeatTime > 3000) {
    lastBPM = 0;
  }

  unsigned long currentTime = millis();
  if (currentTime - lastPrintTime >= 1000) {
    lastPrintTime = currentTime;

    int displayBPM = lastBPM;
    if (displayBPM > 150) {
      displayBPM = 110;
    }

    
    Serial.print(displayBPM);
    Serial.print(",");
    Serial.println(temperature);
  }

  delay(10);
}