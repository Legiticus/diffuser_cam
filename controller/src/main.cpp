#include <Arduino.h>
#include <AccelStepper.h>
#include <LiquidCrystal.h>
#include "PinMap.h"

// Zeros the position of the motor
void zeroPosition();

// Checks if the limit has been triggered
bool limitTriggered(int pin);

// Initialize stepper motor and LCD display
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
LiquidCrystal lcd(RS,EN,D4,D5,D6,D7);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1000);

	pinMode(STEP_PIN, OUTPUT);
	pinMode(DIR_PIN, OUTPUT);
	pinMode(EN_PIN, OUTPUT);
	pinMode(FRONT_LIMIT, INPUT_PULLUP);
	pinMode(BACK_LIMIT, INPUT_PULLUP);

	digitalWrite(EN_PIN, HIGH);
	digitalWrite(STEP_PIN, LOW);
	digitalWrite(DIR_PIN, STEPPER_FORWARD);

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(MAX_ACCL);

}

void loop() {

  // Read line from serial when availible
  if (Serial.available() > 0) {
    String inputStream = Serial.readStringUntil('\n');
    inputStream.trim(); // Removes leading and trailing spaces

    // Convert input stream to char array for strtok
    char inputBuffer[50];
    inputStream.toCharArray(inputBuffer, sizeof(inputBuffer));

    char* cmd = strtok(inputBuffer, " ");
    char* arg = strtok(NULL, " ");

    // Parse Command
    if (strcmp(cmd, "ZERO") == 0) { // Zero laser position

      zeroPosition();
      Serial.println("ZERO_SUCCESS");

    }else if (strcmp(cmd, "MOVE") == 0) { // Move laser position

      if (arg != NULL) {

        bool success = true;
        float arg_val = atof(arg);
        stepper.move(STEPPER_FORWARD * STEPS_PER_MM * arg_val);
        
        // Run motor until position checking for limit switch
        while (stepper.currentPosition() != stepper.targetPosition()) {
          // Run one step
          stepper.run();
          // Check if limit switch triggered
          if (limitTriggered(BACK_LIMIT)) {
            Serial.println("MOVE_FAIL - Back Limit Triggered");
            success = false;
            break;
          }
          else if (limitTriggered(FRONT_LIMIT)) {
            Serial.println("MOVE_FAIL - Front Limit Triggered");
            success = false;
            break;
          }
        }

        // Print return msg
        if (success) Serial.println("MOVE_SUCCESS");

      }else {
        Serial.println("MOVE_FAIL - No argument passed");
      }

    }else {
      Serial.println("ERROR - Not a valid command");
    }
  }
}

void zeroPosition() {
  
  /*
   * Phase 1: Move FORWARD until limit triggered
  */

  // Move to an arbitrary large number of steps FORWARD
  stepper.moveTo(STEPPER_FORWARD * 10000000);
  while (!limitTriggered(FRONT_LIMIT)) {
    stepper.run();
  }
  stepper.setCurrentPosition(stepper.currentPosition()); // Set target to current position

  /*
   * Phase 2: Raise UP until limit not triggered one mm at a time
  */

  while (limitTriggered(FRONT_LIMIT)) {
    stepper.move(STEPPER_BACKWARD * STEPS_PER_MM * 1); //move 1 mm at a time
    stepper.runToPosition();
  }

  /*
   * Phase 3: Move DOWN until limit triggered one tenth of a mm at a time
  */

  while (!limitTriggered(FRONT_LIMIT)) {
    stepper.move(STEPPER_FORWARD * STEPS_PER_MM * 0.1); //move 0.1 mm at a time
    stepper.runToPosition();
  }

  /*
   * Phase 4: Move up to be right above limit
  */
  while (limitTriggered(FRONT_LIMIT)) {
    stepper.move(STEPPER_BACKWARD * STEPS_PER_MM * 0.01);
    stepper.runToPosition();
  }

  /*
   * Set zero position
  */
  stepper.setCurrentPosition(0);

}

bool limitTriggered(int pin) {
  return (digitalRead(pin) == LOW);
}