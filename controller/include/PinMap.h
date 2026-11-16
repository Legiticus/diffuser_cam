// Stepper motor control pins
#define DIR_PIN 8
#define STEP_PIN 7
#define EN_PIN 2

// Stepper motor directions
#define STEPPER_FORWARD 1
#define STEPPER_BACKWARD -1

// Direction control switch pins
#define FORWARD_PIN 9
#define REVERSE_PIN 10

// Adjustment potentiometer pin
#define ADJ_PIN A6

// Sensor pin
#define SENSOR_PIN A7

// LCD pins
#define RS 12
#define EN 11
#define D4 6
#define D5 5
#define D6 4
#define D7 3

// Limit Switch pins
#define FRONT_LIMIT A0 // L1
#define BACK_LIMIT A1 // L2

// Motor Rotation Values
#define STEPS_PER_REV 400 * 16 // 360 degrees per revolution / 0.9 degrees per step = 400 steps per revolution * microstepping factor
#define STEPS_PER_MM STEPS_PER_REV / 8.0 // Steps per rev / mm per rev (lead screw translation)

// Max Stepper Values
#define MAX_RPM 120 // Corresponds to 16 mm/s
#define MAX_SPEED (MAX_RPM / 60.0) * STEPS_PER_REV
#define MAX_ACCL MAX_SPEED