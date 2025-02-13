//Initialize Arduino

#if (ARDUINO >= 100)
 #include <Arduino.h>
#else
 #include <WProgram.h>
#endif

//Initialize Servo Library

#include <Servo.h> //PWM
#include "serial_servo.h" //Serial

//Initialize ROS libraries
#include <ros.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32MultiArray.h>

//Initialize LCD
#include <LiquidCrystal.h>

//Initialize Node Handle
ros::NodeHandle  nh;

//Declare Servo Objects
Servo servoPWM;
Servo servoSpray;
SerialServo servoSerial(Serial2); //Pin 16,17

//Declare LCD Objects
const int RS = 12, EN = 11, D4 = 5, D5 = 4, D6 = 3, D7 = 2;
LiquidCrystal lcd(RS, EN, D4, D5, D6, D7);

float index = 0;
float angle1 = 0;
float angle2 = 0;
float angle3 = 0;
float angle4 = 0;
float totalAngles = 0;
float arduinoPresent = 0.0;



// Servo 1 - PWM Callback - Joint 1 - 65kg
void servoCB1(float servo1){
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo1");
  lcd.setCursor(0, 1);
  lcd.print(servo1);
  
  servoPWM.write(servo1); //set servo angle, should be from 0-180  

  delay(1000);
}

//Servo 2 - Serial Callback - Joint 2 - 45kg

void servoCB2(float servo2){

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo2");
  //lcd.setCursor(0, 1);
  //lcd.print(servo2);

  uint8_t servoID = 1;  // Set Servo ID

  //Send control instructions to the specified servo.
  float targetAngle = servo2;  // Set target angle 0-270
  uint16_t moveTime = 1000;  // Complete turn within X time 0-3000
  servoSerial.move_servo_immediate(servoID, targetAngle, moveTime);
  
  delay(1000);
}

//Servo 3 - Serial Callback - Joint 3 - 45kg

void servoCB3(float servo3){

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo3");
  //lcd.setCursor(0, 1);
  //lcd.print(servo3);

  uint8_t servoID = 3;  // Set Servo ID

  //Send control instructions to the specified servo.
  float targetAngle = servo3;  // Set target angle 0-270
  uint16_t moveTime = 1000;  // Complete turn within X time 0-3000
  servoSerial.move_servo_immediate(servoID, targetAngle, moveTime);
  
  delay(1000);
}

//Servo 4 - Serial Callback - Yaw - Joint 4 - 45kg

void servoCB4(float servo4){

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo4");
  //lcd.setCursor(0, 1);
  //lcd.print(servo4);

  uint8_t servoID = 2;  // Set Servo ID

  //Send control instructions to the specified servo.
  float targetAngle = servo4;  // Set target angle 0-270
  uint16_t moveTime = 1000;  // Complete turn within X time 0-3000
  servoSerial.move_servo_immediate(servoID, targetAngle, moveTime);


  //Get the preset angle and time of the servo.
  float currentAngle = 0.0f;
  uint16_t currentTime = 0;
  t_FuncRet result_get_2 = servoSerial.get_servo_move_immediate(servoID, &currentAngle, &currentTime);
  if (result_get_2 == Operation_Success) {
    lcd.setCursor(0, 1);
    lcd.print(servo4);
  } else {
    lcd.setCursor(0, 1);
    lcd.print("Fail");
  }

  delay(1000);
}

// Servo Spray - PWM Callback
void servoCBSpray(float servo5){
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Nozzle");
  lcd.setCursor(0, 1);
  lcd.print(servo5);

  servoSpray.write(servo5); //set servo angle, should be from 0-180  
 
  delay(1000);
}


//Move Servos

void moveServosCB(){
  lcd.setCursor(0, 0);
  lcd.print("MoveSV");
  lcd.setCursor(0, 1);
  lcd.print(index);
  
  servoCB1(angle1);
  servoCB2(angle2);
  servoCB3(angle3);
  servoCB4(angle4);
  
  delay(100);

}


// Main Callback
void getMessageCB(const std_msgs::Float32MultiArray& jointSetpoints){

  
  index = jointSetpoints.data[0];
  angle1 = jointSetpoints.data[1];
  angle2 = jointSetpoints.data[2];
  angle3 = jointSetpoints.data[3];
  angle4 = jointSetpoints.data[4];
  totalAngles = jointSetpoints.data[5];
  arduinoPresent = jointSetpoints.data[6];
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("MsgCB");
  lcd.setCursor(0, 1);
  //lcd.print(jointSetpoints.data_length);
  lcd.print(index);
  delay(500);

  moveServosCB();
  
}

//  Declare Subscriber
ros::Subscriber<std_msgs::Float32MultiArray> sub1("servo_setpoints", getMessageCB);


void setup() {
  //Initialize ROS Node, Subscriber and Publisher
  nh.initNode();
  
  //Setup LCD
  lcd.clear();
  lcd.begin(16, 2);
  lcd.print("Begin");
  
  //Subscriber
  nh.subscribe(sub1);
  
  //Setup Serial Servo
  Serial2.begin(115200);

  //Setup PWM Servo
  servoPWM.attach(9); //attach it to pin 9
  servoSpray.attach(8); //attach it to pin 8
  
  delay(1000);
}

void loop() {
  // put your main code here, to run repeatedly:
  nh.spinOnce();
  //delay(100);
}
