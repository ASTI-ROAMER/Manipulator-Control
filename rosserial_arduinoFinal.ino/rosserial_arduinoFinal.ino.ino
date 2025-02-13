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



// Servo 1 - PWM Callback
void servoCB1(float servo1){
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo1");
  lcd.setCursor(0, 1);
  lcd.print(servo1);
  
  delay(1000);
}

//Servo 2 - Serial Callback

void servoCB2(float servo2){

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo2");
  lcd.setCursor(0, 1);
  lcd.print(servo2);
  
  delay(1000);
}

void moveServosCB(){
  lcd.setCursor(0, 0);
  lcd.print("MoveSV");
  lcd.setCursor(0, 1);
  lcd.print(index);
  
  servoCB1(angle1);
  servoCB2(angle2);
  
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
  
  delay(1000);
}

void loop() {
  // put your main code here, to run repeatedly:
  nh.spinOnce();
  //delay(100);
}
