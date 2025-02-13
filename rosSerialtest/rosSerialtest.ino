/*
 * rosserial Servo Control Example
 *
 * This sketch demonstrates the control of hobby R/C servos
 * using ROS and the arduiono
 * 
 * For the full tutorial write up, visit
 * www.ros.org/wiki/rosserial_arduino_demos
 *
 * For more information on the Arduino Servo Library
 * Checkout :
 * http://www.arduino.cc/en/Reference/Servo
 */

#if (ARDUINO >= 100)
 #include <Arduino.h>
#else
 #include <WProgram.h>
#endif

#include <Servo.h> 
#include <ros.h>
#include <std_msgs/UInt16.h>
#include "serial_servo.h"

//Initialize LCD
#include <LiquidCrystal.h>


ros::NodeHandle  nh;

Servo servoPWM;
SerialServo servoSerial(Serial2); //Pin 16,17

//Declare LCD Objects
const int RS = 12, EN = 11, D4 = 5, D5 = 4, D6 = 3, D7 = 2;
LiquidCrystal lcd(RS, EN, D4, D5, D6, D7);

void servo_cb1( const std_msgs::UInt16& cmd_msg){
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Servo1");

  float angle = cmd_msg.data;
  
  lcd.setCursor(0, 1);
  lcd.print(angle);
  
  servoPWM.write(cmd_msg.data); //set servo angle, should be from 0-180  
  digitalWrite(13, HIGH-digitalRead(13));  //toggle led  
}

void servo_cb2( const std_msgs::UInt16& cmd_msg){
  
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Servo2");


    
    uint8_t servo_id = 3;  // Set Servo ID

    //Send control instructions to the specified servo.
    float target_angle_1 = cmd_msg.data;  // Set target angle 0-270
    uint16_t move_time_1 = 1000;  // Complete turn within X time 0-3000
    servoSerial.move_servo_immediate(servo_id, target_angle_1, move_time_1);

    //Print current servo status
    lcd.setCursor(0, 1);
    lcd.print(target_angle_1);

    //Get the preset angle and time of the servo.
    float current_angle_2 = 0.0f;
    uint16_t current_time_2 = 0;
    t_FuncRet result_get_2 = servoSerial.get_servo_move_immediate(servo_id, &current_angle_2, &current_time_2);
    if (result_get_2 == Operation_Success) {
        //Serial.print("Current angle_2: ");
        //Serial.println(current_angle_2);
        //Serial.print("Current time_2: ");
        //Serial.println(current_time_2);
    } else {
        //Serial.println("Failed to get servo move details.");
    }
    delay(1000);
}

ros::Subscriber<std_msgs::UInt16> sub1("servo1", servo_cb1);
ros::Subscriber<std_msgs::UInt16> sub2("servo2", servo_cb2);

void setup(){

  //Setup LED
  pinMode(13, OUTPUT);
  
  //Initialize ROS Nodes
  nh.initNode();
  nh.subscribe(sub1);
  nh.subscribe(sub2);

  //Setup LCD
  lcd.clear();
  lcd.begin(16, 2);
  lcd.print("Begin");

  //Setup Serial Servo
  Serial2.begin(115200);

  //Setup PWM Servo
  servoPWM.attach(9); //attach it to pin 9
}

void loop(){
  nh.spinOnce();
  delay(1);
}
