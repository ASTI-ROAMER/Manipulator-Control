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
#include <std_msgs/UInt16MultiArray.h>

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

// Servo 1 - PWM Callback
void servoCB1(float servo1){
  
  lcd.setCursor(0, 0);
  lcd.print("Servo1");
  
  servoPWM.write(servo1); //set servo angle, should be from 0-180  
  digitalWrite(13, HIGH-digitalRead(13));  //toggle led  
  delay(2000);
}

//Servo 2 - Serial Callback

void servoCB2(float servo2){


  lcd.setCursor(0, 0);
  lcd.print("Servo2");

  uint8_t servoID = 2;  // Set Servo ID

  //Send control instructions to the specified servo.
  float targetAngle1 = servo2;  // Set target angle 0-270
  uint16_t moveTime1 = 1000;  // Complete turn within X time 0-3000
  servoSerial.move_servo_immediate(servoID, targetAngle1, moveTime1);

  //Print current servo status


  //Get the preset angle and time of the servo.
  float currentAngle1 = 0.0f;
  uint16_t currentTime1 = 0;
  t_FuncRet resultGet1 = servoSerial.get_servo_move_immediate(servoID, &currentAngle1, &currentTime1);
  if (resultGet1 == Operation_Success) {
      //Serial.print("Current angle_2: ");
      //Serial.println(current_angle_2);
      //Serial.print("Current time_2: ");
      //Serial.println(current_time_2);
  } else {
      //Serial.println("Failed to get servo move details.");
  }
  delay(2000);
}


// Declare Publisher
std_msgs::UInt16 indexFlag;
ros::Publisher pub1("arm_index_flag", &indexFlag); 


float index = 0;
float angle1 = 0;
float angle2 = 0;
float totalAngles = 0;
int startFlag = 0;
float arduinoPresent = 0.0;

// Main Callback
void getMessageCB(const std_msgs::UInt16MultiArray& jointSetpoints){

  
  index = jointSetpoints.data[0];
  angle1 = jointSetpoints.data[1];
  angle2 = jointSetpoints.data[2];
  totalAngles = jointSetpoints.data[3];
  arduinoPresent = jointSetpoints.data[4];
  lcd.setCursor(0, 0);
  lcd.print("Callback");
  lcd.setCursor(0, 1);
  lcd.print(arduinoPresent);
  delay(1000);
}

void moveServosCB(){
  lcd.setCursor(0, 0);
  lcd.print("MoveSV");
  lcd.setCursor(0, 1);
  lcd.print(arduinoPresent);
  if(arduinoPresent==1.1){
    if(startFlag==0){
        if(index <= totalAngles){ 
          indexFlag.data = index;  
          servoCB1(angle1);
          servoCB2(angle2);
          indexFlag.data = 1;
          pub1.publish(&indexFlag);
        }
        startFlag = 1;
    }
    else{      
      if(index <= totalAngles){ 
          indexFlag.data = index;  
          servoCB1(angle1);
          servoCB2(angle2);
          indexFlag.data = index;
          pub1.publish(&indexFlag);
        }
        else{
          //dunno
        }
      
    }
  }
}
//  Declare Subscriber
ros::Subscriber<std_msgs::UInt16MultiArray> sub1("servo_setpoints", getMessageCB);

void setup(){

  //Setup Servo PWM
  pinMode(13, OUTPUT);

  //Setup LCD
  lcd.begin(16, 2);
  lcd.print("Start");
  
  //Initialize ROS Node, Subscriber and Publisher
  nh.initNode();
  nh.subscribe(sub1);
  nh.advertise(pub1);

  //Setup Serial Servo
  Serial2.begin(115200);

  //Setup PWM Servo
  servoPWM.attach(9); //attach it to pin 9

  delay(1000);
}

void loop(){
  moveServosCB();
  
  nh.spinOnce();
}
