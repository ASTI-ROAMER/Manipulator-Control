//Initialize Arduino

#if (ARDUINO >= 100)
 #include <Arduino.h>
#else
 #include <WProgram.h>
#endif

//Initialize ROS libraries
#include <ros.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32MultiArray.h>

//Initialize LCD
#include <LiquidCrystal.h>



//Initialize Node Handle
ros::NodeHandle  nh;

//Declare LCD Objects
const int RS = 12, EN = 11, D4 = 5, D5 = 4, D6 = 3, D7 = 2;
LiquidCrystal lcd(RS, EN, D4, D5, D6, D7);

float index = 0;
float angle1 = 0;
float angle2 = 0;
float totalAngles = 0;
float arduinoPresent = 0.0;

// Main Callback
void getMessageCB(const std_msgs::Float32MultiArray& jointSetpoints){

  
  index = jointSetpoints.data[0];
  angle1 = jointSetpoints.data[1];
  angle2 = jointSetpoints.data[2];
  totalAngles = jointSetpoints.data[3];
  arduinoPresent = jointSetpoints.data[4];
  lcd.setCursor(0, 0);
  lcd.print("Start");
  lcd.setCursor(0, 1);
  //lcd.print(jointSetpoints.data_length);
  lcd.print(jointSetpoints.data[0]);
  delay(1000);
}

//  Declare Subscriber
ros::Subscriber<std_msgs::Float32MultiArray> sub1("servo_setpoints", getMessageCB);


void setup() {
  //Initialize ROS Node, Subscriber and Publisher
  nh.initNode();
  nh.subscribe(sub1);
  //Setup LCD
  lcd.begin(16, 2);
  lcd.print("Begin");
  
  delay(1000);
}

void loop() {
  // put your main code here, to run repeatedly:
  nh.spinOnce();
}
