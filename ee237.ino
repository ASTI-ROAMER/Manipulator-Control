#include <BasicLinearAlgebra.h>
#include <math.h>
using namespace BLA;

//PIN ASSIGNMENTS
#define pi 3.14159

#define j2lock 40
#define j3lock 41

//#define j1cw 2 //towards limit switch
//#define j1ccw 3
#define j2up 4
#define j2down 5
#define j3up 6
#define j3down 7
//#define j4up 8
//#define j4down 9
//#define j5ccw 10
//#define j5cw 11 //towards limit switch
//#define gripclose 12
//#define gripopen 13

//#define limitj1 17
#define limitj2 39
#define limitj3 25
//#define limitj4 29
//#define limitj5 33

#define currPin1 A0
#define currPin2 A1

//Encoder Resolution per Joint
int encoderResJ1=200;
int encoderResJ2=200;
int encoderResJ3=200;
int encoderResJ4=96;
int encoderResJ5=96;

//Reduction Ratio per Joint
int reductionRatioJ1=100;
int reductionRatioJ2=165.3;
int reductionRatioJ3=110;
int reductionRatioJ4=180;
int reductionRatioJ5=110;

//Encoder 236 Version - Polling
int readEncoderAdd(int jointAngleInput, int encoderApin, int encoderBpin, int encoderZpin, int encoderResJ, int reductionRatioJ){
  int encoderAprev=LOW;
  int tickCount=0;
  int motorAngle=0;
  int movedAngle=0;
  while(1){
    int encoderA=digitalRead(encoderApin); //Encoder A - releases multiple pulses
    int encoderB=digitalRead(encoderBpin); //Encoder B - offset of A for determining direction
    int encoderZ=digitalRead(encoderZpin); //Encoder Z - high when 1 turn is completed 
    if((encoderAprev==LOW)&&(encoderA==HIGH)){
        tickCount=tickCount+1;
        motorAngle=tickCount*(360/encoderResJ); //example 200 pulses = 360 degrees (ng motor) for J1
        movedAngle=motorAngle*(1/reductionRatioJ); //example 100 degrees (ng motor) = 1 degree (ng joint) for J1
        
        return movedAngle; 
    }
  }
}


int readEncoderSub(int jointAngleInput, int encoderApin, int encoderBpin, int encoderZpin, int encoderResJ, int reductionRatioJ){
  int encoderAprev=LOW;
  int tickCount=0;
  int motorAngle=0;
  int movedAngle=0;
  while(1){
    int encoderA=digitalRead(encoderApin); //Encoder A - releases multiple pulses
    int encoderB=digitalRead(encoderBpin); //Encoder B - offset of A for determining direction
    int encoderZ=digitalRead(encoderZpin); //Encoder Z - high when 1 turn is completed 
    if((encoderAprev==LOW)&&(encoderA==HIGH)){
        tickCount=tickCount-1;
        motorAngle=tickCount*(360/encoderResJ); //example 200 pulses = 360 degrees (ng motor) for J1
        movedAngle=motorAngle*(1/reductionRatioJ); //example 100 degrees (ng motor) = 1 degree (ng joint) for J1
        return movedAngle; 
    }
  }
}

//Encoder 237 - Interrupt Based
enum PinAssignments {
  encoderPinA = 18,   // right
  encoderPinB = 19,   // left
};

volatile int encoderPosition = 0;  // a counter for the dial
volatile int lastPosition = 1;   // change management
static boolean rotating = false;    // debounce management
float jointAngle2 = 0;

//Interrupt service routine variables
boolean A_set = false;
boolean B_set = false;

void setup(){
  Serial.begin(2000000);

      Serial.print("Rad ");
      Serial.print(" ");
      Serial.print("Iter ");
      Serial.print(" ");
      Serial.print("PWM ");
      Serial.print(" ");
      Serial.print("Tau ");
      Serial.print(" ");
      Serial.println("Rad/s ");

  //Setup Pin Modes
  //>Locks:
  pinMode(40,OUTPUT); //j2lock 40
  pinMode(41,OUTPUT); //j3lock 41
  //>PWM:
  pinMode(4,OUTPUT); //j2up 4
  pinMode(5,OUTPUT); //j2down 5
  pinMode(6,OUTPUT); //j3up 6
  pinMode(7,OUTPUT); //j3down 7
  //Limit Switches:
  pinMode(2,INPUT); //limitj2 39
  pinMode(3,INPUT); //limitj3 25
  //Encoder Inputs:
  pinMode(18,INPUT); //Joint 2 A
  pinMode(19,INPUT); //Joint 2 B

  //Initialize PWM to zero
  analogWrite(4,0); //j2up
  analogWrite(5,0); //j2down
  analogWrite(6,0); //j3up
  analogWrite(7,0); //j3down
  digitalWrite(j2lock, LOW); //initialize joint 2 locks as locked
  digitalWrite(j2lock, HIGH); //initialize joint 2 locks as locked ---------------------UNLOCK
  digitalWrite(j3lock, LOW); //initialize joint 3 locks as locked
  delay(10000);

  //Interrupts
  pinMode(encoderPinA, INPUT); //PWM pin2 for J2
  pinMode(encoderPinB, INPUT); //PWM pin3 for J2
  //Turn on pullup resistors
  digitalWrite(encoderPinA, HIGH);
  digitalWrite(encoderPinB, HIGH);
  //Encoder pin on interrupt 0 (pin 2)
  attachInterrupt(5, readEncoderA, CHANGE); //INT2 --> Pin 18
  //Encoder pin on interrupt 1 (pin 3)
  attachInterrupt(4, readEncoderB, CHANGE); //INT2 --> Pin 19


  //Computation for [q1 q2 q1dot q2dot]' equivalently [x1 x2 x3 x4]'
  float tfinal = 10.0;
  float Iterations = 1000;//tfinal*100;
  float TimeIncrement = 0.01; //10^-2
  float Tau1 = 0;
  float Tau2 = 0;
  
  float err1 = 0;
  float err2 = 0;
  float errsum1 = 0;
  float errsum2 = 0;

  float currentRead1 = 0;
  float currentRead2 = 0;
  int j2_PWM = 0;
  int j3_PWM = 0;
  
  //Constants
  float g = 9.81; //m/s^2
  BLA::Matrix<1,4> g1 = {0, -g, 0, 0};

  float Kv11 = 40;//12
  float Kv12 = 0;
  float Kv21 = 0;
  float Kv22 = 40;//36

  float Kp11 = 400; //For critically damped condition Kv11
  float Kp12 = 0;
  float Kp21 = 0;
  float Kp22 = 400; //For critically damped condition Kp22

  BLA::Matrix<2,2> Kv = {Kv11, Kv12, Kv21, Kv22};
  BLA::Matrix<2,2> Kp = {Kp11, Kp12, Kp21, Kp22};

  float Tau1_max = 20.0; //breakeven tau2 = 19.15 and max motor torque is 23.8 
  float Tau2_max = 7.7; //breakeven tau2 = 5.421 and max motor torque is 15.4

  //Set Tau and Motor Constants
  float I_max = 2.1;//A
  float Ki1 = Tau1_max/2.1;
  float Ki2 = Tau2_max/2.1;

  //Link 1
  float m1 = 4; //kg
  float l1 = 0.250; //m

  ////Link 2
  float m2 = 3.6; //kg
  float l2 = 0.307; //m

  //Number of Joints
  int N = 2;

  //Declare Matrices
 
  BLA::Matrix<4,4> Qrot = { 0, -1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 };

  BLA::Matrix<4,4> I1 = { pow(l1,2)/4*m1, 0, 0, -l1/2*m1,  0, 0, 0, 0,  0, 0, 0, 0,  -l1/2*m1, 0, 0, m1 };

  BLA::Matrix<4,4> I2 = { pow(l2,2)/4*m2, 0, 0, -l2/2*m2,  0, 0, 0, 0,  0, 0, 0, 0,  -l2/2*m2, 0, 0, m2 };

  BLA::Matrix<2,1> Tau = {0.000 , 0.000};

  //Set time boundaries
  float t1 = 0.2*tfinal; //2s
  float t2 = 0.8*tfinal; //8s
  
  float tc = (t1/tfinal)*(Iterations*TimeIncrement); //2s
  float tm = (t2/tfinal)*(Iterations*TimeIncrement); //8s
  float tf = 1*(Iterations*TimeIncrement); //10s

  //Set angle setpoints
  float q1des_main = 45.0/180.0*pi;
  float q2des_main = 0.0/180.0*pi;
  //Serial.println(q1des_main); 
  //Serial.println(q2des_main); 

  //Set Max Velocity
  float q1dotdes_max = 2*q1des_main/(2*tf - tc - (tf-tm)); //1/2(b1+b2)*h
  float q2dotdes_max = 2*q2des_main/(2*tf - tc - (tf-tm));
  //Serial.println(q1dotdes_max); 
  //Serial.println(q2dotdes_max); 

  //Set Max Acceleration
  float q1dotdotdes_max = q1dotdes_max/tc;
  float q2dotdotdes_max = q2dotdes_max/tc;
  //Serial.println(q1dotdotdes_max); 
  //Serial.println(q2dotdotdes_max); 

  //Set Intersection Points
  float qc1 = 0.5*q1dotdotdes_max*pow(tc,2);
  float qm1 = q1des_main - 0.5*q1dotdotdes_max*pow((tf-tm),2);  
  
  float qc2 = 0.5*q2dotdotdes_max*pow(tc,2);
  float qm2 = q2des_main - 0.5*q2dotdotdes_max*pow((tf-tm),2);

  //Initialize Setpoints
  float q1des = 0; 
  float q2des = 0; 
  float q1 = 0.0;
  float q1dotdes = 0; 
  float q2dotdes = 0; 
  float q1dotdotdes = 0; 
  float q2dotdotdes = 0; 
  
  //Compute Matrices Dnk Dnkj and G -----------------------------------------------------
  float q1prev = 0;
  float q2prev = 0;
  for ( int o=0; o<Iterations; o++ ) {
    //Timestamp
    unsigned long Tstart = micros();

    //Read UART/Serial (Desired Setpoints)
    float tcurr = (o)*TimeIncrement;

    if (tcurr < tc){ //2s
      //acceleration
      q1dotdotdes = q1dotdotdes_max;
      q2dotdotdes = q2dotdotdes_max;
      
      //velocity
      q1dotdes = q1dotdotdes_max*tcurr;
      q2dotdes = q2dotdotdes_max*tcurr; 
      
      //angle
      q1des = 0.5*q1dotdotdes_max*pow(tcurr,2);
      q2des = 0.5*q2dotdotdes_max*pow(tcurr,2);
    }
    else if ((tcurr >= tc) && (tcurr <= tm)){ //between 2s and 8s
      //acceleration
      q1dotdotdes = 0;
      q2dotdotdes = 0;
      
      //velocity
      q1dotdes = q1dotdes_max;
      q2dotdes = q2dotdes_max;
      
      //angle
      q1des = (qm1-qc1)/(tm-tc)*(tcurr) - qc1;
      q2des = (qm2-qc2)/(tm-tc)*(tcurr) - qc2;
    }
    else{ //8s to 10s
      //acceleration
      q1dotdotdes = - q1dotdotdes_max;
      q2dotdotdes = - q2dotdotdes_max;
      
      //velocity
      q1dotdes = q1dotdotdes_max*(tf-tcurr); 
      q2dotdes = q2dotdotdes_max*(tf-tcurr); 
      
      //angle
      q1des = q1des_main - 0.5*q1dotdotdes_max*pow(tf-tcurr,2);
      q2des = q2des_main - 0.5*q2dotdotdes_max*pow(tf-tcurr,2);
    }
    
       

    //Prints read encoder pulses from interrupt
    cli();
    rotating = true;  // reset the debouncer
    if (lastPosition != encoderPosition) {
      //Serial.print("Index:-----------------");
      //Serial.println(encoderPosition, DEC);
      //Serial.println(jointAngle2, DEC);
      jointAngle2 = encoderPosition*(360.0/encoderResJ2)*(1.0/reductionRatioJ2)*(pi/180);
      q1 = jointAngle2;
      //Serial.println(q1);
      lastPosition = encoderPosition;
    }
    sei();
    
    float q2 = 0.0;//y0(1,2);
    //Serial.println(q1,6);
    float q1dot = (q1-q1prev)/TimeIncrement;//y0(1,3);
    float q2dot = (q2-q2prev)/TimeIncrement;//y0(1,4);
    
    //Compute error
    float err1 = q1des - q1;
    float err2 = q2des - q2;
  
    float err1dot = q1dotdes - q1dot;
    float err2dot = q2dotdes - q2dot;



    //Generate A01/A12 matrices
    BLA::Matrix<4,4> A01 = {cos(q1), -sin(q1), 0, l1*cos(q1) , sin(q1), cos(q1), 0, l1*sin(q1) , 0, 0, 1, 0 , 0, 0, 0, 1};
    BLA::Matrix<4,4> A12 = {cos(q2), -sin(q2), 0, l2*cos(q2) , sin(q2), cos(q2), 0, l2*sin(q2) , 0, 0, 1, 0 , 0, 0, 0, 1};
    BLA::Matrix<4,4> A01des = {cos(q1des), -sin(q1des), 0, l1*cos(q1des) , sin(q1des), cos(q1des), 0, l1*sin(q1des) , 0, 0, 1, 0 , 0, 0, 0, 1};
    BLA::Matrix<4,4> A12des = {cos(q2des), -sin(q2des), 0, l2*cos(q2des) , sin(q2des), cos(q2des), 0, l2*sin(q2des) , 0, 0, 1, 0 , 0, 0, 0, 1};
    BLA::Matrix<4,4> Identity = {1, 0, 0, 0 , 0, 1, 0, 0 , 0, 0, 1, 0 , 0, 0, 0, 1};
    float trace = 0;
    BLA::Matrix<4,4> TR = { 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0};


    //Compute Gravity Terms

      BLA::Matrix<1,2> G = {0,0}; //G = zeros(1,2);
      BLA::Matrix<2,1> Gd = {0 , 0}; //Gd = zeros(2,1);     
      BLA::Matrix<4,1> p1 = {-l1/2, 0, 0, 1}; //p1 = [-l1/2 0 0 1]';
      BLA::Matrix<4,1> p2 = {-l2/2, 0, 0, 1}; //p2 = [-l2/2 0 0 1]';
      BLA::Matrix<1,1> m1mat = {m1};
      BLA::Matrix<1,1> m2mat = {m2};
      BLA::Matrix<1,1> Gtemp = {0.0};   
      BLA::Matrix<1,1> Gdtemp = {0.0};  
      for (int n=0; n<=1; n++){ //n = 1:2
        //for i = n:N //i=n:N
          if (n==0){
            Gtemp = - m1mat*(g1*Qrot*(A01)*p1) - m2mat*(g1*Qrot*(A01)*(A12)*p2);
            G(0,n) = Gtemp(0,0);
            Gdtemp = - m1mat*(g1*Qrot*(A01des)*p1) - m2mat*(g1*Qrot*(A01des)*(A12des)*p2);
            Gd(n,0) = Gdtemp(0,0);
          }
          else {//n==2
            Gtemp = - m2mat*(g1*(A01)*Qrot*(A12)*p2);
            G(0,n) = Gtemp(0,0);
            Gdtemp = - m2mat*(g1*(A01des)*Qrot*(A12des)*p2);
            Gd(n,0) = Gdtemp(0,0);
          }
        //end
      }
    
    //Handoff of Variables
       
      float D11 = pow(l1,2)*m1/4 + pow(l2,2)*m2/4 + pow(l1,2)*m2 +l1*l2*m2*cos(q2); //Da(1,1)
      float D12 = pow(l2,2)*m2/4 + l1*l2*m2*cos(q2)/2;
      float D21 = D12;
      float D22 = pow(l2,2)*m2/4;


    
      float D111 = 0; //Db(1,1,1)
      float D112 = -l1*l2*m2*sin(q2)/2; //Db(1,1,2);
      float D121 = -l1*l2*m2*sin(q2)/2; //Db(1,2,1);
      float D122 = -l1*l2*m2*sin(q2)/2; //Db(2,1,1); //Db(1,2,2);
      float D211 = l1*l2*m2*sin(q2)/2; //Db(2,2,1);
      float D212 = 0; //Db(2,1,2);
      float D221 = 0;
      float D222 = 0; //Db(2,2,2);
      //Serial.println(D211,6);

      float G1 = G(0,0); //G(1,1)
      float G2 = G(0,1); //G(1,2)
      //float G1 = l2/2*m2*cos(q1+q2)*g + l1/2*m2*cos(q1)*g + l1/2*m1*cos(q1)*g; //G(1,1)
      //float G2 = l2/2*m2*cos(q1+q2)*g; //G(1,2)
      //Serial.println(Gd(0,1),6);
      
      BLA::Matrix<2,2> DA = {D11, D12 , D21, D22}; 
      BLA::Matrix<2,4> DB = {D111, D112, D121, D122 , D211, D212, D221, D222};
      
      float D11_des = pow(l1,2)*m1/4 + pow(l2,2)*m2/4 + pow(l1,2)*m2 +l1*l2*m2*cos(q2des); //Da(1,1)
      float D12_des = pow(l2,2)*m2/4 + l1*l2*m2*cos(q2des)/2;
      float D21_des = D12;
      float D22_des = pow(l2,2)*m2/4;
    
      float D111_des = 0; //Db(1,1,1)
      float D112_des = -l1*l2*m2*sin(q2des)/2; //Db(1,1,2);
      float D121_des = -l1*l2*m2*sin(q2des)/2; //Db(1,2,1);
      float D122_des = -l1*l2*m2*sin(q2des)/2; //Db(1,2,2);
      float D211_des = l1*l2*m2*sin(q2des)/2; //Db(2,1,1);
      float D212_des = 0; //Db(2,1,2);
      float D221_des = 0; //Db(2,2,1);
      float D222_des = 0; //Db(2,2,2);

      //float G1des = l2/2*m2*cos(q1des+q2des)*g + l1/2*m2*cos(q1des)*g + l1/2*m1*cos(q1des)*g; //G(1,1)
      //float G2des = l2/2*m2*cos(q1des+q2des)*g; //G(1,2)

      //BLA::Matrix<2,1> Gd = {G1des, G2des};
      
      BLA::Matrix<2,2> DA_des = {D11_des, D12_des , D21_des, D22_des}; 
      BLA::Matrix<2,4> DB_des = {D111_des, D112_des, D121_des, D122_des , D211_des, D212_des, D221_des, D222_des};
      
      BLA::Matrix<2,1> qdotdotDes = {q1dotdotdes , q2dotdotdes};
      BLA::Matrix<4,1> qdotDes = {q1dotdes*q1dotdes , q1dotdes*q2dotdes , q2dotdes*q1dotdes , q2dotdes*q2dotdes};

      //> store previous angle
      q1prev = q1;
      q2prev = q2;
      
      //> compute output torque
      BLA::Matrix<2,1> err = {err1 , err2};
      BLA::Matrix<2,1> errdot = {err1dot , err2dot};
    
      Tau = (Gd + DA_des*qdotdotDes + DB_des*qdotDes) + DA*(Kv*(errdot)+Kp*(err));
  
      float Tau1 = Tau(0,0);
      float Tau2 = Tau(1,0);

      //> torque limiting
      
      if (Tau1 >= Tau1_max){
        Tau1 = Tau1_max;
      }
      else if (Tau1 < 0){
        Tau1 = 0;
      }
      else {
        Tau1 = Tau1;
      }
      
      if (Tau2 >= Tau2_max){
        Tau2 = Tau2_max;
      }
      else if (Tau2 < 0){
        Tau2 = 0;
      }
      else {
        Tau2 = Tau2;
      } 
      

      //Send data to pwm
      currentRead1 = (float)analogRead(currPin1)/1023.0*5.0;
      //currentRead1 = currReadSimu1(0,o);
      currentRead2 = (float)analogRead(currPin2)/1023.0*5.0;
      //Serial.println(currentRead1);
      //Serial.println(currentRead2);
      float currentDesired1 = Tau1/Ki1;//Tau1/Ki1;
      float currentDesired2 = Tau2/Ki2;

      //Joint2

      j2_PWM = ((Tau1/11.03)*255.0)/2.1;

            if (j2_PWM <= 0.65*(250.0)){
              analogWrite(j2up,j2_PWM);
              //Serial.print(j2_PWM);
              //Serial.println("down");
            }
            else{
              //Serial.print(j2_PWM);
              //Serial.println("up-current overload");
              analogWrite(j2up,0.65*(250.0));
            }
      
      /*
      if(currentRead1>currentDesired1){
        while(currentRead1>currentDesired1){
          currentRead1 = (float)analogRead(currPin1)/1023.0*5.0;
          if(currentRead1>currentDesired1){
            j2_PWM--;
            if (j2_PWM <= 0.5*(1023)){
              analogWrite(j2up,j2_PWM);
              //Serial.println(j2_PWM);
              //Serial.println("down");
            }
            else{
              Serial.println(j2_PWM);
              Serial.println("up-current overload");
              analogWrite(j2up,0.5*(1023));
            }
          }
          else{
            break;
          }
        }
      }
      else{//(currentRead1<currentDesired1){
        while(currentRead1<currentDesired1){
          if(currentRead1<currentDesired1){
            currentRead1 = (float)analogRead(currPin1)/1023.0*5.0;
            j2_PWM++;
            if (j2_PWM <= 0.5*(1023)){
              analogWrite(j2up,j2_PWM);
              //Serial.println(j2_PWM);
              //Serial.println("up");
            }
            else{
              Serial.println(j2_PWM);
              Serial.println("up-current overload");
              analogWrite(j2up,0.5*(1023));
            }
          }
          else{
            break;
          }
        }
      }
      
      //Joint3
      if(currentRead2>currentDesired2){
        j3_PWM--;
        //analogWrite(j3up,j3_PWM);
        //Serial.println(j3_PWM);
      }
      else{//(currentRead2<currentDesired2){
        j3_PWM++;
        //analogWrite(j3down,j3_PWM);
        //Serial.println(j3_PWM);
      }
      */
     
      //Timestamp - get execution time
      unsigned long Tend = micros();
      unsigned long dT = Tend - Tstart;
      //Serial.print("Rad ");
      Serial.print(q1,6);
      Serial.print(",");
      //Serial.print("Iter ");
      Serial.print(o);
      Serial.print(",");
      //Serial.print("PWM ");
      Serial.print(j2_PWM);
      Serial.print(",");
      //Serial.print("Tau ");
      Serial.print(Tau1);
      Serial.print(",");
      Serial.println(q1dot);
      
      
      //Serial.print("Iter");
      //Serial.println(o);
      //Serial.println(dT);
      //Serial.println(G1);
      //Serial.println(G1des);
      //Serial.println(Tau1);
      //Serial.println(Tau2);
      //Serial.println(q1dot);
      //Serial.println(q2dot);
  }
  Serial.println("Done");
  Serial.print("Encoder Pulses ");
  Serial.println(encoderPosition);
  analogWrite(j2up,0); //Turn off pwm
  digitalWrite(j2lock, LOW); //Initialize joint 2 locks as locked
}

void loop(){
  

}

// Interrupt on change A 
void readEncoderA() {
  // debounce
  if ( rotating ) delayMicroseconds(500); //1ms  // wait a little until the bouncing is done

  // Test for change
  if ( digitalRead(encoderPinA) != A_set ) { // debounce once more
    A_set = !A_set;

    // Counter
    if ( A_set && !B_set )
      encoderPosition += 1;

    rotating = false;  // no more debouncing until loop() hits again
  }
}

// Interrupt on change B
void readEncoderB() {
  if ( rotating ) delayMicroseconds(500); //1ms
  if ( digitalRead(encoderPinB) != B_set ) {
    B_set = !B_set;
    //  counter
    if ( B_set && !A_set )
      encoderPosition -= 1;

    rotating = false;
  }
}

