#include <Servo.h>

Servo servoX, servoY;
// twelve servo objects can be created on most boards
String input;
int laserPin = 12;
int laserState = LOW;

void setup() {
  servoX.attach(4);
  servoY.attach(2);
  pinMode(laserPin, OUTPUT);
  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);
  Serial.setTimeout(10);
  servoX.write(90);
  servoY.write(90);
}

void loop() {
  //digitalWrite(laserPin, laserState);
  ;
}

void serialEvent() {
  String serialData = Serial.readString();
  // t for toggle
  Serial.println("received data: "+serialData);
  if(serialData[0]=='t') {
    if(laserState==LOW) laserState = HIGH;
    else laserState = LOW;
    
    digitalWrite(laserPin, laserState);
  }
  else {
    int x = getX(serialData);
    int y = getY(serialData);
    Serial.println(x,y);
    servoX.write(x);
    // problem of servo getting lifted due to wire
    if(y>50) servoY.write(y);
  }
}

int getX(String data){
  data.remove(data.indexOf(" "));
  return data.toInt();
}

int getY(String data) {
  data.remove(0, data.indexOf(' ')+1);
  return data.toInt();
}
