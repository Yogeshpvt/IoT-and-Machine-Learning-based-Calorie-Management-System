import socket

server=socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
server.bind(('B8:27:EB:93:70:AE', 4))

server.listen(1)

client, addr=server.accept()

import RPi.GPIO as gpio
import time
import math
import smbus

PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47


def MPU_Init():
	bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
	bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
	bus.write_byte_data(Device_Address, CONFIG, 0)
	bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
	bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
        value = ((high << 8) | low)
        if(value > 32768):
	        value = value - 65536
        return value


bus = smbus.SMBus(1)
Device_Address = 0x68

MPU_Init()

gpio.setmode(gpio.BCM)
gpio.setwarnings(False)

timeperiod = 5 # in s
radius = 0.4 # in m

noofmagnets = 2

data=client.recv(1024)

cycle_w = 8 # in kg
person_w = int(data.decode('utf-8')) #in kg (trial value)
totw = cycle_w + person_w #in kg

totcalsburned = 0
totcounter = 0
counter = 0
counter2 = 0
prev = 0

hall1pin =18
hall2pin = 23

gpio.setup(hall1pin, gpio.IN)
gpio.setup(hall2pin, gpio.IN)

start = time.time()
ctime = time.time()
while True:
    try:
        if(gpio.input(hall1pin) == False):
            if(prev == 0):
                prev = 1
                counter = counter + 1
                counter2 + counter2 + 1
                totcounter = totcounter + 1
                print("count: ", counter)
                prev = 1
        else:
            prev = 0

        if(time.time() >= start + timeperiod):
            counter = counter/noofmagnets
            rpm = counter/(timeperiod/60)
            rpm2 = counter2/(timeperiod/60)
            circumference = 2*math.pi*radius
            totdist = totcounter*circumference/1000 #in km
            dist = counter*circumference
            speed = rpm*circumference/60 * (60*60)/1000  #in km/h

            acc_y = read_raw_data(ACCEL_YOUT_H)
            acc_z = read_raw_data(ACCEL_ZOUT_H)
            tanx = acc_y/(acc_z+0.00001)
            x = math.degrees(math.atan(tanx)) #in degrees

            met = 0
            if(speed >=10 and speed <13):
                met = 2
            elif(speed >=13 and speed <16):
                met = 4
            elif(speed >=16 and speed <18):
                met = 6
            elif(speed >=18 and speed <22):
                met = 8
            elif(speed >=22 and speed <25):
                met = 10
            elif(speed >=25 and speed <28):
                met = 12
            elif(speed >=28 and speed <32):
                met = 14
            elif(speed >=32):
                met = 16

            totcalsburned = totcalsburned + ((met*3.5*totw)(timeperiod/60)/200)(1+tanx) #in cal
            print (f"Wheel RPM: %d, Total distance: %.2f m, Velocity: %.2f km/h, angle of incline: %.2f,  total calories burned: %.2f cal" % (rpm, totdist, speed, x, totcalsburned))
        
            counter = 0
            counter2 = 0
            start = time.time()

            #    message=input("Enter message: ")
            message = str(round(rpm)) + " " + str(round(totdist,2)) + " " + str(round(speed,2)) + " " + str(round(x,2)) + " " + str(round(totcalsburned,2))
            client.send(message.encode('utf-8'))
    except:
        client.close()
        server.close()
        print("succesfully closed")
        break