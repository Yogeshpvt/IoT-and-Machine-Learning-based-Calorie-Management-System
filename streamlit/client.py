import socket
import csv

client=socket.socket(socket.AF_BLUETOOTH,socket.SOCK_STREAM,socket.BTPROTO_RFCOMM)
client.connect(('B8:27:EB:93:70:AE',4))

person_w=60 # read weight of person from db
client.send(str(person_w).encode('utf-8'))

count = 0
speedsum = 0
try:
    print("Time\tDistance\tSpeed\tCalories")
    while True:
#        message="sent"
#        client.send(message.encode('utf-8'))
        
        data=client.recv(1024)
        if not data:
            break
        ctime, totdist, speed, calsburned, num = data.decode('utf-8').split()
        count += 1
        speedsum += float(speed)
        speed = speedsum/count
        print("Received")
        if float(num) == 0:
            speed = speedsum/count
            
            print(f"{ctime}\t{totdist}\t{speed}\t{calsburned}")
            with open('cycle_data.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([ctime, totdist, speed, calsburned])

except OSError as e:
    pass

client.close()