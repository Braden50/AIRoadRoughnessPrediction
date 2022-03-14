import cv2
import smbus
import time


# See other file for initialization docs
bus = smbus.SMBus(1)
bus.write_byte_data(0x53, 0x2C, 0x0A)
bus.write_byte_data(0x53, 0x2D, 0x08)
bus.write_byte_data(0x53, 0x31, 0x08)


def main():
    videoCaptureObject = cv2.VideoCapture(0)
    numLoops = 30
    timeIncrement = 0.1
    
    # Values accelerometer outputs when still
    x_offset, y_offset, z_offset = calibration(20, 0.2)

    curLoop = 1
    with open('accData.csv', 'w') as f:
        while(curLoop <= numLoops):
            print(f"Loop: {curLoop}")
            time.sleep(timeIncrement)
            while(1):  # ensures camera is ready
                if videoCaptureObject.isOpened():
                    break
                time.sleep(0.1)
                print("waiting to be opened")
            timestamp = time.time()
            x, y, z = collect_acc_data(x_offset, y_offset, z_offset)
            #print("HERE:", x, y, z)
            f.write(f"{timestamp},{x},{y},{z}\n")
            file_name = "pictures/" + str(timestamp).replace(".", "_") + ".jpg"
            
            take_picture(videoCaptureObject, file_name)
            curLoop += 1



def calibration(num_loops=20, time_increment=0.2):
    print("Calibrating...")
    x_offset, y_offset, z_offset = 0, 0, 0
    x_sum, y_sum, z_sum = 0, 0, 0
    for _ in range(num_loops):
        x, y, z = collect_acc_data(0, 0, 0)
        #print(x, y, z)
        x_sum += x
        y_sum += y
        z_sum += z
        time.sleep(time_increment)
    x_offset = int(x_sum / num_loops)
    y_offset = int(y_sum / num_loops)
    z_offset = int(z_sum / num_loops)
    print("Calibration done.")
    return x_offset, y_offset, z_offset


def take_picture(videoCaptureObject, file_name):
    
    ret, frame = videoCaptureObject.read()

    cv2.imwrite(file_name, frame)
    #videoCaptureObject.release()
    cv2.destroyAllWindows()


def collect_acc_data(x_offset=0, y_offset=0, z_offset=0):
    # X Acceleration
    data0 = bus.read_byte_data(0x53, 0x32)
    data1 = bus.read_byte_data(0x53, 0x33)

    xAccl = ((data1 & 0x03) * 256) + data0
    #print(f"Initial x: {xAccl}")
    xAccl -= x_offset
    if xAccl > 511:
        xAccl -= 1024

    # Y Acceleration
    data0 = bus.read_byte_data(0x53, 0x34)
    data1 = bus.read_byte_data(0x53, 0x35)

    yAccl = ((data1 & 0x03) * 256) + data0
    #print(f"Initial y: {yAccl}")
    yAccl -= y_offset
    if yAccl > 511:
        yAccl -= 1024

    # Z Acceleration
    data0 = bus.read_byte_data(0x53, 0x36)
    data1 = bus.read_byte_data(0x53, 0x37)

    zAccl = ((data1 & 0x03) * 256) + data0
    #print(f"Initial z: {zAccl}")
    zAccl -= z_offset
    if zAccl > 511:
        zAccl -= 1024

    return xAccl, yAccl, zAccl


if __name__=="__main__":
    main()


