from picam2multithread import Picam2MultiThread
from motordriver_lite import MotorDriver
from ei_monitor import EncoderImuMonitor
import time
import cv2
import numpy as np
import RPi.GPIO as gpio

debug = 0

# initiate and start camera
picam2mt = Picam2MultiThread(main_size=(820,616),lores_size=(205,154),
                             framerate=20.0,verbose=False,
                             rec_vid=1,
                             vid_keepframe_itv=1,
                             vid_framerate=20.0)
picam2mt.start()
time.sleep(1.0)

# initiate motor driver
# first define PID controller coefficients
Kp_enc,Ki_enc,Kd_enc = [1.0, 0.0, 0.0]
Kp_yaw,Ki_yaw,Kd_yaw = [500.0, 25.0, 0.0]
Kp_turn_pix,Ki_turn_pix,Kd_turn_pix = [0.3, 0.0, 0.4]
Kp_turn_rad,Ki_turn_rad,Kd_turn_rad = [80.0, 0.0, 40.0]
Kp_fwd,Ki_fwd,Kd_fwd = [6.0, 0.0, 2.0]
Kp_grab,Ki_grab,Kd_grab = [0.4, 0.0, 0.03]

# create motor driver object
md = MotorDriver(K_enc=(Kp_enc,Ki_enc,Kd_enc),
                 K_yaw=(Kp_yaw,Ki_yaw,Kd_yaw),
                 K_turn_pix=(Kp_turn_pix,Ki_turn_pix,Kd_turn_pix),
                 K_turn_rad=(Kp_turn_rad,Ki_turn_rad,Kd_turn_rad),
                 K_fwd=(Kp_fwd,Ki_fwd,Kd_fwd),
                 K_grab=(Kp_grab,Ki_grab,Kd_grab),
                 speed_sum_max=100,
                 speed_sum_min=30)

# initiate encoder and imu monitor
monitor = EncoderImuMonitor(log_data=0)
monitor.start()

while True:
    start = input('start now?')
    if start=='y':
        print('Let\'s roll!')
        break

K = np.array([[664.68,0,413.48],
              [0,664.87,322.61],
              [0,0,1]])

TTC_threshold = 45
md.optical_flow_control(picam2mt,
                        monitor,
                        calibration_mat=K,
                        TTC_threshold=TTC_threshold,
                        flow_method=3,
                        debug=debug,
                        max_iter=400,
                        e_threshold=0.3,
                        ttc_buffer_sz=5)

# stop camera
picam2mt.stop()

# stop monitor
monitor.stop.value = 1
time.sleep(0.5)
monitor.process.terminate()
monitor.process.join()

# gpio clean up
gpio.cleanup()