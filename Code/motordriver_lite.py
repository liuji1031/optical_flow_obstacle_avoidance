import numpy as np
import RPi.GPIO as gpio
import matplotlib.pyplot as plt
from threading import Thread
import time
import cv2
from datetime import datetime
from distancesensor import DistanceSensor
from optflow_util import *

class MotorDriver:
    
    def __init__(self,
                 K_enc=[0,0,0],
                 K_yaw=[0,0,0],
                 K_turn_pix=[0,0,0],
                 K_turn_rad=[0,0,0],
                 K_fwd=[0,0,0],
                 K_grab=[0,0,0],
                 speed_sum_max=80,
                 speed_sum_min=0,
                 save_position_data=0):
        freq = 100
        gpio.setmode(gpio.BOARD)
        gpio.setup(31,gpio.OUT) # IN1
        gpio.setup(33,gpio.OUT) # IN2
        gpio.setup(35,gpio.OUT) # IN3
        gpio.setup(37,gpio.OUT) # IN4
        
        self.pwm1 = gpio.PWM(31,freq)
        self.pwm2 = gpio.PWM(33,freq)
        self.pwm3 = gpio.PWM(35,freq)
        self.pwm4 = gpio.PWM(37,freq)
        
        self.pwm1.start(0.)
        self.pwm2.start(0.)
        self.pwm3.start(0.)
        self.pwm4.start(0.)
        
        self.left_pin = 7
        self.right_pin = 12
        # self.pulse_per_rev = 480 # remember the gear ratio

        gpio.setup(self.left_pin,gpio.IN,pull_up_down=gpio.PUD_UP)
        gpio.setup(self.right_pin,gpio.IN,pull_up_down=gpio.PUD_UP)
        
        self.speed_sum_max = speed_sum_max
        self.speed_sum_min = speed_sum_min
        self.controller_hz = 20
        self.ctrl_loop_dur = 1./self.controller_hz
        self.count_per_rev = 40
        self.wheel_d = 6.5
        
        self.camera_center = 413 # optical center in pixel 
        
        self.Kp_enc,self.Ki_enc,self.Kd_enc = K_enc
        self.Kp_yaw,self.Ki_yaw,self.Kd_yaw = K_yaw
        self.Kp_turn_pix,self.Ki_turn_pix,self.Kd_turn_pix = K_turn_pix
        
        self.max_turn_speed = 40
        self.desired_center_x = int((413+19)/4)
        self.desired_center_y = 145
        self.tol_pixel = 7
        self.tol_pixel_coarse = 15 # 1/3 of field of view
        
        self.tol_rad = 1/180*np.pi

        self.min_turn_pwm = [32,32]
        self.max_turn_pwm = [40,45]
        
        self.Kp_turn_rad,self.Ki_turn_rad,self.Kd_turn_rad = K_turn_rad
    
        self.Kp_fwd,self.Ki_fwd,self.Kd_fwd = K_fwd
        self.tol_dist = 2
        
        self.Kp_grab,self.Ki_grab,self.Kd_grab = K_grab

        self.curr_x = 0
        self.curr_y = 0
        self.save_position_data = save_position_data
        if self.save_position_data == 1:
            self.pos_array = []
            self.append_curr_pos()
        
        self.dist_sensor = DistanceSensor()
        self.pause_time = 0.5

    def append_curr_pos(self):
        self.pos_array.append([time.time(), self.curr_x, self.curr_y])

    def gameover(self):
        # set all pins low      
        self.pwm1.ChangeDutyCycle(0.)
        self.pwm2.ChangeDutyCycle(0.)
        self.pwm3.ChangeDutyCycle(0.)
        self.pwm4.ChangeDutyCycle(0.)
    
    def set_speed_left(self,speed=100.0):
        speed_abs = abs(speed)
        speed_abs = min(speed_abs,100.)
        speed_abs = max(speed_abs,0.)
        if speed>0:
            self.pwm1.ChangeDutyCycle(speed_abs)
            self.pwm2.ChangeDutyCycle(0.)
        else:
            self.pwm1.ChangeDutyCycle(0.)
            self.pwm2.ChangeDutyCycle(speed_abs)
    
    def set_speed_right(self,speed=100.0):     
        speed_abs = abs(speed)
        speed_abs = min(speed_abs,100.)
        speed_abs = max(speed_abs,0.)
        if speed>0:
            self.pwm3.ChangeDutyCycle(0.)
            self.pwm4.ChangeDutyCycle(speed_abs)
        else:
            self.pwm3.ChangeDutyCycle(speed_abs)
            self.pwm4.ChangeDutyCycle(0.)
    
    def calculate_rot_matrix_from_rad(self,rad):
        c = np.cos(rad)
        s = np.sin(rad)
        rot_mat = np.array([[c,-s],[s,c]])
        return rot_mat
        
    def convert_rad_in_ref(self,rot_mat, yaw):
        yaw_vec = np.array([[np.cos(yaw)],[np.sin(yaw)]])
        yaw_vec_ref = rot_mat.T.dot(yaw_vec)
        rad_in_ref = np.arctan2(yaw_vec_ref[1],yaw_vec_ref[0]).item()
        return rad_in_ref
    
    def ensure_continuity(self, curr_yaw, last_yaw):
        if last_yaw is None:
            return curr_yaw
        if np.abs(curr_yaw-last_yaw) < np.pi: # small change
            return curr_yaw
        test_yaw = np.array([-np.pi*2,0,np.pi*2])+curr_yaw
        imin = np.argmin(np.power(test_yaw-last_yaw,2))
        return test_yaw[imin]

    def report_current_pose(self,monitor):
        yaw = monitor.imu_reading.value
        yaw_degree = yaw/np.pi*180
        if yaw_degree<0:
            yaw_degree+=360
        print('Current location ({:.2f},{:.2f}), heading angle {:.2f}'.format(self.curr_x,self.curr_y,yaw_degree))

    def constrain_turn_P(self,P,ind):
        sn = np.sign(P)
        absP = abs(P)
        absP = max(self.min_turn_pwm[ind],absP)
        absP = min(self.max_turn_pwm[ind],absP)
        return absP*sn

    def plot_result_hsv(self, frame, flow, mag_max=20):
        hsv = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[:,:,0] = ang * 180 / np.pi / 2
        hsv[:,:,1] = 255
        hsv[:,:,2] = (np.minimum(mag,mag_max)/mag_max*255).astype(np.uint8) # cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return out
    
    def compute_hemi_optflow_mag(self, flow,ds_factor=2):
        mag = np.linalg.norm(flow, axis=2)
        mid = int(self.camera_center/ds_factor)
        optflow_mag_l = np.sum(mag[:,:mid])/mag.size
        optflow_mag_r = np.sum(mag[:,mid:])/mag.size
        return optflow_mag_l, optflow_mag_r

    def get_now_str(self):
        now = datetime.now() # current date and time
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    
    def optical_flow_control(self,
                             picam2mt,
                             monitor,
                             calibration_mat=None,
                             TTC_threshold=18,
                             compute_every_n_frame=1,
                             flow_method=0,
                             debug=0,
                             max_iter=100,
                             e_threshold=0.2,
                             ttc_buffer_sz=3):
        last_timestamp = picam2mt.frame_timestamp
        
        img_downsample_factor=5
        calibration_mat[0,0] /= img_downsample_factor
        calibration_mat[0,2] /= img_downsample_factor
        calibration_mat[1,1] /= img_downsample_factor
        calibration_mat[1,2] /= img_downsample_factor
        print('new K:\n',calibration_mat)
        K_inv = np.linalg.inv(calibration_mat)
        
        Kp = 125
        Kd = 0
        
        frame_buffer = []
        e_lat_last = None
        
        t_start = time.time_ns()
        loop_timestamp_ms = []
        dist_measure = []
        ctrl_output = []
        optflow_mag = []
        ttc_all = []
        turn_degree_all = []
        ttc_buffer = []
        n_iter = 0
        
        optical_flow_params = dict(winSize=(11, 11),maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03),)
        
        while True:
            curr_timestamp = picam2mt.frame_timestamp
            if curr_timestamp < last_timestamp+100:
                continue
            print('iteration',n_iter)
            
            
            # new frame
            frame = self.resize_im(picam2mt.frame, factor=img_downsample_factor)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # add to buffer
            frame_buffer.append(gray.copy())
            if len(frame_buffer)<=compute_every_n_frame:
                continue
            
            # benchmark time for each loop
            loop_timestamp_ms.append(time.time())
            
            # get a distance measure first, serves as 'ground truth'
            dist,_ = self.dist_sensor.get_distance()
            dist_measure.append(dist)
            
            # enough frame in buffer
            prev_gray = frame_buffer.pop(0)
            
            # first calculate TTC
            ttc_curr, C = calculate_ttc_v2(prev_gray, gray, K_inv)
            if np.isnan(ttc_curr):
                ttc_all.append(-1)
                if len(ttc_buffer)==0:
                    ttc = np.NAN
                else:
                    ttc = np.mean(ttc_buffer)
            else:
                ttc_all.append(ttc_curr)
                ttc_buffer.append(ttc_curr)
                if len(ttc_buffer) > ttc_buffer_sz:
                    ttc_buffer.pop(0)
                ttc = np.mean(ttc_buffer)
                
            turn_degree = 0
            if np.isnan(ttc) == False:
                if ttc < TTC_threshold and ttc>0: # turn around
                    # generate a degree (45,90), also random pos or neg
                    turn_degree = np.random.rand()*20.+40.
                    # if np.random.rand()<0.2:
                    #     turn_degree*=-1
                    self.turn_by_degree(monitor,turn_degree)
                    
                    # clear buffer every time the robot turns
                    ttc_buffer = [] 
            print('current ttc',ttc,'turn degree',turn_degree)
            turn_degree_all.append(turn_degree)
            
            # choose different optical flow method
            if flow_method==0 or flow_method==1:
                feature_list_prev = cv2.goodFeaturesToTrack(prev_gray, 20000, 0.03, 0.1)
                
            if flow_method==0:
                # Yi-Chung's implementation
                flow = LK_Optical_Flow(prev_gray, gray, 15, feature_list_prev)
                
            elif flow_method==1:
                # built-in LK with pyramid
                feature_list_new, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, feature_list_prev, None, **optical_flow_params)
                # feature_list_prev, feature_list_new = remove_large_outliers(feature_list_prev[status==1].copy(), feature_list_new[status==1].copy(),threshold=100) # output is n by 2 array
                feature_list_prev, feature_list_new = reshape_feature(feature_list_prev, feature_list_new)
                print(feature_list_prev.shape,feature_list_new.shape)
                # find displacement
                flow = uv_to_flow(prev_gray, feature_list_prev, feature_list_new)

            elif flow_method==2:
                # dense LK
                flow = cv2.optflow.calcOpticalFlowSparseToDense(prev_gray,gray,grid_step=16,sigma=0.05)
                
            elif flow_method==3:
                # Fauneback method
                flow = np.zeros((prev_gray.shape[0],prev_gray.shape[1],2))
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,flow, pyr_scale=0.5, levels=5, winsize=15, iterations=1, poly_n=5, poly_sigma=1.1, flags=0)

            if debug:
                out = self.plot_result_hsv(prev_gray, flow)
                cv2.imshow("optical flow", out)
                cv2.waitKey(1)
            
            # compute left and right hemisphere optical flow magnitude
            optflow_mag_l, optflow_mag_r = self.compute_hemi_optflow_mag(flow,img_downsample_factor)
            optflow_mag.append([optflow_mag_l,optflow_mag_r])
            if optflow_mag_l==0 and optflow_mag_r==0:
                e_lat=0
            else:
                e_lat = (optflow_mag_l - optflow_mag_r)/(optflow_mag_l + optflow_mag_r)
            
            if e_lat_last is None:
                PD = Kp*e_lat
            else:
                PD = Kp*e_lat + Kd*(e_lat - e_lat_last)
            if np.abs(e_lat)<e_threshold: # ignore if too small
                PD = 0
            e_lat_last = e_lat
            
            if np.isnan(PD):
                PD = 0
            # print(e_lat)
            PWM_fwd=30
            # PD=0
            speedL = (PWM_fwd+PD)/2
            speedR = (PWM_fwd-PD)/2
            print('optical flow: {:.2f}, {:.2f}'.format(optflow_mag_l, optflow_mag_r))
            print('setting current speed, {:.2f}, {:.2f}'.format(speedL, speedR))
            self.set_speed_left(speedL)
            self.set_speed_right(speedR)
            ctrl_output.append([speedL,speedR])
            
            n_iter+=1
            if n_iter >= max_iter:
                print('max iteration reached, exiting...')
                break
        t_end = time.time_ns()
        loop_time = float(t_end-t_start)/1e9/n_iter
        print('average loop time: {:.5f}'.format(loop_time))
        self.set_speed_left(0.)
        self.set_speed_right(0.)
        
        # save data
        loop_timestamp_ms = self.convert_array(loop_timestamp_ms)
        dist_measure = self.convert_array(dist_measure)
        ctrl_output = self.convert_array(ctrl_output)
        optflow_mag = self.convert_array(optflow_mag)
        # print(loop_timestamp_ms.shape, dist_measure.shape,ctrl_output.shape, optflow_mag.shape)
        # print(ttc_all)
        ttc_all = self.convert_array(ttc_all)
        # print(turn_degree_all)
        turn_degree_all = self.convert_array(turn_degree_all)
        # print(ttc_all.shape,turn_degree_all.shape)
        data_fn = 'OFData_Method{:d}_'.format(flow_method)+self.get_now_str()+'.txt'
        print('saving data to',data_fn)
        data = np.hstack([loop_timestamp_ms,dist_measure,ctrl_output,optflow_mag,ttc_all,turn_degree_all])
        print('data shape',data.shape)
        np.savetxt(data_fn,data)
    
    def convert_array(self,list):
        n = len(list)
        return np.array(list).reshape((n,-1))

    def resize_im(self, im, factor=4):
        if factor>1:
            im = cv2.resize(im,(int(im.shape[1]/factor),int(im.shape[0]/factor)))
        return im

    def turn_by_degree(self,monitor,turn_degree,verbose=0):
        time.sleep(0.5)
        if turn_degree is None:
            return
        turn_rad = turn_degree/180*np.pi
        start_yaw = monitor.imu_reading.value
        rot_mat = self.calculate_rot_matrix_from_rad(start_yaw)
        
        # initialize parameters
        err_rad_sum = 0.
        err_rad_diff = 0.
        err_rad_last = 0.
        init = 1
        
        last_time = time.time_ns()
        last_yaw_ref = None
        while True:
            curr_time = time.time_ns()
            if curr_time-last_time<=0.05*1e9:
                time.sleep(0.001)
                continue
            else:
                last_time = curr_time
                    
            curr_yaw = monitor.imu_reading.value
            curr_yaw_ref0 = self.convert_rad_in_ref(rot_mat,curr_yaw)
            # check continuity of yaw, problomatic only when the degree is close to pi or -pi
            curr_yaw_ref = self.ensure_continuity(curr_yaw_ref0, last_yaw_ref)
            # print(curr_yaw_ref0,curr_yaw_ref)
            err_rad = turn_rad - curr_yaw_ref
            if abs(err_rad) <= self.tol_rad:
                break
                
            err_rad_sum = err_rad_sum + err_rad
            if init == 0:
                err_rad_diff = err_rad - err_rad_last
            else:
                err_rad_diff = 0
                init = 0
            last_yaw_ref = curr_yaw_ref
            err_rad_last = err_rad
            
            # again use PID control
            P = self.constrain_turn_P(self.Kp_turn_rad*err_rad,1)
            D = self.Kd_turn_rad*err_rad_diff
            speedR = P + D
            speedL = -speedR
            
            self.set_speed_left(speedL)
            self.set_speed_right(speedR)
            if verbose==1:
                print(np.round(curr_yaw,2),np.round(err_rad/np.pi*180,2),
                    np.round(speedL,1),np.round(speedR,1))
        self.set_speed_left(0.)
        self.set_speed_right(0.)
        time.sleep(self.pause_time)

        self.report_current_pose(monitor)

    def turn_by_abs_yaw(self, monitor, target_yaw):
        c = np.cos(target_yaw)
        s = np.sin(target_yaw)
        new_heading = np.array([c,s])

        curr_yaw = monitor.imu_reading.value
        rot_mat = self.calculate_rot_matrix_from_rad(curr_yaw)

        turn_degree = self.calculate_deg_to_turn(rot_mat,new_heading)
        # print(turn_degree)
        self.turn_by_degree(monitor,turn_degree)

    def calculate_rotation_matrix(self,x,y):
        theta = np.arctan2(y,x)
        theta_orth = theta + np.pi/2
        rot_mat = np.array([[np.cos(theta),np.cos(theta_orth)],[np.sin(theta),np.sin(theta_orth)]])
        return rot_mat

    def calculate_deg_to_turn(self,rot_mat,new_heading):
        heading_hat = rot_mat.T.dot(new_heading.reshape((2,1))).flatten()
        theta = np.arctan2(heading_hat[1],heading_hat[0])
        return theta/np.pi*180