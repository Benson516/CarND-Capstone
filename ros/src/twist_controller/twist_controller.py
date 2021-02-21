import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, param_dict):
        # TODO: Implement
        self.yaw_controller = YawController(param_dict["wheel_base"],
                                            param_dict["steer_ratio"],
                                            0.1,
                                            param_dict["max_lat_accel"],
                                            param_dict["max_steer_angle"])
        
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0 # Minimum throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        # Parameters
        #------------------------------#
        self.vehicle_mass = param_dict["vehicle_mass"]
        self.fuel_capacity = param_dict["fuel_capacity"]
        self.brake_deadband = param_dict["brake_deadband"]
        self.decel_limit = param_dict["decel_limit"]
        self.accel_limit = param_dict["accel_limit"]
        self.wheel_radius = param_dict["wheel_radius"]
        #------------------------------#

        # Variables
        #------------------------------#
        self.last_time = rospy.get_time()
        self.last_vel = 0.0
        #------------------------------#


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # return 1., 0., 0.

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_vel = self.vel_lpf.filt(current_vel)

        #

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0


        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0.0
            brake = 700 # N-m
        elif throttle < 0.1 and vel_error < 0.0: # need to decelerate
            throttle = 0.0
            ideal_braking_time = 1.0 # sec.
            decel = max(vel_error*ideal_braking_time, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N-m

        return throttle, brake, steering


