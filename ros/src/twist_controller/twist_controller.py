import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
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

        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
