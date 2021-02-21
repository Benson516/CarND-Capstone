
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_error = 0.

    def reset(self):
        self.int_val = 0.0

    def step(self, error, sample_time):

        # integral = self.int_val + error * sample_time
        integral_delta = error * sample_time
        derivative = (error - self.last_error) / sample_time

        val = self.kp * error + self.ki * self.int_val + self.kd * derivative;

        delta_vel = 0.0
        if val > self.max:
            delta_vel = val - self.max
            val = self.max
        elif val < self.min:
            delta_vel = val - self.min
            val = self.min
        else:
            # self.int_val = integral
            pass
        
        # Anti-windup
        if delta_vel * (self.ki * integral_delta) > 0.0:
            integral_delta = 0.0

        self.int_val += integral_delta
        self.last_error = error

        return val
