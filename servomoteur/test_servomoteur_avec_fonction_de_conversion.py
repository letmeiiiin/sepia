from rpi_hardware_pwm import HardwarePWM
import time



def angle_to_percent(angle):
    if angle > 180 or angle < 0:
        return False
    start = 4
    end = 16
    ratio = (end - start)/360  # Calcul du ratio
    angle_as_percent = angle * ratio
    return start + angle_as_percent

pwm = HardwarePWM(pwm_channel=0, hz=60)
pwm.start(100) # cycle complet
pwm.change_duty_cycle(1)   #angle de 0°
time.sleep(1)
pwm.change_duty_cycle(angle_to_percent(90)) #angle de 90°
time.sleep(1)
pwm.change_duty_cycle(13)   #angle de 180°
time.sleep(1)

pwm.stop()
