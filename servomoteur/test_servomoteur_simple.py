from rpi_hardware_pwm import HardwarePWM
import time

pwm = HardwarePWM(pwm_channel=0, hz=60)
pwm.start(100) #cycle complet
pwm.change_duty_cycle(1) #angle de 0°
time.sleep(1)
pwm.change_duty_cycle(6) #angle de 90°
time.sleep(1)
pwm.change_duty_cycle(13) #angle de 180°
time.sleep(1)

pwm.stop()