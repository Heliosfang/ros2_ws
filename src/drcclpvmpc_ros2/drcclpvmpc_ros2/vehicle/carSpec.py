# carSpec.py
import numpy as np
__author__ = 'Shiming Fang'
__email__ = 'sfang10@binghamton.edu'

def carSpec(parameters: dict) -> dict:
    # print("Loading car parameters...",parameters)
    lf = parameters['lf']
    lr = parameters['lr']
    mass = parameters['mass']
    dr = lr/(lf + lr)
    Iz = parameters['Iz']
    Bf = parameters['Bf']
    Cf = parameters['Cf']
    Df = parameters['Df']
    Br = parameters['Br']
    Cr = parameters['Cr']
    Dr = parameters['Dr']

    Caf = parameters['Caf']
    Car = parameters['Car']

    g = parameters['g']
    wheel_base = parameters['wheel_base']

    max_vx = parameters['max_vx']
    min_vx = parameters['min_vx']

    max_vy = parameters['max_vy']
    min_vy = parameters['min_vy']

    max_omega = parameters['max_omega']
    min_omega = parameters['min_omega']

    max_acc = parameters['max_acc']	# max acceleration [m/s^2]
    min_acc = parameters['min_acc']	# min acceleration [m/s^2]

    max_steer = parameters['max_steer']*np.pi/180	# max steering angle [rad]
    min_steer = parameters['min_steer']*np.pi/180 	# min steering angle [rad]

    max_inputs = [max_acc, max_steer]
    min_inputs = [min_acc, min_steer]
    
    control = parameters['control_type']

    if control == 'acc' or control == 'vx':
        approx = True
    else:
        approx = False

    Cm1 = parameters['Cm1']
    Cm2 = parameters['Cm2']
    Cr0 = parameters['Cr0']
    Cd = parameters['Cd']

    params = {
        'lf': lf,
        'lr': lr,
        'mass': mass,
        'wheel_base': wheel_base,
        'Iz': Iz,
        'Cf': Cf,
        'Cr': Cr,
        'Bf': Bf,
        'Br': Br,
        'Df': Df,
        'Dr': Dr,
        'Caf': Caf,
        'Car': Car,
        'dr': dr,
        'max_vx': max_vx,
        'min_vx': min_vx,
        'max_vy': max_vy,
        'min_vy': min_vy,
        'g': g,
        'max_acc': max_acc,
        'min_acc': min_acc,		
        'max_omega': max_omega,
        'min_omega': min_omega,
        'max_steer': max_steer,
        'min_steer': min_steer,
        'max_inputs': max_inputs,
        'min_inputs': min_inputs,
        'Cm1': Cm1,
        'Cm2': Cm2,
        'Cr0': Cr0,
        'Cd': Cd,
        'approx':approx
        }
    return params