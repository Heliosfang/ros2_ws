import numpy as np
import os
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

class Model_noiseVx:
    def __init__(self,horizon,fix_noise) -> None:
        self.n_states = 5
        self.n_inputs = 2
        self.horizon_ = horizon
 
        self.dir_name = os.path.dirname(__file__)

        fixnoise_state_file = "noise_arrvx.npy"
        accumulated_noise_file = "noise_arr.npy"
        self.num_his = 5

        self.fix_noise_data = os.path.join(self.dir_name,"data",fixnoise_state_file)
        
        accumulated_noise = os.path.join(self.dir_name,"data",accumulated_noise_file)
        
        accumulated_noise = np.load(accumulated_noise)
        length = accumulated_noise.shape[0]
        
        vx_accumulated_noise = np.zeros((length,self.n_states*self.horizon_,1))
        for i in range(length):
            vx_accumulated_noise[i,0:3] = accumulated_noise[i,0:3]
            vx_accumulated_noise[i,3:5] = accumulated_noise[i,4:6]
            
        length_n = vx_accumulated_noise.shape
        
        print("total his samples :",length_n)
        
        if fix_noise:
            self.selected_noise = np.load(self.fix_noise_data)
        else:
            random_rows_indices = np.random.choice(length,self.num_his,replace=False)
            print("random rows indices:",random_rows_indices)
            self.selected_noise = vx_accumulated_noise[random_rows_indices]

    def get_noise_arr(self):
        return self.selected_noise
    
    def save_noise_arr(self):
        np.save(self.fix_noise_data,self.selected_noise)

if __name__=='__main__':
    horizon = 6
    model_noise = Model_noise(horizon,False)
    model_noise.save_noise_arr()
    # new_noise = np.array([0,0,0,0,0,0])
    # model_noise.random_update_noise_matrix()
    # print("updated_noise :",model_noise.get_noise_arr())


    # z_states, m_noise = model_noise.get_noise_arr()
    # print(z_states[0].shape)
    # print(m_noise[0].shape)
    
