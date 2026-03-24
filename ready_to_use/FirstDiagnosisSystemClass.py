import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os
from DiagnosisSystemClass import DiagnosisSystemClass


class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SubNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)

class GreyBoxSystem(nn.Module):
    def __init__(self, num_states, num_inputs, T_sample=0.05):
        super(GreyBoxSystem, self).__init__()
        self.T = T_sample
        self.num_states = num_states
        total_features = num_states + num_inputs
        
        self.g_func = SubNetwork(total_features, num_states)
        self.h_func = SubNetwork(total_features, 1)

    def step(self, u_t, x_t):
        inputs = torch.cat((x_t, u_t), dim=1)
        y_hat_t = self.h_func(inputs)
        x_next = x_t + self.T * self.g_func(inputs)
        return y_hat_t, x_next


class ExampleDiagnosisSystem(DiagnosisSystemClass):
    def __init__(self):
        super().__init__()
        
        # Definicje kolumn dla MSO 0
        self.u0_cols = ['Intercooler_pressure', 'intercooler_temperature', 'throttle_position', 'engine_speed']
        self.y0_cols = ['intake_manifold_pressure']
        
        # Definicje kolumn dla MSO 10
        self.u10_cols = ['delta_pressure', 'air_mass_flow', 'throttle_position']
        self.y10_cols = ['injected_fuel_mass']
        
        # Definicje kolumn dla MSO 1
        self.u1_cols = ['ambient_pressure', 'ambient_temperature', 'intercooler_temperature', 'throttle_position', 'engine_speed', 'injected_fuel_mass', 'wastegate_position']
        self.y1_cols = ['intake_manifold_pressure']

        self.ywaf_cols = ['injected_fuel_mass']

        self.th0 = 2801.1176         
        self.th1 = 1523.1193 
        self.th10 = 0.0002 
        self.thwaf = 0.0001      

    def Initialize(self):
        print("Grey-Box init. Reading weights and scalers...")
        
        self.model0 = GreyBoxSystem(num_states=1, num_inputs=4) 
        self.model0.load_state_dict(torch.load('params\weights_mso0.pth'))
        self.model0.eval()
        self.scaler_u0 = joblib.load('params\scaler_u_mso0.pkl')
        self.scaler_y0 = joblib.load('params\scaler_y_mso0.pkl')
        self.x0 = torch.zeros(1, 1)
        self.e0_filt = 0.0
        
        self.model10 = GreyBoxSystem(num_states=1, num_inputs=3)
        self.model10.load_state_dict(torch.load('params\weights_mso10.pth'))
        self.model10.eval()
        self.scaler_u10 = joblib.load('params\scaler_u_mso10.pkl')
        self.scaler_y10 = joblib.load('params\scaler_y_mso10.pkl')
        self.x10 = torch.zeros(1, 1)
        self.e10_filt = 0.0
        
        self.model1 = GreyBoxSystem(num_states=5, num_inputs=7)
        self.model1.load_state_dict(torch.load('params\weights_mso1.pth'))
        self.model1.eval()
        self.scaler_u1 = joblib.load('params\scaler_u_mso1.pkl')
        self.scaler_y1 = joblib.load('params\scaler_y_mso1.pkl')
        self.x1 = torch.zeros(1, 5) 
        self.e1_filt = 0.0

        self.modelwaf = GreyBoxSystem(num_states=1, num_inputs=3)
        self.modelwaf.load_state_dict(torch.load('params\weights_waf.pth'))
        self.modelwaf.eval()
        self.scaler_uwaf = joblib.load('params\scaler_u_waf.pkl')
        self.scaler_ywaf = joblib.load('params\scaler_y_waf.pkl')
        self.xwaf = torch.zeros(1, 1) 
        self.ewaf_filt = 0.0
        
        print("Models loaded. Ready to use.")

    def Input(self, sample):
        with torch.no_grad():
            
            u0_raw = sample[self.u0_cols].values
            u0_norm = torch.tensor(self.scaler_u0.transform(u0_raw), dtype=torch.float32)
            y0_hat_norm, self.x0 = self.model0.step(u0_norm, self.x0)
            y0_hat = self.scaler_y0.inverse_transform(y0_hat_norm.numpy())
            e0 = abs(sample[self.y0_cols].values[0][0] - y0_hat[0][0])
            self.e0_filt = 0.001 * e0 + 0.999 * self.e0_filt 
            
            pim = sample['intake_manifold_pressure'].values[0]
            pic = sample['Intercooler_pressure'].values[0]
            delta_p = np.sqrt(np.abs(pim - pic))
            
            amf = sample['air_mass_flow'].values[0]
            thr = sample['throttle_position'].values[0]
            
            u10_raw = np.array([[delta_p, amf, thr]])
            
            u10_norm = torch.tensor(self.scaler_u10.transform(u10_raw), dtype=torch.float32)
            y10_hat_norm, self.x10 = self.model10.step(u10_norm, self.x10)
            y10_hat = self.scaler_y10.inverse_transform(y10_hat_norm.numpy())
            e10 = abs(sample[self.y10_cols].values[0][0] - y10_hat[0][0])
            self.e10_filt = 0.001 * e10 + 0.999 * self.e10_filt                  
            
            u1_raw = sample[self.u1_cols].values
            u1_norm = torch.tensor(self.scaler_u1.transform(u1_raw), dtype=torch.float32)
            y1_hat_norm, self.x1 = self.model1.step(u1_norm, self.x1)
            y1_hat = self.scaler_y1.inverse_transform(y1_hat_norm.numpy())
            e1 = abs(sample[self.y1_cols].values[0][0] - y1_hat[0][0])
            self.e1_filt = 0.001 * e1 + 0.999 * self.e1_filt                                   

            engine_speed = sample['engine_speed'].values[0]
            air_mass_flow = sample['air_mass_flow'].values[0]
            throttle_pos = sample['throttle_position'].values[0]
            
            epsilon = 1e-6
            waf_x1 = np.log(engine_speed + epsilon) * air_mass_flow
            waf_x2 = air_mass_flow
            waf_x3 = np.log(throttle_pos + epsilon)
            
            uwaf_raw = np.array([[waf_x1, waf_x2, waf_x3]])
            uwaf_norm = torch.tensor(self.scaler_uwaf.transform(uwaf_raw), dtype=torch.float32)
            
            ywaf_hat_norm, self.xwaf = self.modelwaf.step(uwaf_norm, self.xwaf)
            ywaf_hat = self.scaler_ywaf.inverse_transform(ywaf_hat_norm.numpy())
            
            y_true_waf = sample[self.ywaf_cols].values[0][0]
            ewaf = abs(y_true_waf - ywaf_hat[0][0])
            self.ewaf_filt = 0.001 * ewaf + 0.999 * self.ewaf_filt  

        b0 = 1 if self.e0_filt > self.th0 else 0
        b10 = 1 if self.e10_filt > self.th10 else 0
        b1 = 1 if self.e1_filt > self.th1 else 0
        bwaf = 1 if self.ewaf_filt > self.thwaf else 0

        detection = [1] if (b0 or b10 or b1 or bwaf) else [0]
        
        isolation = np.zeros((1, 5)) 
        
        if detection[0] == 1:
            observed_signature = np.array([b0, b10, b1, bwaf])
            
            expected_fpic = np.array([1, 1, 0, 0]) 
            expected_fpim = np.array([1, 1, 1, 0]) 
            expected_fwaf = np.array([0, 0, 0, 1]) 
            expected_fiml = np.array([1, 0, 1, 1]) 
            
            signatures = np.array([expected_fpic, expected_fpim, expected_fwaf, expected_fiml])
            
            distances = np.sum(np.abs(signatures - observed_signature), axis=1)
            probabilities = np.maximum(0, 1.0 - distances / 4.0) 
            
            isolation[0, :4] = probabilities
            
            if np.sum(probabilities) == 0:
                isolation[0, 4] = 1.0 
                
        return detection, isolation