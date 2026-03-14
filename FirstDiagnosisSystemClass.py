import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os
from DiagnosisSystemClass import DiagnosisSystemClass

# ====================================================================
# 1. UNIWERSALNE DEFINICJE SIECI (Wymagane do wczytania wag)
# ====================================================================
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
        
        # g_func zwraca tyle pochodnych, ile mamy stanów
        self.g_func = SubNetwork(total_features, num_states)
        self.h_func = SubNetwork(total_features, 1)

    # NOWOŚĆ: Funkcja do symulacji krok po kroku (czas rzeczywisty)
    def step(self, u_t, x_t):
        inputs = torch.cat((x_t, u_t), dim=1)
        y_hat_t = self.h_func(inputs)
        x_next = x_t + self.T * self.g_func(inputs)
        return y_hat_t, x_next

# ====================================================================
# 2. GŁÓWNA KLASA SYSTEMU DIAGNOSTYCZNEGO
# ====================================================================
class ExampleDiagnosisSystem(DiagnosisSystemClass):
    def __init__(self):
        super().__init__()
        
        # Definicje kolumn dla MSO 0
        self.u0_cols = ['Intercooler_pressure', 'intercooler_temperature', 'throttle_position', 'engine_speed']
        self.y0_cols = ['intake_manifold_pressure']
        
        # Definicje kolumn dla MSO 10
        self.u10_cols = ['air_mass_flow', 'intercooler_temperature', 'intake_manifold_pressure', 'throttle_position']
        self.y10_cols = ['Intercooler_pressure']
        
        # Definicje kolumn dla MSO 1
        self.u1_cols = ['ambient_pressure', 'ambient_temperature', 'intercooler_temperature', 'throttle_position', 'engine_speed', 'injected_fuel_mass', 'wastegate_position']
        self.y1_cols = ['intake_manifold_pressure']

        # ====================================================================
        # PROGI ALARMOWE (DO DOSTROJENIA!)
        # Wpisz tu wartości nieco wyższe niż średnie błędy (MAE) z Twoich testów
        # ====================================================================
        self.th0 = 2783.3   # Próg dla MSO 0 (np. 6000 Pa)
        self.th1 = 1713.18  # Próg dla MSO 1 (był głośniejszy, więc wyższy próg)
        self.th10 = 3000.0  # Próg dla MSO 10

    def Initialize(self):
        print("Inicjalizacja Grey-Box AI. Wczytywanie 3 modeli i 6 skalerów...")
        
        # --- Wczytywanie MSO 0 ---
        self.model0 = GreyBoxSystem(num_states=1, num_inputs=4) 
        self.model0.load_state_dict(torch.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\0\szara_skrzynka_mso0.pth"))
        self.model0.eval()
        self.scaler_u0 = joblib.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\0\scaler_u_mso0.pkl")
        self.scaler_y0 = joblib.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\0\scaler_y_mso0.pkl")
        self.x0 = torch.zeros(1, 1) # Tutaj też musi być 1 zaro
        self.e0_filt = 0.0        # Wyzerowany błąd
        
        # --- Wczytywanie MSO 10 ---
        self.model10 = GreyBoxSystem(num_states=1, num_inputs=4)
        self.model10.load_state_dict(torch.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\10\szara_skrzynka_mso10_wagi_zmiejszone.pth"))
        self.model10.eval()
        self.scaler_u10 = joblib.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\10\scaler_u_mso10_zmniejszone.pkl")
        self.scaler_y10 = joblib.load(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\source\10\scaler_y_mso10_zmniejszone.pkl")
        self.x10 = torch.zeros(1, 1)
        self.e10_filt = 0.0
        
        # --- Wczytywanie MSO 1 ---
        self.model1 = GreyBoxSystem(num_states=5, num_inputs=7)
        self.model1.load_state_dict(torch.load('source\\1\\szara_skrzynka_mso1_wagi.pth'))
        self.model1.eval()
        self.scaler_u1 = joblib.load('source\\1\\scaler_u_mso1.pkl')
        self.scaler_y1 = joblib.load('source\\1\\scaler_y_mso1.pkl')
        self.x1 = torch.zeros(1, 5) # Ten model ma aż 4 stany początkowe!
        self.e1_filt = 0.0
        
        print("Modele załadowane pomyślnie. Gotowość do pracy.")

    def Input(self, sample):
        # Wyłączamy gradienty, by inferencja trwała ułamek milisekundy
        with torch.no_grad():
            
            # --- Przetwarzanie próbki przez MSO 0 ---
            u0_raw = sample[self.u0_cols].values
            u0_norm = torch.tensor(self.scaler_u0.transform(u0_raw), dtype=torch.float32)
            y0_hat_norm, self.x0 = self.model0.step(u0_norm, self.x0)
            y0_hat = self.scaler_y0.inverse_transform(y0_hat_norm.numpy())
            e0 = abs(sample[self.y0_cols].values[0][0] - y0_hat[0][0])
            self.e0_filt = 0.001 * e0 + 0.999 * self.e0_filt # Filtr wygładzający skoki     # bylo 0.003
            
            # --- Przetwarzanie próbki przez MSO 10 ---
            u10_raw = sample[self.u10_cols].values
            u10_norm = torch.tensor(self.scaler_u10.transform(u10_raw), dtype=torch.float32)
            y10_hat_norm, self.x10 = self.model10.step(u10_norm, self.x10)
            y10_hat = self.scaler_y10.inverse_transform(y10_hat_norm.numpy())
            e10 = abs(sample[self.y10_cols].values[0][0] - y10_hat[0][0])
            self.e10_filt = 0.001 * e10 + 0.999 * self.e10_filt                             # bylo 0.005
            
            # --- Przetwarzanie próbki przez MSO 1 ---
            u1_raw = sample[self.u1_cols].values
            u1_norm = torch.tensor(self.scaler_u1.transform(u1_raw), dtype=torch.float32)
            y1_hat_norm, self.x1 = self.model1.step(u1_norm, self.x1)
            y1_hat = self.scaler_y1.inverse_transform(y1_hat_norm.numpy())
            e1 = abs(sample[self.y1_cols].values[0][0] - y1_hat[0][0])
            self.e1_filt = 0.001 * e1 + 0.999 * self.e1_filt                                    # bylo 0.01

        # ====================================================================
        # LOGIKA DETEKCJI I IZOLACJI W OPARCIU O FSM
        # ====================================================================
        # Konwersja błędów na sygnały binarne: 1 jeśli błąd przekracza nasz próg, w przeciwnym razie 0
        b0 = 1 if self.e0_filt > self.th0 else 0
        b10 = 1 if self.e10_filt > self.th10 else 0
        b1 = 1 if self.e1_filt > self.th1 else 0

        # DETEKCJA: Czy jakikolwiek model zgłosił błąd?
        detection = [1] if (b0 or b10 or b1) else [0]
        
        # IZOLACJA: 5 elementów zgodnie z wymaganiami [fpic, fpim, fwaf, fiml, fx_unknown]
        isolation = np.zeros((1, 5)) 
        
        if detection[0] == 1:
            # Wektor błędów, który "widzimy" z sieci
            observed_signature = np.array([b0, b10, b1])
            
            # Wektory oczekiwane wyciągnięte z Twojej analizy FSM: [MSO0, MSO10, MSO1]
            expected_fpic = np.array([1, 1, 0]) # Uszkodzony czujnik IC
            expected_fpim = np.array([1, 1, 1]) # Uszkodzony czujnik kolektora
            expected_fwaf = np.array([0, 1, 0]) # Uszkodzony przepływomierz
            expected_fiml = np.array([1, 0, 1]) # Dziura/wyciek
            
            signatures = np.array([expected_fpic, expected_fpim, expected_fwaf, expected_fiml])
            
            # Obliczanie odległości (podobieństwa) naszej aktualnej sygnatury do tych z FSM
            distances = np.sum(np.abs(signatures - observed_signature), axis=1)
            probabilities = np.maximum(0, 1.0 - distances / 3.0) 
            
            isolation[0, :4] = probabilities
            
            # Jeśli nasza sygnatura jest całkowicie "kosmiczna" i nie pasuje do żadnej znanej:
            if np.sum(probabilities) == 0:
                isolation[0, 4] = 1.0 # Oznaczamy jako fx_unknown
                
        return detection, isolation