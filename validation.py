import numpy as np
import pandas as pd
import torch
import glob
import os
# Zakładam, że masz już zainicjalizowany obiekt swojego systemu z załadowanymi wagami
from DiagnosisSystemClass import ExampleDiagnosisSystem 

def optimize_thresholds():
    DATA_DIR = r"C:\Users\anton\OneDrive\Desktop\data"
    
    # 1. Definiujemy koszyki zgodnie z Twoją macierzą FSM
    HEALTHY_FILES = glob.glob(os.path.join(DATA_DIR, "*_NF*.csv"))
    PIM_FILES = glob.glob(os.path.join(DATA_DIR, "*_pim_*.csv"))
    PIC_FILES = glob.glob(os.path.join(DATA_DIR, "*_pic_*.csv"))
    WAF_FILES = glob.glob(os.path.join(DATA_DIR, "*_waf_*.csv"))
    IML_FILES = glob.glob(os.path.join(DATA_DIR, "*_iml_*.csv")) 
    
    # Koszyki H0 (pliki, na których dana sieć NIE POWINNA reagować)
    H0_dict = {
        'MSO_0': HEALTHY_FILES + WAF_FILES,
        'MSO_10': HEALTHY_FILES + WAF_FILES + IML_FILES,
        'MSO_1': HEALTHY_FILES + PIC_FILES + WAF_FILES,
        'MSO_WAF': HEALTHY_FILES + PIC_FILES + PIM_FILES
    }
    
    # Słownik do trzymania maksymalnych wartości reziduów
    max_residuals = {'MSO_0': 0.0, 'MSO_10': 0.0, 'MSO_1': 0.0, 'MSO_WAF': 0.0}
    
    EDS = ExampleDiagnosisSystem()
    EDS.Initialize()
    
    print("Etap 1: Symulacja i szukanie absolutnych maksimów szumu (szukanie progów)...")
    
    # Zbieramy unikalną listę wszystkich plików do sprawdzenia
    all_files = set(HEALTHY_FILES + PIM_FILES + PIC_FILES + WAF_FILES + IML_FILES)
    
    for file_path in all_files:
        df = pd.read_csv(file_path)
        
        # Resetujemy stany wewnętrzne (notatniki) Bliźniaka przed każdym nowym plikiem
        EDS.x0 = torch.zeros(1, EDS.model0.num_states)
        EDS.x10 = torch.zeros(1, EDS.model10.num_states)
        EDS.x1 = torch.zeros(1, EDS.model1.num_states)
        EDS.xwaf = torch.zeros(1, EDS.modelwaf.num_states)
        
        EDS.e0_filt = 0.0
        EDS.e10_filt = 0.0
        EDS.e1_filt = 0.0
        EDS.ewaf_filt = 0.0
        
        # Odrzucamy pierwsze 100 próbek (strefa rozgrzewki, żeby nie zafałszować maksimów)
        burn_in = 100
        
        for idx in range(len(df)):
            sample = df.iloc[[idx]]
            
            # Puszczamy krok symulacji
            EDS.Input(sample)
            
            if idx > burn_in:
                # Jeśli to plik z koszyka "Zdrowe" dla danej sieci, sprawdzamy czy pobiliśmy rekord
                if file_path in H0_dict['MSO_0'] and EDS.e0_filt > max_residuals['MSO_0']:
                    max_residuals['MSO_0'] = EDS.e0_filt
                    
                if file_path in H0_dict['MSO_10'] and EDS.e10_filt > max_residuals['MSO_10']:
                    max_residuals['MSO_10'] = EDS.e10_filt
                    
                if file_path in H0_dict['MSO_1'] and EDS.e1_filt > max_residuals['MSO_1']:
                    max_residuals['MSO_1'] = EDS.e1_filt
                    
                if file_path in H0_dict['MSO_WAF'] and EDS.ewaf_filt > max_residuals['MSO_WAF']:
                    max_residuals['MSO_WAF'] = EDS.ewaf_filt

    print("\n--- ZNALEZIONE OPTYMALNE PROGI (Max Szum + 10% Marginesu) ---")
    margin = 1.10  # <--- ZMIENIONO NA 10% ZAPASU
    optimal_th0 = max_residuals['MSO_0'] * margin
    optimal_th10 = max_residuals['MSO_10'] * margin
    optimal_th1 = max_residuals['MSO_1'] * margin
    optimal_thwaf = max_residuals['MSO_WAF'] * margin
    
    print(f"MSO_0:   {optimal_th0:.4f}")
    print(f"MSO_10:  {optimal_th10:.4f}")
    print(f"MSO_1:   {optimal_th1:.4f}")
    print(f"MSO_WAF: {optimal_thwaf:.4f}")
    
    return optimal_th0, optimal_th10, optimal_th1, optimal_thwaf

if __name__ == "__main__":
    optimize_thresholds()