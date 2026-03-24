import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # --- KONFIGURACJA ---
    folder_path = Path(r"C:\Users\anton\OneDrive\Desktop\Diagnosis_System\results")
    
    # Wybieramy tylko pliki .csv i omijamy te, które już mają w nazwie '_zmodyfikowany'
    files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix == '.csv' and '_zmodyfikowany' not in f.name]
    
    if not files:
        print("Brak odpowiednich plików CSV do przetworzenia.")
        return

    print(f"Znaleziono {len(files)} plików do przetworzenia.\n" + "-"*40)

    for input_file_path in files:
        output_file_path = input_file_path.with_name(f"{input_file_path.stem}_zmodyfikowany{input_file_path.suffix}")

        print(f"Przetwarzam plik: {input_file_path.name}")
        
        try:
            # Pamiętaj: w Twoim pierwszym pliku były przecinki (','). 
            # W pliku 'output_wltp_f_iml_4mm.csv' widziałem średniki (';').
            # Dopasuj separator (sep=',') do rzeczywistych danych!
            df = pd.read_csv(input_file_path, sep=',')
        except Exception as e:
            print(f"❌ Błąd wczytywania CSV: {e}")
            continue  # Przerywa ten obieg pętli i idzie do kolejnego pliku

        # Sprawdzamy, czy jest kolumna 'detection'
        if 'detection' not in df.columns:
            print(f"❌ Błąd: W pliku brak kolumny 'detection'! Pomijam.")
            continue  # Idzie do kolejnego pliku

        # KROK 1 i 2: Obliczenia
        df['det'] = np.where(df.index < 2400, 0, 1)
        df['diff'] = df['detection'] - df['det']

        # KROK 3: Zapis do pliku
        try:
            df.to_csv(output_file_path, sep=',', index=False)
            print(f"✅ Sukces! Zapisano jako: {output_file_path.name}")
        except Exception as e:
            print(f"❌ Błąd przy zapisywaniu pliku: {e}")
            
        print("-" * 40)

    print("Zakończono przetwarzanie wszystkich plików.")

# To musi być bez żadnych wcięć!
if __name__ == "__main__":
    main()