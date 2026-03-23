import subprocess
from pathlib import Path

def main():
    # Ścieżka do folderu z danymi
    folder_path = Path(r"C:\Users\anton\OneDrive\Desktop\data")
    
    # Nazwa skryptu do uruchomienia (zakładamy, że jest w bieżącym katalogu roboczym)
    script_to_run = "RunDiagnoser.py"

    # Sprawdzenie, czy folder istnieje
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Błąd: Folder '{folder_path}' nie istnieje lub nie jest katalogiem.")
        return

    # Pobranie wszystkich plików z folderu
    files = [f for f in folder_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"W folderze '{folder_path}' nie ma żadnych plików.")
        return

    print(f"Znaleziono {len(files)} plików. Rozpoczynam przetwarzanie...\n")
    print("-" * 50)

    # Iteracja po każdym pliku i uruchomienie skryptu RunDiagnoser.py
    for file_path in files:
        print(f"Przetwarzam plik: {file_path.name}")
        
        # Przygotowanie komendy jako listy argumentów (bezpieczniejsze niż zwykły string)
        command = ["python", script_to_run, str(file_path)]
        
        try:
            # Uruchomienie komendy i czekanie na jej zakończenie
            # capture_output=True pozwala przechwycić to, co RunDiagnoser.py wypisuje na ekran
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Sprawdzenie, czy skrypt wykonał się poprawnie (kod wyjścia 0)
            if result.returncode == 0:
                print(f"✅ Sukces: {file_path.name}")
                # Jeśli chcesz widzieć output z RunDiagnoser, odkomentuj poniższą linijkę:
                # print(result.stdout)
            else:
                print(f"❌ Błąd podczas przetwarzania {file_path.name} (Kod: {result.returncode})")
                print(f"Szczegóły błędu:\n{result.stderr}")
                
        except Exception as e:
            print(f"⚠️ Wystąpił nieoczekiwany wyjątek podczas uruchamiania pliku {file_path.name}: {e}")
            
        print("-" * 50)

    print("\nZakończono przetwarzanie wszystkich plików.")

if __name__ == "__main__":
    main()