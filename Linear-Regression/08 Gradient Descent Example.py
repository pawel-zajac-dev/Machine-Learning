import numpy as np

w = 20
tolerance = 1e-6  # minimalna zmiana, przy której uznajemy, że w się "nie zmienia"
max_iter = 1000  # aby uniknąć nieskończonej pętli

for i in range(max_iter):
    w_new = w - 0.1 * 2 * w  # aktualizacja
    change = abs(w_new - w)

    if change < tolerance:  # jeśli zmiana jest bardzo mała
        print(f"Zatrzymano: zmiana < {tolerance}")
        break
    if w_new > w:  # jeśli wartość zaczyna rosnąć
        print("Zatrzymano: wartość zaczęła rosnąć")
        break

    w = w_new
    print(w)
