import csv
import matplotlib.pyplot as plt

x_vals, y_vals = [], []

# Leggiamo dal CSV
with open("Dati_Training.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di training")
plt.grid(True)
plt.show()

with open("Dati_Test.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di test")
plt.grid(True)
plt.show()

