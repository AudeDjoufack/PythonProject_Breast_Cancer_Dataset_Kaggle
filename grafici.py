import csv
import matplotlib.pyplot as plt

x_vals, y_vals = [], []

# Leggiamo dal CSV
with open("Dati_Training_loss.csv", mode="r") as f:
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

with open("Dati_Test_loss.csv", mode="r") as f:
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

with open("Dati_Training_total_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di training --> la loss function totale di ogni epoch")
plt.grid(True)
plt.show()

with open("Dati_Test_total_loss.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di test --> la loss function totale di ogni epoch")
plt.grid(True)
plt.show()
plt.grid(True)
plt.show()

with open("Dati_Training_accuracy.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di training --> l'accuracy di ogni epoch")
plt.grid(True)
plt.show()
plt.grid(True)
plt.show()

with open("Dati_Test_accuracy.csv", mode="r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_vals.append(float(row["epoch"]))
        y_vals.append(float(row["loss"]))

# Creiamo il grafico
plt.plot(x_vals, y_vals, marker="o")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Grafico da dati di test --> l'accuracy di ogni epoch")
plt.grid(True)
plt.show()

