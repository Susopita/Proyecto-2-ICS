import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from PIL import Image
import numpy as np
import os

# Cargamos el dataset de digit
digits = datasets.load_digits()

# Generamos el DataFrame con los promedios de cada digito
df_digits = pd.DataFrame(digits.data)
df_digits["target"] = digits.target

mean_digits = df_digits.groupby("target").mean()

# Mostramos la imagen promedio que el usuario escoja
print("Seleccione el número de la imagen que desea ver")
for i in range(10):
    print(f"{i} - imagen promedio")
print("-1 - todas las imágenes promedio")

opc = input(": ")

if opc == "-1":
    fig, axs = plt.subplots(2, 5, figsize=(15, 5))
    for ax, i in zip(axs.ravel(),range(10)):
        ax.imshow(mean_digits.iloc[i].values.reshape(8,8), cmap='viridis')
    plt.tight_layout()
    plt.show()
else:
    for i in range(10):
        if opc == str(i):
            plt.imshow(mean_digits.iloc[i].values.reshape(8,8), cmap='viridis')
            plt.show()
            break

if not os.path.exists("images"):
    os.makedirs("images")

fotos = os.listdir("images")

print("Mencione la imagen que desea utilizar...")
for i in fotos:
    print(f" - {i}")

opc = input(": ")

imagen = Image.open(f"./images/{opc}")

# Escalamos la imagen a 8x8
imagen = imagen.resize((8,8), Image.LANCZOS)

# Escalamos los valores de la imagen a un rango de 0 a 16
imagen = imagen.convert("L")
imagen = imagen.point(lambda x: x * (16 / 255))
imagen = np.array(imagen)

# Funcion de distancia
def distancia(x, y):
    return np.linalg.norm(x - y)

# Obtener los 3 digitos más cercanos
df_digits['dist'] = df_digits.drop(columns="target").apply(lambda x: distancia(np.array(x), imagen.flatten()), axis=1)

df_digits_sorted = df_digits.sort_values(by="dist")

# Mostramos los 3 digitos más cercanos
print(df_digits_sorted.head(3))