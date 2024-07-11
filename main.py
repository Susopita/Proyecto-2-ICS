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

while True:
    opc = input(": ")
    if opc in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-1"):
        break
    else:
        print("Opción inválida")

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
for i, foto in enumerate(fotos, start=1):
    print(f" {i}) {foto[:-4]}")

while True:
    opc = input(": ")
    if opc in [f"{i}" for i in range(1, len(fotos) + 1)]:
        break
    else:
        print("Opción inválida")

imagen = Image.open(f"./images/{fotos[int(opc)-1]}")

# Escalamos la imagen a 8x8
imagen = imagen.resize((8,8), Image.LANCZOS)

# Escalamos los valores de la imagen a un rango de 0 a 16
imagen = imagen.convert("L")
imagen = imagen.point(lambda x: (255 - x) * (16 / 255))
imagen = np.array(imagen)

# Funcion de distancia
def distancia(x, y):
    return np.linalg.norm(x - y)

# Obtener los 3 digitos más cercanos
df_digits['dist'] = df_digits.drop(columns="target").apply(lambda x: distancia(np.array(x), imagen.flatten()), axis=1)

df_digits_sorted = df_digits.sort_values(by="dist")

# Mostramos los 3 digitos más cercanos
cercanos_digits = df_digits_sorted.head(5)
print(cercanos_digits[['target', 'dist']])

# Mostramos el número que la IA ha detectado
target_cercanos = cercanos_digits['target'].value_counts()

identifico = False
for target, i in target_cercanos.items():
    if i >= 3:
        print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {target}")
        identifico = True

if identifico == False:
    print("Soy la inteligencia artificial, y no he podido identificar correctamente el número ingresado")

mean_digits['dist'] = mean_digits.apply(lambda x: distancia(np.array(x), imagen.flatten()), axis=1)

mean_cercanos_digits = mean_digits.sort_values(by="dist")

number_predicted = mean_cercanos_digits.head(1).index[0]
print(f"Soy  la  inteligencia  artificial  versión  2,  y  he detectado  que  el  dígito  ingresado  corresponde  al  número  {number_predicted}")

"""
h) Indique cuál de los dos métodos cree usted que es mejor, el de la versión 1  o el de la versión 2.

El método de la versión 2 es mejor, ya que se basa en la distancia entre la imagen ingresada y todas las imagenes de los dígitos, lo que permite una mejor clasificación de la imagen ingresada al tener mayores casos de prueba. Por otro lado, la versión 1 se basa en la imagen promedio de cada dígito, lo que puede no ser tan efectivo si la imagen ingresada tiene variaciones en la forma del dígito.

Pese a que la version 2, es efectiva en su trabajo, las ligeras variaciones en la imagen como  algun tipo de ruido puede afectar al resultado final.

Una observacion adicional, es que la version 2 esta adaptada a si o si predecir un numero que conozca, mientras que la version 1, puede predecir un numero que no conozca, ya que se basa en vecinos cercano y si tiene 3 vecinos distintos, puede retornar que no se identifica correctamente el numero.

Otra obseración, aparentemente la version 2 es efectiva en los casos que el numero tiene un grosor normal que sea distingible, pero si el numero es muy delgado o muy grueso, puede que no se identifique correctamente el numero. Lo que no pasa con la version 1, al tener varios casos de prueba y distintas forma de clasificar el numero.
"""