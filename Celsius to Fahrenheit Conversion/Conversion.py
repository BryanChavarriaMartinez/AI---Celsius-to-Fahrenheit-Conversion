import tensorflow as tf
import numpy as np

# Se declaran las variables que usaremos como entrada y salida (resultados).
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Keras nos permite simplificar el entrenamiento
# Dense nos dice que hara conexiones de una neurona hacia todas las neuronas de la siguiente capa
# Units es la cantidad de neuronas en la capa de salida
# Input shape es la entrada, con una neurona
# Modelo Sequential es uno basico para pocas neuronas
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Optimizador Adam, ajusta los pesos y cezgos de manera eficiente, para que aprenda y no desaprenda
# Entre menor sea el numero de Adam, dara mejores resultados, pero tardara mas entrenando
# La funcion de pedida Mean squared error considera que una pequena cantidad de errores grande es peor que una gran catidad de errores pequenos
modelo.compile(
	optimizer = tf.keras.optimizers.Adam(0.1),
	loss = 'mean_squared_error'
)

# Para entrenar usamos la funcion fit, donde declaramos:
# La entrada, salida, la cantidad de vueltas que hara en los datos, y verbose para limpiar los datos
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=false)
print("Modelo entrenado!")

# Funcion de perdida, nos muestra una grafica con la magnitud de perdida
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

# Funcion para imprimir los valores del peso y el sezgo que considero para sus calculos
print("Variables internas del modelo")
print(capa.get_weights())