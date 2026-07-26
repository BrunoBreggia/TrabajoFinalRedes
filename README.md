# Clasificador pre-entrenado para Fashion/MNIST

Para el proyecto final de la asignatura de posgrado Redes Neuronales se desarrolla un clasificador convolucional profundo para las imágenes de prendas de vestir de Fashion-MNIST. El mismo consta de un encoder pre-entrenado (con pesos congelados) y un clasificador, ambos convolucionales.

El proyecto consta de tres etapas:

1. Entrenamiento y ablación de un auto-encoder (encoder y decoder) convolucional.
2. Entrenamiento y ablación de un clasificador convolucional, a utilizar en conjunto con el encoder entrenado previamente (con sus pesos congelados). El clasificador se concatena a la salida del encoder.
3. Probar distintos esquemas de entrenamiento con el encoder y clasificador definidos previamente:
   1. Encoder pre-entrenado y con pesos congelados, más clasificador a entrenar
   2. Encoder sin pre-entrenar y sin congelar, a entrenar en conjunto con el clasificador 
   3. Encoder pre-entrenado pero sin congelar, haciendo fine-tunning a la par del entrenamiento del clasificador
   4. Encoder sin pre-entrenar y pesos congelado, más clasificador a entrenar
   
En base a los resultados obtenidos, se adjuntará en la carpeta de /docs un informe de no más de 4 páginas detallando la experiencia.
