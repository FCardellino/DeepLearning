# Segundo Trabajo de Aprendizaje Profundo

> Aplicación de lo aprendido en la materia optativa **Aprendizaje Profundo** de la Diplomatura en Ciencias de Datos FaMAFyC 2022.

> **Integrantes**: Candela Spitale | Fernando Cardellino | Carina Giovine | Carlos Serra

> **Profesores**: Johanna Frau y Mauricio Mazuecos

> **Problema**: Predecir la categoría de un artículo de MELI a partir de los títulos.

## Análisis y Visualización

Hicimos un análisis y visualización de la base de datos de **Meli Challenge 2019**, en los conjuntos de datos de `entrenamiento`, `validación` y `test` del idioma **spanish**.

Para cada conjunto, vimos:

* Cantidad y nombre de columnas con sus tipos:
  - object: `language`, `label_quality`, `title`, `category`, `split`, `tokenized_title`, `data`
  - int64: `target`, `n_labels`, `size`

* Cantidad de registros totales (`size`): diferían entre conjuntos, lógico debido a la división usual de los tipos de conjuntos, más cantidad en el de entrenamiento.
* Cantidad de valores nulos: ninguna para cualquier conjunto
* Primeros registros del conjunto para visualizar algunos valores únicos
* Cantidad de categorías distintas (`n_labels`): igual cantidad para todos, 632
* Cantidad de títulos distintos: coincidían en la cantidad de registros

## Preprocesamiento y tokenización de los datos

Luego de encontrar que los conjuntos a tratar ya están preprocesados y tokenizados, con las columnas `data` referente a **title** y `target` referente a **category**, no fue necesario hacer un preprocesamiento y tokenización, con acceder a estas columnas bastaba. El procedimiento de este desarrollo se encuentra en el repositorio de la materia en el archivo *experiment/preprocess_meli_data.ipynb*, donde también se puede encontrar el origen de otras columnas como `tokenized_title`, `n_labels` y `size`.

Básicamente, primero se concatenan los 3 conjuntos para evitar que el proceso en los datos no asigne el mismo token a la misma palabra en los distintos conjuntos (train/validation vs test) y ser una posible causa de un bajo rendimiento en el conjunto de test. Para el **preprocesamiento** se utilizan los módulos `stopwords`, `word_tokenize` de la librería `nltk` y el módulo `preprocessing` de la librería `gensim` los conjuntos concatenados. En cuanto a la **tokenización** que sigue, se utiliza el modelo `Dictionary` de el módulo `corpora` de la librería `gensim` y varios métodos de este modelo para lograrlo.

Como resultado se guarda por un lado, el conjunto *spanish_token_to_index.json.gz* para usarlos en los embeddings y, por otro lado, los 3 conjuntos por separadp, para poder dividir en las etapas de entrenamiento, validación y prueba final, en los archivos *spanish.test.jsonl.gz*, *spanish.train.jsonl.gz*, *spanish.validation.jsonl.gz*

## Manejador del dataset

Creamos una clase para modelar un conjunto de datos (cualquiera de os 3 que se instancie), que hereda de la clase `IterableDataset` de PyTorch, que si bien, no permite hacer shuffling de datos de forma fácil como la clase `Dataset` de Pytorch, el conjunto de datos es bastante grande (y podría serlo aún más en otro año para levantarlo en memoria).

Instanciamos los 3 conjuntos de datos con este módulo.

Creamos los dataloaders para cada conjunto haciendo uso de la clase `Dataloader`, que nos ayuda a entrenar con *mini-batches* (elegimos `128` para agilizar el tiempo de entrenamiento) al modelo para aumentar la eficiencia evitando iterar de a un elemento.

Para su definición creamos el módulo `PadSequences` para usar de *collation function* en el parámetro `collate_fn` que, dado que trabajamos con secuencias de palabras (representadas por sus índices en un vocabulario) y que el dataloader espera que los datos del *batch* tengan la misma dimensión (para poder llevarlos todos a un tensor de dimensión fija), se necesita redefinir los valores máximo, mínimo y de relleno distintos de los que están por defecto, de forma que dada una lista de secuencias, devuelve un tensor con *padding* sobre dichas secuencias.

Como en este caso trabajamos con secuencias de palabras (representadas por sus índices en un vocabulario), cuando queremos buscar un *batch* de datos, el `DataLoader` de PyTorch espera que los datos del *batch* tengan la misma dimensión (para poder llevarlos todos a un tensor de dimensión fija). Esto lo podemos lograr mediante el parámetro de `collate_fn`. En particular, esta función se encarga de tomar varios elementos de un `Dataset` y combinarlos de manera que puedan ser devueltos como un tensor de PyTorch. Muchas veces la `collate_fn` que viene por defecto en `DataLoader` sirve (como se vio en el notebook 2), pero este no es el caso. Se define un módulo `PadSequences` que toma un valor mínimo, opcionalmente un valor máximo y un valor de relleno (*pad*) y dada una lista de secuencias, devuelve un tensor con *padding* sobre dichas secuencias.

## Clase para el modelo

Para la clasificación utilizaremos un modelo de red convolucional que contará con 3 capas convolucionales (a las cuales se le aplicará _max pooling_) y 2 capas lineales (siendo una de ella la de salida), no profundizamos mucho para esta desición, entendimos que es arbitraria fuera de que el input y output obliguen a que al menos haya dos capas ocultas, y luego de explorar se podía decidir mejor.

En particular, tenemos la primera capa de `embeddings` que es rellenada con los valores de **word embeddings** (conversión del texto a una representación por vectores) continuos preentrenados en español de [SBW](https://crscardellino.ar/SBWCE/), de 300 dimensiones, en formato bz2, por eso utilizamos la librería `bz2` para descomprimir el archivo que los contiene (descargado en la carpata `data`). A su vez instanciamos el resto de las capas de la red con los tamaños pasados como argumento.

Además en la función de *forward*, aplicamos la matriz de embeddings ya creada al input, luego su salida es pasada por las redes convolucionales a las cuales se aplica max pooling (esto se ejecuta mendiante la función *conn_global_max_pool*). Posteriormente, estandarizamos el ancho de la matriz (para que pueda ser tomada por las capas lineales) la concatenación de cada vector de la matriz tensor, luego al resultado le aplicamos la función de activación `Relu` (escuchado en clase que es la que más se utiliza) a la primer capa lienal, y luego aplicamos la capa del output.

### 1ra Parte: Red Convolucional

> Creamos las funciones:

* **train_model**: entrena el modelo; por épocas ejecuta el **Back Propagation** y la **Optimización** con el optimizador pasado, calculando la función de pérdida; y por último loguea la época, la iteración y la función de pérdida de entrenamiento (`train_loss`) cada `50` mini-batches.

* **train_and_eval**: hace algo similar que la función anterior solo que también para el conjunto de **validación** (también valida) y que loguea la función de pérdida de entrenamiento (`train_loss`) por época y además loguea la métrica de `balance_accuracy` (aprovechando que usamos el conjunto de validación, decidimos medirla en este conjunto para compararla luego con la del conjunto de **test**) y la función de pérdida para el conjunto de validación (`val_loss`).

* **test_model**: evaluamos y predecimos con el conjunto de test y reportamos la métrica de `balance_accuracy` para este conjunto.

y

* **run_experiment**: ejecutamos un run del experimento; asignamos la función de pérdida `CrossEntropyLoss` al trabajar con un problema multiclase, llamamos a `train_and_eval` y a `train_model` con los dataloaders pasados y, si se desea además testear, llamamos a `test_model`. Registramos los **hiperparámetros**: la arquitectura del modelo, la función de pérdida, las épocas, la taza de aprendizaje y el optimizador.

* **run__mlflow_experiment**: ejecutamos un experimento; instanciamos el modelo pasando como parámetros el archivo de word embeddings, los datos tokenizados, el tamaño de vector (el tamaño de los embeddings, 300), el uso de barras de progreso activado, la cantidad de filtros y el lennght de cada uno, y el tamaño de la capa lineal. Enviamos el modelo a GPU y loguemos los **hiperparámetros**. Corremos el run con los dataloaders de entrenamiento y validación y, si se desea testear, agregamos el dataloader de test. Por último logueamos las métricas devueltas del run en MLFlow y calculamos las predicciones de a batches guardandolas en un archivo nuevo comprimido como artefacto de MLFlow.

Por último, creamos dos experimentos en MLFlow:

* `experiment_CNN_Classifier_w_3epochs`: para las etapas de entrenamiento y validación, el cual comprimimos en `data/op_experiments_w_3epochs_CNN_Classifier_FiltersCount100.csv.gz`

* `test_experiment_CNNClassifier_w_3epochs`: para la etapa de testeo, el cual comprimimos en `data/test_op_experiments_w_3epochs_CNNClassifier_FiltersCount100.csv.gz`.

### Arquitectura e hiperparámetros:

En todos los runs del experimento hacemos una red convolucional de 3 capas covolucionales (a las cuales luego aplicamos max pooling) y 2 capa lienales con los siguientes tamaños:

* Capas convolucionales:   
    * Canales de entrada: 300 (coincidente con la dimensión de los embbedings),  
    * Canales de salida: 100 (coincidente con la cantidad de filtros),  
    * Tamaño del kernel/filtro: 2, 3 y 4 (en cada capa respectivamente)
* Capas lineales o full connected:  
    * Primer capa: 1024, considerando que el tamaño de input es 300  
    * Capa de salida: 1024 que discrimina entre 632 categorías de salida (casi el doble que el tamaño de input)

Además, utilizamos `3` **épocas**, de forma arbitraria considerando que es un mínimo para agilizar el modelo.

Variamos a modo de exploración un poco aleatoria el **optimizador** y la **taza de aprendizaje**, si bien optamos por los dos optimizadores que se mencionaron más usuales en clase (`Adam` y `RMSprop`), y distintos valores de tazas de aprendizaje que reducimos a cantidad de dos debido a la gran demora en tiempo de ejecución y problemas de conexión con la máquina externa proporcionada que aumentaban el tiempo de dedicación al trabajo. Sin embargo anteriormente, probando con valores mayores de taza de aprendizaje nos dimos cuenta que a medida que aumentaba el valor se reducía el rendimiento del modelo, por lo cual, escogimos dos valores que ya devolvían diferencias grandes de métricas (`0.0001` y `0.001`).

### Experimento de Entrenamiento y Validación

#### Vista general de runs

> **Duración** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1olt886KYBwFW4Xh2nLdyRteJ_0_Q3B2Z' name='DuracionGeneral'>

> **Hiperparámetros** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1rtSXTjuOs4rRJdLkKSRfa26QOadntS65' name='ParamsGeneral'>

> **Métricas** 4to Run | 3er Run | 2do Run | 1er Run

<img src='https://drive.google.com/uc?id=1FSLKxDDNI7aiwONYXUFAhFompQ-zgUy_' name='MetricasGeneral'>

> Comparación de **loss** a través de las 3 épocas entrenamiento y validación

<img src='https://drive.google.com/uc?id=1P5tM0qywbfNAsQ4zc9xdCOFdY8M87oQ_' name='train_loss'>

<img src='https://drive.google.com/uc?id=1HdkVIcliO8i4K_WfunchHKjjLXcKv1Vw' name='val_loss'>

> Comparación **train_loss** vs **optimizador**

<img src='https://drive.google.com/uc?id=139GG5McDanoys694WLkyPjrP0WMmA4wN' name='train_loss_vs_Optim'>

> Comparación **val_loss** vs **optimizador**

<img src='https://drive.google.com/uc?id=1Cuc6R5mwOeygxiWTTAzO5zRrZpqs-KsR' name='val_loss_vs_Optim'>

#### Vista particular de runs

* **1er Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1ZDMikGSshf9OYSH4Ab_Gli2ffWcFiVzP' name='1erRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1GgDLS5j3b3XNfgHsv_oLUo0N0K5fum_X' name='1erRunMetricas'>

* **2do Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=13A0mYtl_XCEZVmqRYXzk8y-a1KVnt_7s' name='2doRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1OLB3q6reng1T4GwV89MX_b4KqBXHQb11' name='2doRunMetricas'>

* **3er Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=122Xr_0pmcvNDSIYNyJmF-Zaz_WylfokU' name='3erRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1y-vgm6hGJVO2LKMAk53fdlWthcISHJsz' name='3erRunMetricas'>

* **4to Run**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1WK5HhZzI-mwRo4X4T5VNt9Idr6D7uHe-' name='4toRunParametros'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1dQ6z4rqEGmHl8lCFODPzx0bnv9c_DvxM' name='4toRunMetricas'>


### Experimento de Validación y Test con el mejor modelo entrenado

#### Optimizador **Adam** y Learning Rate **0.0001**

> **Hiperparámetros**

<img src='https://drive.google.com/uc?id=1VRw2D1bOD2mFxzDchA8VfecaeXH8EUXG' name='TestParams'>

> **Métricas**

<img src='https://drive.google.com/uc?id=1bolsDZxKgA_6xPsJUaLp6vp6oznr6Ts0' name='TestMetricas'>

## Conclusión general:

* Se logró obtener un buen valor, con el set de test, para la métrica `balanced_accuracy`: **0.8**
* Comparando con el valor obtenido para la misma métrica con el modelo del Perceptrón Multicapa (**0.81**), el nuevo valor es menor pero no por mucho.
* Como próximos pasos, luego de haber obtenido un *base line* satisfactorio para ambos tipos de redes neuronales, la idea sería ver si los resultados obtenidos para `balanced_accuracy` pueden incrementarse sumando otras técnicas para optimizar los modelos como:  
  * modificar los *pesos de regularización*,  
  * o aplicar *dropouts*. 

## Contenido:

* TP_AprendizajeProfundo.ipynb: Jupyter Notebook con el trabajo resuelto.
* README.md: Informe del trabajo presentado
* Directorio data:
  - Directorio `experiments`: contiene los experimentos comprimidos `op_experiments_w_3epochs_CNN_Classifier_FiltersCount100.csv.gz` y `test_op_experiments_w_3epochs_CNNClassifier_FiltersCount100.csv.gz`

## Notas:

* Durante la ejecución del trabajo se descargarán el siguiente directorio y archivo dentro del directorio `data/`:

  - Directorio `meli-challenge-2019`: datasets del Melli Challenge 2019. Conjuntos a utilizar referentes a entrenamiento, validación y test en español y su concatenación, respectivamente: `spanish.train.jsonl.gz`, `spanish.validation.jsonl.gz`, `spanish.test.jsonl.gz`, `spanish_token_to_index.json.gz`

  - `SBW-vectors-300-min5.txt.bz2`: archivo de Word Embeddings utilizado

* Se utilizó la máquina externa nabucodonosor proporcionada para ejecutar el trabajo con recursos más grandes. Se puede acceder a la misma y ejecutarlo entrando a la terminal y corriendo:

$ `ssh -L localhost:{PORT}:localhost:{PORT} {usernabu}@nabucodonosor.ccad.unc.edu.ar`

(Instalar todos los paquetes y crear entorno virtual dictados en [0_set_up](https://github.com/DiploDatos/AprendizajeProfundo/blob/master/0_set_up.ipynb)).

Luego:

$ `git clone https://github.com/Knd9/optativa_deep_learning.git`

$ `jupyter notebook --port {PORT} --no-browser`

Para ver MLFlow

Cerrar el jupyter y correr:

$ `mlflow ui --port {PORT}`
