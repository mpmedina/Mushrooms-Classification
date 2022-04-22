import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sys import platform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from results import Results

if platform == "darwin":
    # Esto soluciona el error de macOS "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CNN:
    """Clase para clasificar las imágenes utilizando transfer learning o fine-tuning on una red CNN pre entrenada.

        Ejemplos:
            1. Entrenar y evaluar la CNN. Opcionalmente se puede guardar el modelo.
                cnn = CNN()
                cnn.train(training_dir, validation_dir, base_model='DenseNet121')
                cnn.predict(validation_dir)
                cnn.save(filename)

            2. Cargamos una red CNN para evaluar un conjunto de pruebas nunca visto. N to evaluate against a previously unseen test set.
                cnn = CNN()
                cnn.load(filename)
                cnn.predict(test_dir)

    """

    def __init__(self):
        """Se inicializan las clases con la CNN de transfer learning"""
        self._model_name = ""
        self._model = None
        self._target_size = None
        self._preprocessing_function = None

    def train(self, training_dir: str, validation_dir: str, base_model: str, epochs: int = 1,
              unfreezed_convolutional_layers: int = 0, training_batch_size: int = 32, validation_batch_size: int = 32,
              learning_rate: float = 1e-4): 
# Hemos establecido un valor por defecto a los hiperparámetros del modelo. 
    # El número de epochs por defecto es 1, pero como hemos comentado en el notebook este valor se ha modificado para mejorar de forma considerable el accuracy del modelo. 
    # El número de capas convolucionales sin descongelar es 0, en este caso hemos comprobado que el accuracy del modelo no incrementa de forma significativa y a medida que se añaden más capas convolucionales aumenta el tiempo de ejecución.
    # El learning rate por defecto que hemos seleccionado es 0.0001, observamos que para learning rates más bajos tenemos mucha variabilidad en el accuracy del modelo, mientras que para valores más bajos, se queda atascado en algún punto sin llegar al accuracy óptimo.
    
        """Usamos transfer learning o fine-tuning para entrenar una red neuronal básica para clasificar nuevas categorías.

        Args:
            training_dir: Ruta relativa al directorio de training (e.g., 'dataset/training').
            validation_dir: Ruta relativa al directorio de validation (e.g., 'dataset/validation').
            base_model: CNN pre-entrenado { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            epochs: Número de veces que el dataset pasa para alante y para atrás por la red neuronal.
            unfreezed_convolutional_layers: Empezando por el final, número de capas convolucionales entrenables.
            training_batch_size: Número de imágenes de entrenamiento usadas en una iteración.
            validation_batch_size: Número de imágenes de validación usadas en una iteración.
            learning_rate: Ratio de aprendizaje.

        """
        # Inicializamos una CNN preentrenada sin la capa de clasificación
        self._initialize_base_model(base_model, unfreezed_convolutional_layers, include_top=False)

        # Configuramos las funciones de carga de los datos, preprocesado y data augmantation
        print('\n\nReading training and validation data...')
        training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self._preprocessing_function,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,  # Giramos de forma aleatoria la mitad de las imágenes horizontalmente
            fill_mode='nearest'  # Estrategia usada para rellenar los nuevos píxeles que aparecen después de transformar las imágenes
        )

        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        training_generator = training_datagen.flow_from_directory(
            training_dir,
            target_size=self._target_size,
            batch_size=training_batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self._target_size,
            batch_size=validation_batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Añadimos una nueva capa de salida del tipo SoftMax para aprendernos las clases del conjunto de entrenamiento
        self._add_output_layers(training_generator.num_classes)

        # Compilamos el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self._model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Representamos un resumen del modelo
        print('\n\nModel summary')
        self._model.summary()

        # Callbacks. Check https://www.tensorflow.org/api_docs/python/tf/keras/callbacks for more alternatives.
        # EarlyStopping and ModelCheckpoint are probably the most relevant.

        # Para lanzar TensorBoard escribe lo siguiente en una Terminal: tensorboard --logdir /path/to/log/folder
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.abspath("./logs"), histogram_freq=0,
            write_graph=True, write_grads=False,
            write_images=False, embeddings_freq=0,
            embeddings_layer_names=None, embeddings_metadata=None,
            embeddings_data=None, update_freq='epoch'
        )

        callbacks = [tensorboard_callback]

        # Entrenamos la red
        print("\n\nTraining CNN...")

        history = self._model.fit(
            training_generator,
            epochs=epochs,
            steps_per_epoch=len(training_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks
        )

        # Representamos el historial de entrenamiento del modelo
        if epochs > 1:
            self._plot_training(history)

    def predict(self, test_dir: str, dataset_name: str = "", save: bool = True):
        """Evalúa un nuevo set de imágenes usando la CNN entrenada.

        Args:
            test_dir: Ruta relativa al directorio de datos de validación (e.g., 'dataset/test').
            dataset_name: Nombre descriptivo del dataset.
            save: Guardamos los resultados en un fichero de Excel.

        """
        # Configuramos las funciones de carga y preprocesado de datos
        print('Reading test data...')
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self._preprocessing_function)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self._target_size,
            batch_size=1,  # Con un batch = 1 nos aseguramos que todas las imégenes de test se procesen
            class_mode='categorical',
            shuffle=False
        )

        # Categorías predichas
        predictions = self._model.predict(test_generator)
        predicted_labels = np.argmax(predictions, axis=1).ravel().tolist()

        # Formateamos los resultados y calculamos las estadísticas de clasificación
        results = Results(test_generator.class_indices, dataset_name=dataset_name)
        accuracy, confusion_matrix, classification = results.compute(test_generator.filenames, test_generator.classes,
                                                                     predicted_labels)
        # Mostramos y guardamos los resultados
        results.print(accuracy, confusion_matrix)

        if save:
            results.save(confusion_matrix, classification, predictions)

    def load(self, filename: str):
        """Cargamos un modelo CNN entrenado y la información de preprocesado correspondiente.

        Args:
           filename: Ruta relativa al archivo sin la extensión.

        """
        # Cargamos el modelo Keras
        self._model = tf.keras.models.load_model(filename + '.h5')

        # Cargamos la información básica del modelo
        with open(filename + '.json') as f:
            self._model_name = json.load(f)

        self._initialize_attributes()

    def save(self, filename: str):
        """Guardamos el modelo en un archivo .h5 y el nombre del modelo en un archivo .json

        Args:
           filename: Ruta relativa al archivo sin la extensión.

        """
        # Guardamos el modelo Keras
        self._model.save(filename + '.h5')

        # Guardamos la información básica del modelo
        with open(filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(self._model_name, f, ensure_ascii=False, indent=4, sort_keys=True)

    def _initialize_base_model(self, base_model: str, unfreezed_convolutional_layers: int, include_top: bool = True,
                               pooling: str = 'avg'):
        """Inicializamos el modelo base.

        Args:
            base_model: Pre-trained CNN { DenseNet121, DenseNet169, DenseNet201, InceptionResNetV2, InceptionV3,
                                          MobileNet, MobileNetV2, NASNetLarge, NASNetMobile, ResNet50, VGG16, VGG19,
                                          Xception }.
            unfreezed_convolutional_layers: Empezando por el final, número de capas convolucionales entrenables.
            include_top: True para usar el modelo base completo; false para eliminar las últimas capas de clasificación.
            pooling: Modo pooling opcional para extracción de features cuando include_top es False
                - None: El output del modelo será el output del tensor 4D del último bloque convolucional.
                - 'avg': La media global del pooling se aplica al output del último bloque convolucional, y por tanto, el output
                         del modelo será un tensor 2D.
                - 'max': Se aplicará Global max pooling.

        Raises:
            TypeError: Si el parámetro unfreezed_convolutional_layers no es un entero.
            ValueError: Si el parámetro unfreezed_convolutional_layers parameter no es un número positivo (>= 0).
            ValueError: Si no se conoce el modelo base.

        """
        self._model_name = base_model
        self._initialize_attributes()

        input_shape = self._target_size + (3,)

        # Inicializamos el modelo base. Cargamos los pesos de la red de disco.
        # NOTA: Si es la primera vez que ejecutas esta función, los pesos se descargarán de Internet.
        self._model = getattr(tf.keras.applications, base_model)(weights='imagenet', include_top=include_top,
                                                              input_shape=input_shape, pooling=pooling)

        # Capas convolucionales "congeladas"
        if type(unfreezed_convolutional_layers) != int:
            raise TypeError("unfreezed_convolutional_layers must be a positive integer.")

        if unfreezed_convolutional_layers == 0:
            freezed_layers = self._model.layers
        elif unfreezed_convolutional_layers > 0:
            freezed_layers = self._model.layers[:-unfreezed_convolutional_layers]
        else:
            raise ValueError("unfreezed_convolutional_layers must be a positive integer.")

        for layer in freezed_layers:
            layer.trainable = False

    def _initialize_attributes(self):
        """Inicializamos el input shape de la imagen con la función de pre-processing.

        Raises:
            ValueError: Si el modelo es desconocido.

        """
        if self._model_name in ('DenseNet121', 'DenseNet169', 'DenseNet201'):
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.densenet.preprocess_input
        elif self._model_name == 'InceptionResNetV2':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.inception_resnet_v2.preprocess_input
        elif self._model_name == 'InceptionV3':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.inception_v3.preprocess_input
        elif self._model_name == 'MobileNet':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
        elif self._model_name == 'MobileNetV2':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
        elif self._model_name == 'NASNetLarge':
            self._target_size = (331, 331)
            self._preprocessing_function = tf.keras.applications.nasnet.preprocess_input
        elif self._model_name == 'NASNetMobile':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.nasnet.preprocess_input
        elif self._model_name == ('ResNet50'):
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.resnet50.preprocess_input
        elif self._model_name == ('ResNet101'):
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras_applications.resnext.resnext101.preprocess_input
        elif self._model_name == ('ResNet50V2'):
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.resnet_v2.preprocess_input
        elif self._model_name == 'VGG16':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.vgg16.preprocess_input
        elif self._model_name == 'VGG19':
            self._target_size = (224, 224)
            self._preprocessing_function = tf.keras.applications.vgg19.preprocess_input
        elif self._model_name == 'Xception':
            self._target_size = (299, 299)
            self._preprocessing_function = tf.keras.applications.xception.preprocess_input
        else:
            raise ValueError("Base model not supported. Possible values are 'DenseNet121', 'DenseNet169', "
                             "'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', "
                             "'NASNetLarge', 'NASNetMobile', 'ResNet50', 'VGG16', 'VGG19' and 'Xception'.")

    def _add_output_layers(self, class_count: int, fc_layer_size: int = 1024):
        """Agregamos una neurona superficial completamente conectada con el output de la capa softmax al final del modelo base.

        Args:
          class_count: Número de clases (i.e., número de neuronas softmax).
          fc_layer_size: Número de neuronas en la capa oculta

        """
        # Creamos un modelo nuevo
        model = tf.keras.models.Sequential()

        # Añadimos el modelo convolucional base
        model.add(self._model)

        # Añadimos nuevas capas
        model.add(tf.keras.layers.Dense(fc_layer_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(class_count, activation='softmax'))

        # Asignamos el modelo al atributo de clase
        self._model = model

    @staticmethod
    def _plot_training(history):
        """Graficamos la evolución del accuracy y función de pérdida, tanto del set de entrenamiento como del set de validación.

        Args:
            history: Historial de entrenamiento.

        """
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(training_accuracy))

        # Accuracy
        plt.figure()
        plt.plot(epochs, training_accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, validation_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
