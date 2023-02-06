# %%
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt 

# %%
print("Loading CIFAR10 dataset...")
(ds_cifar10_train, ds_cifar10_test), ds_cifar10_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    data_dir='/projectnb/ds549/datasets/tensorflow_datasets',
    shuffle_files=True, # load in random order
    as_supervised=True, # Include labels
    with_info=True, # Include info
)

# Optionally uncomment the next 3 lines to visualize random samples from each dataset
#fig_train = tfds.show_examples(ds_cifar10_train, ds_cifar10_info)
#fig_test = tfds.show_examples(ds_cifar10_test, ds_cifar10_info)
#plt.show()  # Display the plots

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

# Prepare cifar10 training dataset
ds_cifar10_train = ds_cifar10_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_cifar10_train = ds_cifar10_train.cache()     # Cache data
ds_cifar10_train = ds_cifar10_train.shuffle(ds_cifar10_info.splits['train'].num_examples)
ds_cifar10_train = ds_cifar10_train.batch(32)  # <<<<< To change batch size, you have to change it here
ds_cifar10_train = ds_cifar10_train.prefetch(tf.data.AUTOTUNE)

# Prepare cifar10 test dataset
ds_cifar10_test = ds_cifar10_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_cifar10_test = ds_cifar10_test.batch(32)    # <<<<< To change batch size, you have to change it here
ds_cifar10_test = ds_cifar10_test.cache()
ds_cifar10_test = ds_cifar10_test.prefetch(tf.data.AUTOTUNE)


# %%
# Define the model here
def base_model(samepadding, batchnormal, units, lr ):
    model = tf.keras.models.Sequential([keras.Input(shape=(32, 32, 3))])

    model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
    if samepadding:
        model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
    else:
        model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    if batchnormal:
        model.add(layers.BatchNormalization())
    else:
        model.add(layers.GroupNormalization(16))
    model.add(layers.Dropout(0.2))
    

    model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    if samepadding:
        model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    else:
        model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    if batchnormal:
        model.add(layers.BatchNormalization())
    else:
        model.add(layers.GroupNormalization(32))
    model.add(layers.Dropout(0.2))


    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    if samepadding:
        model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    else:
        model.add(layers.Conv2D(128, 3, activation='relu'))
    model.add(layers.MaxPooling2D())
    if batchnormal:
        model.add(layers.BatchNormalization())
    else:
        model.add(layers.GroupNormalization(64))
    model.add(layers.Dropout(0.2))


    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)

    return model

def build_model(hp):
    samepadding = hp.Boolean("samepadding")
    batchnormal = hp.Boolean("batchnormal")
    units = hp.Int("units", min_value=384, max_value=512, step=128)
    lr = hp.Float("lr", min_value=5e-4, max_value=1e-3, sampling="log")
    return base_model(samepadding, batchnormal, units, lr)
# print(build_model(keras_tuner.HyperParameters()).summary())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=keras_tuner.Objective("val_sparse_categorical_accuracy", direction="max"),
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    directory="/projectnb/ds549/students/fjgao/ml-549-course/homeworks/spring23_hw1",
    project_name="tuner_hw1",
)
# print(tuner.search_space_summary())
tuner.search(ds_cifar10_train, epochs=6, validation_data=ds_cifar10_test)

# %%
best_hp = tuner.get_best_hyperparameters(1)[0]
print(tuner.results_summary(1))


