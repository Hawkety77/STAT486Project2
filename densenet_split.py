import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
# import cv2

# tf.config.threading.set_inter_op_parallelism_threads(8)
# tf.config.threading.set_intra_op_parallelism_threads(30)
random.seed(77)

images_dir = "training/training/"
labels = pd.read_csv("training_labels.csv")

# add the directory to the filename
labels['ID'] = labels['ID'].apply(lambda x: os.path.join(images_dir, x))

# Initialize the ImageDataGenerator

# def adjust_contrast(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     l_eq = cv2.equalizeHist(l)
#     lab_eq = cv2.merge((l_eq, a, b))
#     image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
#     return image_eq

# Note: adjust the brightness_range according to your needs
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip = True, 
    fill_mode='nearest',
    validation_split=0.2,
    rescale=1./255,
    # brightness_range=[0.2, 1.2] 
    # preprocessing_function=adjust_contrast
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
)

# Create the training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # You can change the size of the image
    batch_size=32, # You can change the batch_size
    class_mode='categorical',  
    subset='training', 
    seed = 77, 
    shuffle = True
)

validation_generator = datagen_val.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # Should match training size
    batch_size=32, # Should match training
    class_mode='categorical',  
    subset='validation', 
    seed = 77, 
    shuffle = True
)

### VGG16 ###

# base_model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# unfreezed_layers = 2

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# for layer in base_model.layers[(len(base_model.layers) - unfreezed_layers):]:
#     layer.trainable = True

# flatten_layer = tf.keras.layers.Flatten()
# dense_layer_1 = tf.keras.layers.Dense(50, activation='relu')
# dense_layer_2 = tf.keras.layers.Dense(20, activation='relu')
# prediction_layer = tf.keras.layers.Dense(5, activation='softmax')


# model = tf.keras.models.Sequential([
#     base_model,
#     flatten_layer,
#     dense_layer_1,
#     dense_layer_2,
#     prediction_layer
# ])

### Resnet ###

# base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# unfreezed_layers = 2

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# for layer in base_model.layers[(len(base_model.layers) - unfreezed_layers):]:
#     layer.trainable = True

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense_layer = tf.keras.layers.Dense(1024, activation='relu')
# prediction_layer = tf.keras.layers.Dense(5, activation='softmax')  # num_classes should be set to the number of your classes

# # Combine the base model and the custom layers using a Sequential model
# model = tf.keras.models.Sequential([
#     base_model,
#     global_average_layer,
#     dense_layer,
#     prediction_layer
# ])

### EfficientNet ###

# base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top= False, input_shape= (224, 224, 3), pooling= 'max')

# #block7a_expand_conv
# for layer in base_model.layers:
#   if layer.name == 'block7d_project_conv':
#     break
#   layer.trainable = False
#   #print('Layer ' + layer.name + ' frozen.')     

# x = tf.keras.layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)(base_model.output)
# x = tf.keras.layers.Dense(256, kernel_regularizer= tf.keras.regularizers.l2(l= 0.016), activity_regularizer= tf.keras.regularizers.l1(0.006),
#                 bias_regularizer= tf.keras.regularizers.l1(0.006), activation= 'relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.1),
#                      bias_initializer=tf.keras.initializers.Zeros())(x)
# x = tf.keras.layers.Dropout(rate= 0.45, seed= 123)(x)                
# x = tf.keras.layers.Dense(5, activation= 'softmax')(x) 

# model = tf.keras.models.Model(base_model.input, x)

### Mobile Net ### Best

# base_model= tf.keras.applications.MobileNet(
#     weights='imagenet',  # Load weights pre-trained on ImageNet.
#     input_shape=(224, 224, 3),
#     include_top=False)  # Do not include the ImageNet classifier at the top.

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# unfrozen_layers = 0

# for layer in base_model.layers[(len(base_model.layers) - unfrozen_layers):]:
#     layer.trainable = True

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense_layer = tf.keras.layers.Dense(512, activation='relu')
# prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')

# model = tf.keras.models.Sequential([
#     base_model,
#     global_average_layer,
#     dense_layer,
#     prediction_layer
# ])

# ### Inception ###

# base_model= tf.keras.applications.InceptionResNetV2(
#     weights='imagenet',  # Load weights pre-trained on ImageNet.
#     input_shape=(224, 224, 3),
#     include_top=False)  # Do not include the ImageNet classifier at the top.

# base_model.trainable = False

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense_layer = tf.keras.layers.Dense(1024, activation='relu')
# prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')

# model = tf.keras.models.Sequential([
#     base_model,
#     global_average_layer,
#     dense_layer,
#     prediction_layer
# ])

### Densenet ###

base_model= tf.keras.applications.DenseNet201(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

unfrozen_layers = 0

for layer in base_model.layers[(len(base_model.layers) - unfrozen_layers):]:
    layer.trainable = True

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense_layer = tf.keras.layers.Dense(1024, activation='relu')
prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')

model = tf.keras.models.Sequential([
    base_model,
    global_average_layer,
    dense_layer,
    prediction_layer
])

### Convnexttiny ###

# base_model= tf.keras.applications.ConvNeXtTiny(
#     weights='imagenet',  # Load weights pre-trained on ImageNet.
#     input_shape=(224, 224, 3),
#     include_top=False)  # Do not include the ImageNet classifier at the top.

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # unfrozen_layers = 0

# # for layer in base_model.layers[(len(base_model.layers) - unfrozen_layers):]:
# #     layer.trainable = True

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense_layer = tf.keras.layers.Dense(1024, activation='relu')
# prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')

# model = tf.keras.models.Sequential([
#     base_model,
#     global_average_layer,
#     dense_layer,
#     prediction_layer
# ])


## ResNet152 ###

# base_model = tf.keras.applications.ResNet152V2(include_top=False,
#                   input_shape=(224,224,3),
#                    weights = 'imagenet',
#                     pooling='avg'
#                   )
# base_model.trainable = False

# dense_layer = tf.keras.layers.Dense(128,activation='relu')
# prediction_layer = tf.keras.layers.Dense(5,activation='softmax')

# model = tf.keras.models.Sequential([
#     base_model,
#     dense_layer,
#     prediction_layer
# ])

### NasNetLarge ###

# base_model= tf.keras.applications.NASNetLarge(
#     weights='imagenet',  # Load weights pre-trained on ImageNet.
#     input_shape=(224, 224, 3),
#     include_top=False)  # Do not include the ImageNet classifier at the top.

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # unfrozen_layers = 0

# # for layer in base_model.layers[(len(base_model.layers) - unfrozen_layers):]:
# #     layer.trainable = True

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense_layer = tf.keras.layers.Dense(1024, activation='relu')
# prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')

# model = tf.keras.models.Sequential([
#     base_model,
#     global_average_layer,
#     dense_layer,
#     prediction_layer
# ])

#############

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy',  # Change this according to your problem
              optimizer=optimizer,
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=10, 
                                                # min_delta=0.001, 
                                                mode='min', 
                                                restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50, 
    callbacks=[early_stopping]
)

### Fine tune ###

base_model.trainable = True
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=5, 
                                                min_delta=0.001, 
                                                mode='min', 
                                                restore_best_weights=True)

history2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50, 
    callbacks=[early_stopping]
)

### Test ###

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'testing/',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Since you're predicting, there are no labels
    shuffle=False  # Keep data in the same order as in the directory
)

predictions = model.predict(test_generator)

predicted_class_indices = np.argmax(predictions, axis=1)

# Get filenames (keeping the '.jpg' extension)
filenames = test_generator.filenames
filenames = [filename.split('/')[-1] for filename in filenames]  # Keeps the full filename with '.jpg'

# Get class names for each index, from train_generator's class_indices
class_names = list(train_generator.class_indices.keys())
predicted_class_names = [class_names[idx] for idx in predicted_class_indices]

# Create a DataFrame
results_df = pd.DataFrame({"ID": filenames, "PredictedClass": predicted_class_names})

results_df.to_csv(r'submissions/submission_split.csv', index = False)

# Save Model 
model.save('models/dense_split.h5')