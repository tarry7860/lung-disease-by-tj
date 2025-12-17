import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, VGG16, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
from preprocessing import apply_preprocessing
import pickle 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "dataset_split") 
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

MODELS_TO_TRAIN = {
    'InceptionResNetV2': InceptionResNetV2,
    'DenseNet121': DenseNet121,
    'VGG16': VGG16,
    'Xception': Xception
}

def setup_environment():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" GPU Detected: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)

def cosine_decay_schedule(epoch, lr):
    if epoch < 5:
        return lr
    return lr * 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (EPOCHS - 5)))

def build_data_generators():
    print("\n Loading Data Generators...")
    
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_preprocessing,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(preprocessing_function=apply_preprocessing)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED
    )
    
    val_generator = test_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_specific_model(model_name, model_class, num_classes):
    print(f"\n Building {model_name}...")
    
    
    base_model = model_class(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_single_model(model_name, model_class, train_gen, val_gen):
    print(f"\n{'='*40}")
    print(f" TRAINING MODEL: {model_name}")
    print(f"{'='*40}")
    
    model, base_model = build_specific_model(model_name, model_class, train_gen.num_classes)
    
    checkpoint_name = f"model_{model_name}.keras"
    history_name = f"history_{model_name}.pkl" 
    
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    lr_scheduler = LearningRateScheduler(cosine_decay_schedule, verbose=1)
    
    
    print(f"Phase 1: Frozen Base ({model_name})")
    history_phase1 = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[checkpoint], verbose=1)
    
    
    print(f"Phase 2: Fine-Tuning ({model_name})")
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_phase2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=10,
        validation_data=val_gen,
        callbacks=[checkpoint, lr_scheduler],
        verbose=1
    )

    
    full_history = {key: history_phase1.history[key] + history_phase2.history[key] 
                    for key in history_phase1.history}
    with open(history_name, 'wb') as f:
        pickle.dump(full_history, f)
    
    print(f" Finished training {model_name}. Saved to {checkpoint_name} and history to {history_name}")

def main():
    if not os.path.exists(WORK_DIR):
        print(" Dataset not found! Please run 'train.py' first to split the dataset.")
        return

    setup_environment()
    train_gen, val_gen = build_data_generators()
    
    for name, cls in MODELS_TO_TRAIN.items():
        train_single_model(name, cls, train_gen, val_gen)

if __name__ == "__main__":
    main()