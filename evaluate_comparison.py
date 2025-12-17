import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing import apply_preprocessing

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "dataset_split")
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

MODELS = ['InceptionResNetV2', 'DenseNet121', 'VGG16', 'Xception']

def evaluate_all():
    if not os.path.exists(WORK_DIR):
        print(" Dataset not found.")
        return

    print(" Preparing Test Data...")
    test_datagen = ImageDataGenerator(preprocessing_function=apply_preprocessing)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(WORK_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    print("\n Evaluating Models & Generating Figure 12 Replica...")
    
    for i, model_name in enumerate(MODELS):
        model_path = f"model_{model_name}.keras"
        
        if not os.path.exists(model_path):
            print(f" Model {model_name} not found at {model_path}. Skipping.")
            continue
            
        print(f"   -> Processing {model_name}...")
        model = tf.keras.models.load_model(model_path)
        
        
        test_generator.reset()
        preds = model.predict(test_generator, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        
        
        cm = confusion_matrix(y_true, y_pred)
        
       
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels, 
                    ax=axes[i], annot_kws={"size": 14})
        
        axes[i].set_title(f"{model_name}", fontsize=16)
        axes[i].set_ylabel('True label')
        axes[i].set_xlabel('Predicted label')
        
       
        print(f"\n--- {model_name} Report ---")
        print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
        
    plt.tight_layout()
    plt.savefig('figure_12_replica.png')
    print("\n Comparison Figure saved as 'figure_12_replica.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_all()
