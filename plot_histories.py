import os
import matplotlib.pyplot as plt
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = ['InceptionResNetV2', 'DenseNet121', 'VGG16', 'Xception']

def plot_model_history(model_name, history_data, save_dir='.'):
    """
    Plots the training and validation accuracy and loss for a given model.
    """
    epochs = range(len(history_data['accuracy']))

    plt.figure(figsize=(12, 5))
    
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history_data['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_data['loss'], label='Training Loss')
    plt.plot(epochs, history_data['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'plot_{model_name}.png')
    plt.savefig(plot_filename)
    print(f" Saved plot for {model_name} to '{plot_filename}'")
    # plt.show() # Uncomment it to display plots immediately

def main():
    print(" Generating Loss and Accuracy Plots for each model...")
    for model_name in MODELS:
        history_file = os.path.join(BASE_DIR, f'history_{model_name}.pkl')
        if os.path.exists(history_file):
            with open(history_file, 'rb') as f:
                history_data = pickle.load(f)
            plot_model_history(model_name, history_data, BASE_DIR)
        else:
            print(f" History file for {model_name} not found at {history_file}. Please run 'train_all_models.py' first.")

if __name__ == "__main__":
    main()
