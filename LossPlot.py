import pandas as pd
import matplotlib.pyplot as plt

def load_and_plot_losses(file_path):
    data = pd.read_pickle(file_path)

    train_losses = data['train_losses']
    test_losses = data['test_losses']

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss', linestyle='--')
    plt.title('Training and Test Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

file_path = './losses.pkl'

load_and_plot_losses(file_path)
