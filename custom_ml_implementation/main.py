from custom_ml_implementation.LinearRegression import LinearRegression
from custom_ml_implementation.LogisticRegression import LogisticRegression
from custom_ml_implementation.NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np

train_or_load = 'load'

def main():
    # Test the XOR dataset
    df = pd.read_csv('./custom_ml_implementation/datasets/xor.csv', sep=',')
    features = df[['x', 'y']].values
    labels = df['label'].values
    EPOCHS = 2500
    LEARNING_RATE = 0.5
    REGULARIZATION = 0.5
    nn = NeuralNetwork(2, 1)
    if(train_or_load=='train'):
        nn.addHiddenLayer(4)
        nn.train(np.array(features), np.array(labels),
                LEARNING_RATE, REGULARIZATION, EPOCHS)
        like_or_nah = input("Do you like the results? 1/0")
        if(like_or_nah=='1'):
            nn.save_model('xor_model')
    else:
        nn.load_model('xor_model')    
    nn.print_weights()
    print("Cost: ", nn.cost_function(np.array(features),np.array(labels)))
    print(nn.predict([0.001, 0.001]), "Expected 0")
    print(nn.predict([0.001, 1.0]), "Expected 1")
    print(nn.predict([1.0, 0.001]), "Expected 1")
    print(nn.predict([1.0, 1.0]), "Expected 0")
    

if __name__ == "__main__":
    main()
