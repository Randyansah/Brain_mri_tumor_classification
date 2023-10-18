from lib.load_dataset import load_dataset
from lib.train import train_model
from lib.predict import predict


def main():
    x_train,y_train,x_val,y_val=load_dataset('./Data/Training','./Data/Testing')
    train_model(x_train,y_train,x_val,y_val)
    predict("MRI_brain.keras")

if __name__=='__main__':
    main()