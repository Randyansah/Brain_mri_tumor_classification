from lib.load_dataset import load_dataset
from lib.train import train_model
from lib.predict import predict


def main():
    x_train,x_val,x_test,y_train,y_val,y_test=load_dataset('./Data/Training','./Data/Testing')
    train_model(x_train,y_train,x_val,y_val,200)
    predict('MRI_brain.keras',"./","Te-no_0011.jpg")

if __name__=='__main__':
    main()