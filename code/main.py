import tensorflow as tf
from model import ResNetModel



def main():
    resnet = ResNetModel()

    # Create the ResNet model
    resnet_model = resnet.create_resnet_model((128,128,3))
    resnet_model.summary()
    resnet_model.train_model()

if __name__ == "__main__":
    main()