from keras.optimizers import Adam

import model

if __name__ == "__main__":

    # Test pretrained model
    network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    network.compile(optimizer=adam, loss='categorical_crossentropy')

    