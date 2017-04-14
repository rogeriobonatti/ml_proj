from keras.optimizers import SGD
import model

if __name__ == "__main__":

    # Test pretrained model
    network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    network.compile(optimizer=sgd, loss='categorical_crossentropy')
    print "YEY"