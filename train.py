import model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 224, 224
train_data_dir = '/data/datasets/rbonatti/data_processed/1/train80'
validation_data_dir = '/data/datasets/rbonatti/data_processed/1/train20'
nb_train_samples = 7998*2
nb_validation_samples = 2002*2
epochs = 50
batch_size = 16

if __name__ == "__main__":

    # Test pretrained model
    network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    network.compile(optimizer=adam, loss='binary_crossentropy')

    # prepare data augmentation configuration
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale')

    # fine-tune the model
    network.fit_generator(
        generator=train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples
        )
        