import model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np

img_width, img_height = 224, 224
# valid_data_dir = '/data/datasets/rbonatti/data_processed/1/valid'
valid_data_dir = '/data/datasets/rbonatti/data_processed/data_processed_test/test'
nb_validation_samples = 10000
batch_size = 10
val_samples=10000

if __name__ == "__main__":

	network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
	network.load_weights('/data/datasets/rbonatti/ml_weights/weights.25-0.00.hdf5')

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	network.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	# prepare data augmentation configuration
	datagen = ImageDataGenerator(rescale=1. / 255)

	valid_generator = datagen.flow_from_directory(
	    valid_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode=None,
	    color_mode='grayscale',
	    shuffle=False)

	predictions=network.predict_generator(
	    generator=valid_generator,
	    val_samples=val_samples
	    )

	predictions[predictions>=0.5]=1
	predictions[predictions<0.5]=2
	np.savetxt('/data/datasets/rbonatti/ml_prediction_augmented_test1.out', predictions, delimiter=',')