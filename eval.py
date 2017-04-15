import model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

img_width, img_height = 224, 224
valid_data_dir = '/data/datasets/rbonatti/data_processed/1/valid'
nb_validation_samples = 10000
batch_size = 16

if __name__ == "__main__":

	network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
	network.load_weights('/data/datasets/rbonatti/ml_weights/weights.00-0.76.hdf5')

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	network.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	# prepare data augmentation configuration
	datagen = ImageDataGenerator(rescale=1. / 255)

	valid_generator = datagen.flow_from_directory(
	    valid_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode='binary',
	    color_mode='grayscale')

	predictions=network.predict_generator(
	    generator=valid_generator,
	    validation_data=validation_generator
	    )

	np.savetxt('/data/datasets/rbonatti/ml_prediction.out', predictions, delimiter=',')