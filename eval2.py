import model_q2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np

img_width, img_height = 224, 224
valid_data_dir = '/data/datasets/rbonatti/data_processed/2/valid'
batch_size = 1
val_samples=10000

if __name__ == "__main__":

	network = model_q2.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
	network.load_weights('/data/datasets/rbonatti/ml_weights2/weights.25-1.04.hdf5')

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	network.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

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

	predictions=np.argmax(predictions,axis=1)
	predictions.astype(int)
	a=np.array([1])
	predictions=predictions+a

	np.savetxt('/data/datasets/rbonatti/ml_prediction_q2.out', predictions, delimiter=',')