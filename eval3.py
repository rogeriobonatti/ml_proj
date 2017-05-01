import model_q3
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.cluster import KMeans

img_width, img_height = 224, 224
valid_data_dir = '/data/datasets/rbonatti/data_processed/3'
batch_size = 1
val_samples=300

if __name__ == "__main__":

	network = model_q3.VGG_16('/data/datasets/rbonatti/ml_weights2/weights.25-2.47.hdf5')

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

	# predictions=np.argmax(predictions,axis=1)
	# predictions.astype(int)
	# a=np.array([1])
	# predictions=predictions+a

	scores=np.zeros(20)
for i in range(5):
	kmeans = KMeans(n_clusters=i+1).fit(predictions)
	scores[i]=kmeans.score(predictions)

	np.savetxt('/data/datasets/rbonatti/ml_prediction_q3.out', predictions, delimiter=',')