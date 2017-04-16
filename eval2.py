import model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import cv2

img_width, img_height = 224, 224
valid_data_dir = '/data/datasets/rbonatti/data_processed/1/valid'
nb_validation_samples = 10000
batch_size = 10
val_samples=10000

if __name__ == "__main__":

	network = model.VGG_16('/data/datasets/rbonatti/vgg16_weights_with_name.h5')
	network.load_weights('/data/datasets/rbonatti/ml_weights/weights.31-0.00.hdf5')

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	network.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	predictions=np.array([])
	for i in range(10000):
im_path=valid_data_dir+'/no_class/'+str(i+1)+'_orig.png'
im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(im, (img_height, img_width))
resized_image= np.reshape(im,(1,img_height,img_width))
resized_image = np.expand_dims(resized_image, axis=0)
pred_out=network.predict(resized_image,batch_size=1)
predictions=np.append(predictions,pred_out)




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

	np.savetxt('/data/datasets/rbonatti/ml_prediction.out', predictions, delimiter=',')