from Data import StartingDataset

(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
show_dataset_examples(train_images)
print (train_images.shape)

train_images_reshaped = np.expand_dims(train_images, 3)
train_images_scaled = (train_images_reshaped - 127.5)/127.5
