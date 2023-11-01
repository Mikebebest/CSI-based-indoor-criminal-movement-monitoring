import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.optimizers import Adam


#load the data and resize 
img_size = [600,600]
def load_images_from_folder(folder, img_size):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, img_size)
            images.append(img_resized)

    return images


#load the respective data and label with anchor, positive, negative
def prepare_siamese_data(base_path, img_size=(600,600)):

    subpaths = ["anchor", "positive", "negative"]
    data = {}

    for sub in subpaths:
        images = load_images_from_folder(os.path.join(base_path, sub), img_size)

        # Convert to numpy array
        images_np = np.array(images)

        # Save the numpy array to the given path
        np.save(os.path.join(base_path, f"{sub}_data.npy"), images_np)

        data[sub] = images_np

    return data['anchor'], data['positive'], data['negative']



# Example usage:
base_path = "./Data/train"
train_anchor, train_positive, train_negative = prepare_siamese_data(base_path)

train_anchor  = train_anchor.reshape(-1, 600, 600, 3).astype('float32')
train_positive = train_positive.reshape(-1, 600, 600, 3).astype('float32')
train_negative = train_negative.reshape(-1, 600, 600, 3).astype('float32')


#normalize
train_anchor = train_anchor / 255
train_positive = train_positive / 255
train_negative = train_negative / 255

print(train_anchor.shape)
print(train_positive.shape)
print(train_negative.shape)

#check for the correct size
print(train_anchor.shape)
print(train_positive.shape)
print(train_negative.shape)


input_shape = (600, 600, 3)

#CNN-based share network 
def Triplet_network(inputshape):

    input_layer = Input(shape = input_shape)
    c_1 = Conv2D(32, (5,5), activation = 'relu')(input_layer)
    p_1 = MaxPooling2D((2,2))(c_1)
    c_2 = Conv2D(64, (5,5), activation = 'relu')(p_1)
    p_2 = MaxPooling2D((2,2))(c_2)
    d =  Dropout(0.25)(p_2)
    flatten = Flatten()(d)
    embedding = Dense(128, activation='relu')(flatten)

    model = Model(input_layer, embedding)
    return model


siamese_network = base_network(input_shape)

# Siamese network inputs
input_anchor  = Input(shape=input_shape)
input_positive = Input(shape=input_shape)
input_negative = Input(shape=input_shape)

# Apply the siamese network to the inputs
embedding_anchor  = siamese_network(input_anchor)
embedding_positive = siamese_network(input_positive)
embedding_negative = siamese_network(input_negative)

output = Concatenate()([embedding_anchor, embedding_positive,embedding_negative])


#Loss fuction of computing Euclidean distance 
def triplet_loss(y_true, y_pred, margin = 1):

    anchor, positive, negative = y_pred[:,0:128], y_pred[:,128:256], y_pred[:,256:]

    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1, keepdims=True)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1, keepdims=True)
    base_loss = positive_dist - negative_dist + margin
    loss = tf.maximum(base_loss, 0.0)

    return loss

# Siamese network model
siamese_model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=output )

# Compile the model
siamese_model.compile(loss= triplet_loss, optimizer=Adam(learning_rate=0.00001))

# Create labels for the contrastive loss
#positive_labels = np.ones((train_positive.shape[0], 1))
#negative_labels = np.zeros((train_negative.shape[0], 1))

dummy_y = np.zeros((train_anchor.shape[0], 128*3))


#train the model 
train_history = siamese_model.fit([train_anchor, train_positive, train_negative], dummy_y,
                validation_split=0.2, batch_size = 32, epochs= 200)

#save model
siamese_model.summary()


#from tensorflow.keras.models import load_model

# Assuming your model structure is saved in 'siameseCNN_model.h5'
# and weights in 'siameseCNN_model.weight'
#model_path = 'siameseCNN_model.h5'
#weights_path = 'siameseCNN_model.weight'

# Load your trained model
#siamese_model = load_model(model_path, custom_objects={"triplet_loss": triplet_loss})
#siamese_model.load_weights(weights_path)

# Assuming you have a test dataset loader function like before
# For instance: prepare_siamese_data but for test dataset
test_base_path = "./Data/test"
test_anchor, test_positive, test_negative = prepare_siamese_data(test_base_path)

test_anchor  = test_anchor.reshape(-1, 600, 600, 3).astype('float32')
test_positive = test_positive.reshape(-1, 600, 600, 3).astype('float32')
test_negative = test_negative.reshape(-1, 600, 600, 3).astype('float32')


# Use model to get predictions on test set
predictions = siamese_model.predict([test_anchor, test_positive, test_negative])
anchor, positive, negative = predictions[:, 0:128], predictions[:, 128:256], predictions[:, 256:]

positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

positive_dist = positive_dist.numpy()
negative_dist = negative_dist.numpy()

# If the distance between the anchor and the positive image is less
# than the distance between the anchor and the negative image,
# then it's a correct (similar) prediction, so label it as 1. Otherwise, label as 0.
predicted_labels = np.where(positive_dist < negative_dist, 1, 0)

# Given your folder structure, the ground truth would be:
true_labels = np.ones(len(test_anchor)) # Because all anchors are paired with positive samples in your case

# Compute accuracy
accuracy = np.mean(predicted_labels == true_labels)

# Compute F1 Score
f1 = f1_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")


direc_path = './Data/confusion_matrix'
#computing confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

#draw the confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(direc_path,'Confusion Matrix_AA.jpg'))
plt.show()
