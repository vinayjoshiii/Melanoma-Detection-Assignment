# Melanoma-Detection-Assignment

Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

CNN Architecture Design
I have built a custom CNN model with various regularisation techniques to classify skin cancer using skin lesion images and achieve higher accuracy in the classification task.

Rescaling Layer:

Rescales the input image pixel values from the original [0, 255] range to the [0, 1] range. This normalization step helps the model converge more effectively during training.
Data Augmentation (Preprocessing):

Data augmentation is applied using random flipping, rotation, zooming, width/height shifting, brightness, and contrast changes to expand the training dataset artificially. This helps prevent overfitting and makes the model more robust to variations in input images.
Convolutional Layers:

The model consists of four convolutional blocks. Each block applies a convolution operation to the input to extract important features like edges, shapes, and textures.
The kernel sizes vary (e.g., (3, 3), (11, 11)) to capture both fine and coarse details in the images.
Activation Function: Each convolutional layer uses the ReLU (Rectified Linear Unit) activation function, which helps overcome the vanishing gradient problem.
Batch Normalization:

Used after each convolutional layer to normalize the activations. This stabilizes and speeds up training, allowing the model to learn more effectively.
Pooling Layers:

Max pooling layers are used after each convolutional block to reduce the dimensions of the feature maps, preserving the most important features while reducing the computational cost.
Dropout Layers:

Applied after each pooling layer and in the fully connected layers to randomly drop units during training. This helps prevent overfitting by ensuring that the model doesn't become overly reliant on any specific set of neurons.
Different dropout rates (e.g., 0.25, 0.5) are used to balance regularization strength throughout the model.
Global Average Pooling Layer:

Reduces the spatial dimensions of the feature maps by taking the average of each feature map. This creates a more compact representation of the features, connecting directly to the dense (fully connected) layers.
Flatten Layer:

Converts the 3D output from the final convolutional block into a 1D feature vector. This vector can then be fed into the fully connected (dense) layers for final classification.
Dense (Fully Connected) Layers:

A series of dense layers with varying numbers of neurons (e.g., 256, 128, 64) is added to combine the features extracted by the convolutional layers.
Each dense layer uses the ReLU activation function, enabling the model to learn complex patterns from the feature vector.
Output Layer (Softmax Activation):

The final dense layer has len(class_names) neurons, each corresponding to one of the classes in the dataset.
The softmax activation function is used in the output layer to predict a probability distribution over the 9 classes. This ensures the sum of all predicted probabilities equals 1, providing a clear classification outcome.
Key Regularization Techniques:
Dropout: Included throughout the network to reduce overfitting.
Batch Normalization: Applied to normalize activations, stabilize learning, and further act as a regularizer.
This model combines data augmentation, regularization (dropout, batch normalization), and a carefully structured series of convolutional blocks to handle the complexity of skin lesion classification effectively.

Final Architecture: 

![image](https://github.com/user-attachments/assets/3b6114e5-3574-4e84-a37b-40b907177c5c)

Performance of the final model:
![image](https://github.com/user-attachments/assets/3d0b9306-0044-48a7-9f9e-4bd00296a1d9)

