Potato Leaves Classification Project
This project focuses on classifying potato leaves using Jetson Inference and the Imagente dataset. The goal is to develop a model that can accurately classify different types of potato leaves, which can be useful in agricultural applications for disease detection and crop management.
![Corn_Common_Rust (994)](https://github.com/saanvick1/Plant_disease_classifier/assets/95539800/15ed6071-0052-4486-bc1d-0ab71876031e)

Requirements
To run this project, you will need:

NVIDIA Jetson device (e.g., Jetson Nano, Jetson Xavier NX)
JetPack SDK installed on the Jetson device
Jetson Inference library installed on the Jetson device
Imagente dataset downloaded and prepared for training
Installation
Clone the Jetson Inference repository:


git clone https://github.com/dusty-nv/jetson-inference.git
Build the Jetson Inference library by following the instructions provided in the repository's README.

Download and prepare the https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset dataset. Refer to the dataset documentation for instructions on how to obtain and preprocess the dataset.Also this model is only trained on the potato leaves not the tomatoes. 

Usage
Once the Jetson Inference library is built, navigate to the jetson-inference/build/aarch64/bin directory.

Copy the prepared dataset to the jetson-inference/data/imagente directory.

Run the following command to train the potato leaves classification model:


Copy code
./imagenet --model=models/potato/potato-model.py --epochs=50 --data=imagente --batch=8 --seed=777
Adjust the parameters (--epochs, --batch, --seed, etc.) as needed for your specific configuration and requirements.

After training, the model will be saved to the jetson-inference/data/networks/potato directory.

To perform inference on new potato leaf images, use the following command:

Model



![image](https://github.com/saanvick1/Plant_disease_classifier/assets/95539800/cb3b9e43-06a1-40cb-8bd4-65ce35ea414e)





The potato leaves classification model used in this project is based on a imagenet architecture trained using the above dataset. The model achieves accurate classification results and can be used for inference on new potato leaf images. The model is a part of jetson-infernce so giving you access to other pre - trained neural networks. 
Imagenet 
ImageNet is a large-scale image dataset that is widely used in computer vision research. It contains millions of labeled images spanning thousands of object categories. The ImageNet project also includes the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which is an annual competition that evaluates the state-of-the-art algorithms for object detection and image classification.

ImageNet works by providing a vast collection of labeled images along with their corresponding class labels. These images cover a wide range of object categories, such as animals, vehicles, household items, and more. The dataset is carefully curated and labeled by human annotators to ensure accurate and consistent annotations.

ImageNet has been instrumental in the development and advancement of deep learning techniques, especially convolutional neural networks (CNNs), for image classification. CNNs are a type of neural network architecture that is particularly effective in processing visual data, such as images.

The typical workflow for using ImageNet involves training a CNN model on a subset of the dataset, usually referred to as the "training set." During training, the CNN learns to extract meaningful features from the input images and map them to the corresponding class labels. This is achieved by iteratively adjusting the network's weights through a process called backpropagation, which minimizes the difference between the predicted labels and the ground truth labels in the training data.

Once the CNN model is trained, it can be used for inference on new, unseen images. During inference, the model takes an input image and applies a series of convolutional and pooling operations to extract relevant features. These features are then passed through fully connected layers, which make the final classification predictions. The output of the model is a probability distribution over the possible classes, indicating the likelihood of the input image belonging to each class.

Dataset
The Imagente dataset used in this project contains various classes of potato leaves, including healthy leaves and leaves affected by different diseases or pests. The dataset is labeled, providing ground truth for each image. Make sure to adhere to the dataset's licensing and usage terms.

Acknowledgments
The Jetson Inference library: https://github.com/dusty-nv/jetson-inference
The  dataset: [https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset]


Contact
If you have any questions or need further assistance, feel free to contact Saanvi Chakraborty at chakrs12@gmail.com.


