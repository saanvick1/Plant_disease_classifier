Potato Leaves Classification Project
This project focuses on classifying potato leaves using Jetson Inference and the Imagente dataset. The goal is to develop a model that can accurately classify different types of potato leaves, which can be useful in agricultural applications for disease detection and crop management.

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
The potato leaves classification model used in this project is based on a imagenet architecture trained using the above dataset. The model achieves accurate classification results and can be used for inference on new potato leaf images. The model is a part of jetson-infernce so giving you access to other pre - trained neural networks. 

Dataset
The Imagente dataset used in this project contains various classes of potato leaves, including healthy leaves and leaves affected by different diseases or pests. The dataset is labeled, providing ground truth for each image. Make sure to adhere to the dataset's licensing and usage terms.

Acknowledgments
The Jetson Inference library: https://github.com/dusty-nv/jetson-inference
The  dataset: [https://www.kaggle.com/datasets/alyeko/potato-tomato-dataset]


Contact
If you have any questions or need further assistance, feel free to contact Saanvi Chakraborty at chakrs12@gmail.com.

Model
The potato leaves classification model used in this project is based on a imagenet architecture trained using the above dataset. The model achieves accurate classification results and can be used for inference on new potato leaf images. The model is a part of jetson-infernce so giving you access to other pre - trained neural networks. 
