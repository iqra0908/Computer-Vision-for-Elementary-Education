# Computer Vision for Elementary Education
> #### _Archit, Chad, Iqra | Spring '23 | Duke AIPI 540 Computer Vision Project_
&nbsp;

## Project Description
Computer vision is an interdisciplinary field that deals with enabling machines to interpret and understand visual information in the same way that humans do. In the context of elementary education, computer vision can have a huge impact on the way students learn and interact with technology. By using computer vision algorithms, educational software and devices can analyze and interpret images and videos, providing a wealth of interactive and engaging opportunities for young children. 

For example, computer vision can be used to develop interactive educational games that help students learn various things as they grow up, such as names of animals, basic math, counting objects, etc. The technology can also be integrated into educational toys and devices, making it easier for students to visualize complex concepts and see them in action. Overall, computer vision has the potential to revolutionize the way small kids learn and interact with technology, and its integration into elementary education will play a significant role in preparing students for the technology-driven world of tomorrow.

The goal of this project is to create prototype modules that teach children some elementary education using computer vision. This allows kids to learn in a fun way using their toys.
>![img.png](data/images/toys.png)

&nbsp;
## Modules

&nbsp;
### Module 1: Learning Animals (Object detection & Classification)
The animal learning module is a fusion of object detection and classification, designed to teach kids the names of various animals through detecting animal toys in an image. Utilizing computer vision algorithms, the module identifies and locates the animal toys in the image and classifies each toy into its appropriate animal category. This module offers a captivating and interactive approach for kids to learn about different animals and their unique characteristics, providing a hands-on experience that enriches their knowledge of the animal kingdom.

&nbsp;
![img.png](data/images/module1.png)

&nbsp;
### Module 2: Learning math using toys (Object Counting)
This module will count objects i.e. toys in a given image. This can be used to teach kids counting and basic math. To teach counting, a kid can add or remove toys from the live camera frame and count will go up and down respectively. To teach math, a question will pop up such as 2+2 and the kid will place that many toys in the picture so computer can respond if the answer (i.e. frame) is correct or not. Currently we are using a Yolo model to identify the count of objects by performing object detection. In future we would like to move our focus towards unsupervised object counting, as then we would be able to count aby kind of objects and not just toys.

This link has a few papers on object counting along with datasets to use;
https://paperswithcode.com/task/object-counting/codeless#datasets

&nbsp;
![img.png](data/images/module2.png)

&nbsp;
### Module 3: Simple Finger Math
This module for slightly older children will ask simple arithmetic questions with answers between 0 and 10, and the answers provided on webcam by how many fingers are raised. The fingers must be visible on the webcam with the palms facing the camera and the hands positioned 'upright' such that a raised finger tip is higher than the knuckle. There will be a slider control for the number of seconds allowed, and choices for addition, subtraction, or mixed math problems. The outer shell test_finger.py imports get_finger_count from the finger_counter_webcam2 module. At this point, the arithmetic questions in the outer shell are crudely built to provide addition or subtraction problems (default is 5 addition problems but variables for num_probs, which_type, and countdown length in seconds could be set by the streamlit interface). Currently it is set to NOT draw the landmarks on the hands, but could be modified through a variable assignment in finger_counter_webcam2.

&nbsp;
![img.png](data/images/module3.png)


&nbsp;
## Data Capturing & Labelling
We have collected the data ourselves for the purpose of this project prototype. We bought some animal toys of [Amazon.](https://www.amazon.com/Fisher-Price-Little-People-Animal-Friends/dp/B07MM6QX97/ref=sr_1_2?crid=1CT3PFM11D3JC&keywords=animal+toys+little+people&qid=1676085377&sprefix=animal+toys+little+peop%2Caps%2C156&sr=8-2). There are eight unique animal toys in our dataset - [Dog, Cow, Hen, Lama, Horse, Sheep, Goat, Pig]. Multiple images of each toy were captured at different angles, positions, light intenisty and with a variety of backgrounds. All the raw images were labelled with bounding boxes. We used [Roboflow](https://roboflow.com/?ref=ultralytics) to label and prepare custom data automatically in YOLO format. We considered examples of single and multi object images. All images were resized to Yolo v5 format

&nbsp;
## Model Training and Evaluation
We trained three models for comparison - ResNet18, YoloV5 and a Support Vector Machine:

| Model     |  Accuracy (Test) |
| --------- | :--------------: |
| Resnet18  |        80%       |
| Resnet50  |        84%       |

&nbsp;
| Model     |  MAP@0.5:0.95    |
| --------- | :--------------: |
| YoloV5    |        0.75      |


&nbsp;
### Model 1: Resnet18
We started with a simple image classifier - Resnet18. The labelled images were split into train and test datasets and loaded in Pytorch Dataset Objects. A pretrained Resnet18 model was loaded and its fully connected head was stripped off to perform transfer learning. The new trasnfer learned resnet takes in input images of size 640x640x3 and gives the predicted probability of presence of each class in the image.

&nbsp;
### Following are the steps to run the code and train a Resnet18 model:

**1. Create a conda environment:** 
```
conda create --name environ python=3.7.15
conda activate environ
```
**2. Install python requirements:** 
```
pip install -r requirements.txt 
```
**3. Update the model parameters** 
- You can tweak the model parameters in the configuration file stored at `ResnetTraining/config.py`

**4. Use the training driver notebook provided** 
- If you want to train the model on Google Colab, we have provided a Jupyter Driver notebook which imports the required python scripts from the `ResnetTraining` module and trains a ResNet model on your data
- The driver notebook is available at `Notebooks/Resnet18_Transfer_Learning.ipynb`

**5. To train the model using python scripts** 
- You can use alternatively train a model direcltly by runnning the driver python script for Resnet transfer learning: `ResnetTraining/resnet_transfer_learning.py`
```
python ./ResnetTraining/resnet_transfer_learning.py
```



&nbsp;
### Model 2: YoloV5
The second and much better model than our resnet we used was the Yolo architecture. YOLO (You Only Look Once) is a state-of-the-art object detection model that is designed for real-time predictions and is highly accurate. It is an end-to-end model that performs object detection in a single forward pass and has good generalization properties, making it well-suited for real-world applications. YOLO's combination of speed and accuracy has made it popular for various use cases. We have trained our custom Yolo V5 model, which is basically an implemetation of Yolo V3 architecture in pytorch instead of the original Darknet. To train a yolo v5 we used the set of instruction provided by [ultralytics/yolov5](https://github.com/ultralytics/yolov5), the creators of this pytorch varient.

&nbsp;
### Following are the steps to run the code and train a YoloV5 model:
**1. Clone YOLOv5 repository**
```
git clone https://github.com/ultralytics/yolov5
```
**2. Move to the downloaded repo and reset it:** 
```
%cd yolov5
!git reset --hard fbe67e465375231474a2ad80a4389efc77ecff99
```
**3. Install dependencies as necessary:** 
```
pip install -qr requirements.txt 
```
**4. Install Roboflow to generate yolo configs:** 
```
pip install -q roboflow
```
**5. Upload labelled data to google drive**
**6. Update the number of classes** 
- Modify number of classes according to your data in the following config file`YoloTraining/custom_yolo5s.yaml`
**6. Continue model training steps using the notebook provided** 
- Use the file `notebooks/YoloV5_Transfer_Learning.ipynb`

&nbsp;
### Model 3: SVM
The third model trained is a SVM (Support Vector Machine). SVMs are linear classifiers that perform poorly with image data due to its high-dimensional and complex nature. Deep learning models, such as Convolutional Neural Networks (CNNs) are better suited for image data as they can learn non-linear relationships, handle spatial invariance, capture hierarchical features, and incorporate contextual information. We just trained a SVM to proove that linear machine learning models dont perform well on image data.

&nbsp;
### Following are the steps to run the code and train a SVM model:

**1. Create a conda environment:** 
```
conda create --name environ python=3.7.15
conda activate environ
```
**2. Install python requirements:** 
```
pip install -r requirements.txt 
```
**3. Update the model parameters** 
- You can tweak the model parameters in the configuration file stored at `SVMTraining/config.py`

**4. Use the training driver notebook provided** 
- If you want to train the model on Google Colab, we have provided a Jupyter Driver notebook which imports the required python scripts from the `SVMTraining` module and trains a SVM Classifier on your data 
- The driver notebook is available at `Notebooks/SVM_Classifier.ipynb`

**5. To train the model using python scripts** 
- You can use alternatively train a model direcltly by runnning the driver python script for Resnet transfer learning: `SVMTraining/SVM_training.py`
```
python ./SVMTraining/SVM_training.py
```



&nbsp;
## Running the demo (StreamLit)

**1. Clone this repository and switch to the streamlit-demo branch**
```
git clone https://github.com/iqra0908/Computer-Vision-for-Elementary-Education
```
**2. Create a conda environment:** 
```
conda create --name environ python=3.7.16
conda activate environ
```
**3. Install requirements:** 
```
pip install -r requirements.txt
```
**4. Create a jupyter kernal from this environment:** 
```
python -m ipykernel install --user --name=environ
```
**5. Run the application**
```
streamlit run streamlit_app.py
```
**6. StreamLit Appication:**
* Here you can play around with the streamlit demo
>![img.png](data/images/dashboard.png)

## Citation

```
@inproceedings{m_Ranjan-etal-CVPR21,
  author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
  title = {Learning To Count Everything},
  year = {2021},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
https://arxiv.org/pdf/2104.08391.pdf


