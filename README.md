# Computer Vision for Elementary Education
> #### _Archit, Chad, Iqra | Spring '23 | Duke AIPI 540 Computer Vision Project_
&nbsp;

## Project Description
Computer vision is an interdisciplinary field that deals with enabling machines to interpret and understand visual information in the same way that humans do. In the context of elementary education, computer vision can have a huge impact on the way students learn and interact with technology. By using computer vision algorithms, educational software and devices can analyze and interpret images and videos, providing a wealth of interactive and engaging opportunities for young children. 

For example, computer vision can be used to develop interactive educational games that help students learn various things as they grow up, such as names of animals, basic math, counting objects, etc. The technology can also be integrated into educational toys and devices, making it easier for students to visualize complex concepts and see them in action. Overall, computer vision has the potential to revolutionize the way small kids learn and interact with technology, and its integration into elementary education will play a significant role in preparing students for the technology-driven world of tomorrow.

The goal of this project is to create prototype modules that teach children some elementary education using computer vision. This allows kids to learn in a fun way using their toys.

&nbsp;
&nbsp;
## Running the demo (StreamLit)

**1. Clone this repository and switch to the streamlit-demo branch**
```
git clone https://github.com/iqra0908/Computer-Vision-for-Elementary-Education
git checkout streamlit-demo
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
**4. Run the application**
```
streamlit run streamlit_app.py
```
**5. StreamLit Appication:**
* Here you can play around with the streamlit demo 
>![img.png](data/images/dashboard.png)

&nbsp;
# References

1. Learning to Count Everything | [Viresh Ranjan1, Udbhav Sharma, Thu Nguyen, Minh Hoai](https://arxiv.org/pdf/2104.08391.pdf)   
_Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)_, 2021. 

2. Latest on Object Counting | [Papers with Code](https://paperswithcode.com/task/object-counting/codeless#datasets)

3. MediaPipe ML Module | [Google](https://google.github.io/mediapipe/solutions/hands.html)

4. Animal Toys Dataset | [Amazon](https://www.amazon.com/Fisher-Price-Little-People-Animal-Friends/dp/B07MM6QX97/ref=sr_1_2?crid=1CT3PFM11D3JC&keywords=animal+toys+little+people&qid=1676085377&sprefix=animal+toys+little+peop%2Caps%2C156&sr=8-2)

5. Dataset Labelling Tool| [Roboflow](https://roboflow.com/?ref=ultralytics)

6. YoloV5 Transfer Learning Package| [Ultralytics](https://github.com/ultralytics/yolov5)


