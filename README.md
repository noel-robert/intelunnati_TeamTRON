###### intelunnati_TeamTRON

# Intel Unnati Project - Team TRON

This repository was created as part of Intel UNNATI Industrial Training Project, 2023. The topic of our project was "Road Object Detection with Deep Learning". <br>

<p> Our aim was to develop a model that could identify objects in roads, specifically Indian roads. The reason that other commonly available models could not work properly in this environment is due to the variety of objects found in Indian roads. <br><br>

Follow these steps to use this project:

1. Clone | fork this project - [intelunnati_TeamTRON](https://github.com/noel-robert/intelunnati_TeamTRON)

2. Download the dataset (_Indian Driving Dataset_) from [http://idd.insaan.iiit.ac.in](http://idd.insaan.iiit.ac.in) [_you will be asked to create an account_]. IDD Detection (_22.8 GB_) is the dataset being used in this case. <br> - Note that dataset directory here does not have _Annotations_  and _JPEGImages_ folders due to the large size of the dataset.

3. Extract downloaded dataset and place into TeamTRON_MarBaseliosCollegeOfEngineeringAndTechnology_RoadObjectDetectionWithDeepLearning/data. This will replace the *IDD_Detection* folder already present.

4. While in the *intelunnati_TeamTRON* folder, create a virtual environment using the following command - `python -m venv yolov5-env`. This creates a virtual environment named _yolov5-env_.
