###### *intelunnati_TeamTRON*

# **<u>Intel Unnati Project - Team TRON</u>**

This repository was created as part of Intel UNNATI Industrial Training Project, 2023. The topic of our project was "Road Object Detection with Deep Learning". <br>

<p> Our aim was to develop a model that could identify objects in roads, specifically Indian roads. The reason that other commonly available models could not work properly in this environment is due to the variety of objects found in Indian roads. <br><br>

#### Setting-up environment and Data pre-processing:

1. Clone | fork this project - [intelunnati_TeamTRON](https://github.com/noel-robert/intelunnati_TeamTRON)

2. Download the dataset (_Indian Driving Dataset_) from [http://idd.insaan.iiit.ac.in](http://idd.insaan.iiit.ac.in) [_you will be asked to create an account_]. IDD Detection (_22.8 GB_) is the dataset being used in this case. <br> - Note that dataset directory here does not have _Annotations_  and _JPEGImages_ folders due to the large size of the dataset.

3. Extract downloaded dataset and place into TeamTRON_MarBaseliosCollegeOfEngineeringAndTechnology_RoadObjectDetectionWithDeepLearning/**data**. This will replace the ***IDD_Detection*** folder already present.

4. While in the ***intelunnati_TeamTRON*** folder, create a virtual environment named _yolov5-env_ using the following command - `python -m venv yolov5-env`. [*This step is recommended so that modules needed for this project will not affect any other projects.*]<br> - To activate the environment, type `yolov5-env\Scripts\activate` in your terminal.

5. Navigate to TeamTRON_MarBaseliosCollegeOfEngineeringAndTechnology_RoadObjectDetectionWithDeepLearning/**models** and clone the [YOLOv5 Github repository]([GitHub - ultralytics/yolov5: YOLOv5 🚀 in PyTorch &gt; ONNX &gt; CoreML &gt; TFLite](https://github.com/ultralytics/yolov5)) into this using the terminal command `git clone https://github.com/ultralytics/yolov5`. *[You will notice that there is already a file named yolov5 when you download the main repository itself, but it won't actually contain any files as it just links to an external repository]*<br>Navigate further into the cloned directory using `cd yolov5` and use `pip install -r requirements.txt` to install required modules.

6. Navigate to TeamTRON_MarBaseliosCollegeOfEngineeringAndTechnology_RoadObjectDetectionWithDeepLearning/**code** and run `pip install lxml` to install *lxml* module, which is needed for data preprocessing.

7. Run *datasetPreprocessing.py* using the command `python datasetPreprocessing.py`. This will create a new folder ***modified_dataset*** in ***data*** folder, which is where dataset is stored in the proper format. <br>

8. As of 13/07/2023, it has been noticed that a package Pillow v10.0.0 is causing issues. So, it is recommended to downgrade to version 9.5 using the code `pip install --upgrade Pillow==9.5`.

###### (Optional) Installing CUDA for training on GPU:

1. Check if your [NVIDIA GPU supports CUDA](https://developer.nvidia.com/cuda-gpus).

2. To download CUDA, check [CUDA Toolkit 12.2 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads), and for older versions check [CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive).

3. [Download](https://pytorch.org/get-started/locally/) PyTorch. Make sure that Compute Platform versions of PyTorch and CUDA match.

4. Optionally, try installing [cuDNN](https://developer.nvidia.com/cudnn), but a *NVIDIA Developer Program Membership* is required for this.

#### Model training:

1. YOLOv5 has 5 different pretrained models to choose from - *YOLOv5n*, *YOLOv5s*, *YOLOv5m*, *YOLOv5l*, and *YOLOv5xl*. The .yaml files for these models along with configuration file *idd.yaml* are present within the ***models*** folder. These files have been modified to suit requirements of this project.

2. Move to ***models/yolov5*** directory and use the below command for training the model
   
   ```python
   python train.py --img <image_size> --batch <batch_size> --epochs <num_epochs> --data <data/data.yaml> --cfg <path_to_model_config>
   ```
   
   add '--device cuda:0' if you are using a dedicated GPU.
   
   *Optional: use `--weights <path_to_weights>` if you have pretrained weights from any previous runs.*
   
   *Optional: you can set the number of workers with `--workers <no_of_workers>` but do take note that workers should not exceed the number of cores your CPU has.*
   
   For example during the first training, the command was:
   
   ```python
   python train.py --img 640 --batch 8 --epochs 50 --data ../idd.yaml --cfg ../yolov5n.yaml --device cuda:0 --workers 8
   ```

3. Results of model training can be found in ***models/yolov5/runs/trains/exp_no***.<br>A ***weights*** folder is also present containing *best.pt* and *last.pt*.The *best.pt* file contains the weights to be used for next iteration.

#### Validating the trained model:

1. Validation images are present inside ***modified_dataset/images/val***. To run the program for using trained weights to validate, use the following code while in ***models/yolov5*** directory:<br>
   
   ```python
   python val.py --data ../idd.yaml --weights runs/train/exp5/weights/best.pt --device cuda:0
   ```
   
   you might need to change the path for weights to point to latest set of 

2. ***yolov5/runs/val/exp_no*** contains output after running the command.

#### Detecting object in images:

1. We have successfully trained a model on our dataset and validated. Now, our model is ready.

2. You need to place your custom image inside ***data/custom_test_images*** and run the following command `python detect.py --source <path/to/images> --weights <path/to/weights.pt> --conf 0.4`. Here, only those with a confidence threshold of 0.4 is chosen. Example code to directly run detect.py is:<br>
   
   ```python
   python detect.py --source ../../data/custom_test_images/test1.jpeg --weights runs/train/exp5/weights/best.pt --conf 0.4
   ```

3. Results can be found inside ***yolov5/runs/detect/exp_no***.

Collaborators:  
 [@Josh-Alex](https://github.com/JoshAlex12)  
 [@noel-robert](https://github.com/noel-robert)  
 [@nubifathima](https://github.com/nubifathima)
