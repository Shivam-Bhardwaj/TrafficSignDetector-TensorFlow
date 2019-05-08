## Traffic sign detector using TensorFlow
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) <img src="https://engineering.nyu.edu/sites/default/files/2019-01/tandon_long_color.png" alt="NYU Logo" width="130" height="whatever">

------

The following project is a part of Udacity’s Self Driving car engineering NanoDegree program. The aim of project is to successfully classify traffic sign from German Traffic sign dataset having 43 classes.

------

The Project
---

The steps of this project are the following:

* Dataset is downloaded automatically.
* Dataset is converted into the python friendly format.
* Preprocessing the data
* Designing a neural network architecture in TensorFlow.
* Training and testing the Network on the dataset.
* Testing the network on random images.
* Showing the Top 5 Softmax Probabilities For Each Image Found on the Web.

------

## Prerequisites

- Pip 
- Python 3
- Virtual Environment

## Install instructions

`open terminal`

```bash
$ git clone https://github.com/Shivam-Bhardwaj/TrafficSignDetector-TensorFlow.git
$ virtualenv --no-site-packages -p python3 venv 
$ source venv/bin/activate
$ cd TrafficSignDetector-TensorFlow
$ pip install -r requirements.txt
$ jupyter notebook
```

`open Traffic_Sign_Classifier.ipnyb`

------

## Downloading dataset

**You don't have to think about the dataset. My code does everything for you :)** 

The code under `Step 0: Load The Data` does the following:

1. Download the dataset in an external folder.

2. Unzip the dataset.

3. Delete the original Zip file as it is no longer required.

4. Load the data in Pickle format.

5. Save the data in 

   ```python
   X_train, y_train # For training data and labels
   X_valid, y_valid # For validation data and labels
   X_test, y_test # For Final test data and labels
   ```

------

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

------

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

The given README.md file is an extensive writeup of the project. For any questions, please contact 

Shivam Bhardwaj 

 [LinkedIn](<https://www.linkedin.com/in/shivamnyu/>) [Instagram](https://www.instagram.com/lazy.shivam/) [Facebook](<https://www.facebook.com/shivambhardwaj2008>) [Github](https://github.com/Shivam-Bhardwaj)

Mail to shivam.bhardwaj@nyu.edu

------

### Data Set Summary & Exploration

As explained in the code I used the pandas library to calculate summary statistics of the traffic
signs data set:

- The size of training set is 34799 images
- The size of the validation set is 4410 images
- The size of test set is 12630
- The shape of a traffic sign image is (32, 32, 3)
- The number of unique classes/labels in the data set is 43

| ClassId | SignName                                           |
| ------- | -------------------------------------------------- |
| 0       | Speed limit (20km/h)                               |
| 1       | Speed limit (30km/h)                               |
| 2       | Speed limit (50km/h)                               |
| 3       | Speed limit (60km/h)                               |
| 4       | Speed limit (70km/h)                               |
| 5       | Speed limit (80km/h)                               |
| 6       | End of speed limit (80km/h)                        |
| 7       | Speed limit (100km/h)                              |
| 8       | Speed limit (120km/h)                              |
| 9       | No passing                                         |
| 10      | No passing for vehicles over 3.5 metric tons       |
| 11      | Right-of-way at the next intersection              |
| 12      | Priority road                                      |
| 13      | Yield                                              |
| 14      | Stop                                               |
| 15      | No vehicles                                        |
| 16      | Vehicles over 3.5 metric tons prohibited           |
| 17      | No entry                                           |
| 18      | General caution                                    |
| 19      | Dangerous curve to the left                        |
| 20      | Dangerous curve to the right                       |
| 21      | Double curve                                       |
| 22      | Bumpy road                                         |
| 23      | Slippery road                                      |
| 24      | Road narrows on the right                          |
| 25      | Road work                                          |
| 26      | Traffic signals                                    |
| 27      | Pedestrians                                        |
| 28      | Children crossing                                  |
| 29      | Bicycles crossing                                  |
| 30      | Beware of ice/snow                                 |
| 31      | Wild animals crossing                              |
| 32      | End of all speed and passing limits                |
| 33      | Turn right ahead                                   |
| 34      | Turn left ahead                                    |
| 35      | Ahead only                                         |
| 36      | Go straight or right                               |
| 37      | Go straight or left                                |
| 38      | Keep right                                         |
| 39      | Keep left                                          |
| 40      | Roundabout mandatory                               |
| 41      | End of no passing                                  |
| 42      | End of no passing by vehicles over 3.5 metric tons |

------

### Testing parameters

The code was tested on the following specifications

- **CPU:** `Intel(R) Core(TM) i9-8950HK CPU @ 4.8 Ghz`
- **GPU:** `Nvidia GeForce GTX 1050 Ti Mobile`
- **OS:** `Ubuntu 16.04.6 LTS (Xenial Xerus)` 
- **Kernal:** `4.15.0-48-generic`

Training for 50 epochs takes  ~ 80 Sec

[//]: #	"Image References"
[image1]: ./camera_cal/calibration1.jpg	"Undistorted"
[image2]: ./test_images/test1.jpg	"Road Transformed"
[image3]: ./examples/binary_combo_example.jpg	"Binary Example"
[image4]: ./examples/warped_straight_lines.jpg	"Warp Example"
[image5]: ./examples/color_fit_lines.jpg	"Fit Visual"
[image6]: ./examples/example_output.jpg	"Output"
[video1]: ./project_video.mp4	"Video"



