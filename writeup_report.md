# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left.jpg "left"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/right.jpg "right"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` or `writeup_report.pdf` summarizing the results (this file)

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

`drive.py` is all the same with the one from original project GitHub repository, but the target speed was modified to 20km/h which is a bit faster.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture looks like this in my code:

```
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

Output should be in size 1x1, since the only control input is steering input in this case.

#### 2. Attempts to reduce overfitting in the model

The model's training epoch is set to 5 by trial and error. After this, the loss value does not converges further.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Both training and validation dataset was selected randomly using `sklearn.model_selection.train_test_split()`. The validation dataset is 20% of overall dataset.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and mirrored image of the center lane driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adjust input and output size of the published CNN for this problem.

My first step was to use a convolution neural network model similar to the PilotNet, the model published in 2017 by NVIDIA [here](https://arxiv.org/abs/1704.07911). I thought this model might be appropriate because this is the model that is exactly built for the same problem with this project and works well.

In order to adjust to this project, I put normalization layer with `keras.layers.core.Lambda()` function and the normalization formula is `pixel/255 - 0.5`.

Also, I added cropping layer in order to eliminate scene area. Here, I cropped out upper 70 pixels and lower 25 pixels with `keras.layers.convolutional.Cropping2D()` function.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	                   					| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 3@160x320 RGB image       							| 
| Normalization     	| output = input/255 - 0.5, outputs 3@160x320        	|
| Cropping           	| eliminate upper 70px, lower 25px, outputs 3@65x320  	|
| Convolution 5x5	    | 2x2 stride, same padding, outputs 24@31x158           |
| RELU					| -								        				|
| Convolution 5x5	    | 2x2 stride, same padding, outputs 36@14x77            |
| RELU					| -								        				|
| Convolution 5x5	    | 2x2 stride, same padding, outputs 48@5x37             |
| RELU					| -								        				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 64@3x35             |
| RELU					| -								        				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 64@1x33             |
| RELU					| -								        				|
| Flatten       		| outputs 2112     						        		|
| Fully connected		| outputs 100  							        		|
| Fully connected		| outputs 50  							        		|
| Fully connected		| outputs 10  							        		|
| Fully connected		| outputs 1  							        		|
| Output          		| 1 (steering input)    				        		|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. For better performance, I utilized mouse input to generate realistic steering input data (since I am kind of good in playing racing video games). Here is an example image of center lane driving:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I then applied correction factor to the steering input data for left and right images. The correction factor for center/left/right image was 0/0.25/-0.25. These factor was added to the originally captured steering angles from the simulator. This procedure is in order to teach how to steer back to the desired position in lane when the vehicle is not at the desired position in the lane (which is mostly center).

Also, for more data points and generalization of the model, I mirrored all the images horizontally using `cv2.flip()` and all the corrected steering angles by inverting its sign.

After the collection and correction process, I had 23418 number of data points, which corresponds to 23418/3 = 7806 time steps (including both simulated and virtually created). 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by trial and error. I used an adam optimizer so that manually training the learning rate wasn't necessary. The loss function was defined as mean square error of the steering angle since this is regression model.
