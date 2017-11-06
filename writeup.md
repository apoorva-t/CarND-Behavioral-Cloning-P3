# **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/conv1.png "Conv Layer 1"
[image2]: ./examples/views.png "Center Camera View"
[image3]: ./examples/visual1.png "Histogram"
[image4]: ./examples/data_hist.png "Final dataset"
[image5]: ./examples/conv_ip.png "Input Image"
[image6]: ./examples/comv2.png "Conv Layer 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 the video of running in autonomous mode on track 1 from the center camera on the car

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have chosen to use the network architecture employed by the DAVE-2 project as described in the Nvidia paper 'End to End Learning for Self Driving Cars'.

The model consists of 5 convolutional layers followed by 3 fully connected layers. The first 3 convolutional layers use a stride of 2x2 with filter size of 5x5 and 24 filters. The next 2 layers use a non-strided convolution with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity after every stage except the last fully connected stage. The data is normalized in the model using a Keras lambda layer (code line 18). In order to reduce the data size and be in sync with the Nvidia architecture, the images are resized to 66x200 in a Keras lambda layer. Also, since the region of interest only lies in the part where the lane and its markings are, the images are cropped to remove the top 70 pixels and bottom 20 pixels.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after every fully connected layer (except the last one) in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). The default learning rate of 0.001 of the Adam optimizer proved to be good for training the model and there was no need for tuning the learning rate in later epochs.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used image data from all 3 cameras on board the car - center, left and right. A steering angle correction was applied for images from the left and right cameras.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with an existing Neural Network architecture and build on top of it. The two architectures I decided to use as my baseline were - LeNet architecture and the Nvidia architecture used in DAVE-2. 

LeNet seemed like a good starting point, since I had already used a modified version of LeNet in the Traffic Sign Classifier project. I added another convolutional layer to the original Lenet, and removed a fully connected layer. The Nvidia architecture was another obvious candidate for the model since this architecture has been proven to work well specifically for the purpose of predicting steering angles. My strategy was to train with both these models until I reach a point where I can discard one of these based on the autonomous driving results.

In order to gauge how well the models were working, I split my image and steering angle data into a training and validation set, with 20% of the original samples used for validation. My first runs with both the models showed a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I chose to add dropout with a keep probability of 0.5 (which had served me well in the previous Traffic Sign classification project). I also added 'ReLU' activations after every layer in my Nvidia based model. With this, the validation error became more in sync with the training error, decreasing with each epoch.

At this point I ran simulator in autonomous mode to see how well the car was driving around track one. The car drove well enough on the straight and slightly curved portions of the track. However, it veered off track on a sharper curve near the first bridge. To improve the driving behavior in these cases, I padded the original training dataset with more samples from simulator recordings driving on just these sharp curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

After evaluating simulator runs with both the models, I found the model based on Nvidia network architecture more robust to changes in training data and to yield smoother turns. In fact, after adding data from track 2, my LeNet based model was failing on track1 too. I suspect this is because deeper layers in the Nvidia architecture help it detect features more accurately.

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image     					    | 
| Normalization        	|      					                        | 
| Cropping	        	| Crop by (70,20) on (top,bottom)    		    | 
| Resize	        	| 66x200x3 RGB image    		  	  			| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 62x196x24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 58x192x36  |
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 54x188x48  |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 52x186x64  |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 50x184x64  |
| RELU					|												|
| Fully connected		| Outputs  100   								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| Outputs  50   								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| Outputs  10   								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| Outputs  1	   								|

Here is a visualization of the output for the first 2 convolutional layers:

Input image:
![alt text][image5] 

Convolutional Layer 1:
![alt text][image1]

Convolutional Layer 2:
![alt text][image6]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

This is a histogram of the steering angles obtained from training data collected on track 1 from the center camera viewpoint:
![alt text][image3]

As can be seen, the training data is biased towards driving straight as the number of samples with zero steering angle is much higher. 

To get a more balanced set of steering angles so that the model can learn to navigate curves in the track, I also added images from the left and right camera views to the training/validation datasets. The steering angle was corrected by +/-0.18 (found by trial and error) from the center steering angle for the left/right views. 

This is a view from the center, left and right cameras along with the corresponding steering angles for a particular frame during training:
![alt text][image2]

Track 1 mostly has left turns, so in order to balance the data set I drove a lap in the reverse direction for data collection. 

I then recorded the vehicle recovering from the left side and right sides of the road back to the center for a few curves so that the vehicle would learn how to correct its course if it happens to steer off the center of the track.

At this point, the car was running well in autonomous mode on track 1 up until the sharp curve after the bridge, where it veered off into the sand track. To help with such curves, I made 2-3 more recordings of the car making just these turns and patched the training data with these. With this, I was able to successfully complete track 1.

I then repeated this process on track two in order to get more data points. A surprising outcome was that after using training data from both track 1 and 2, the performance of the car on track 1 improved further - it took smoother turns and stayed closer to the center of the road than before.

I tried to augment the training data by flipping the images horizontally and flipping the signs of the steering angles. However, this did not help and in fact, made the performance worse on track 2. So after this point, I stuck to the method of patching more training data for just the sharp curves and turns on steep slopes.

After the collection process, I had 27135 number of data points. This is a histogram of the steering angles on my final training dataset:
![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. A generator with a batch size of 100 was used to feed training and validation data. The ideal number of epochs was 10 for which the model performed well on both tracks- as evidenced by the final training error = 0.5 and validation error = 0.46. I used an adam optimizer so that manually training the learning rate wasn't necessary. 

This model was able to successfully complete track 1 without leaving the drivable portion of the track, and it was able to complete almost 3/4th of track 2 - crashing at a point where there is a sharp right curve following a sharp left curve.
