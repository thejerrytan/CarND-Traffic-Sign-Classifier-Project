
# ** Jerry's Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[visualization]: ./report/visualization.png "Visualization"
[activation_visualization]: ./report/activation_visualization.png "Activation visualization"
[histogram_train]: ./report/histogram_train.png "histogram train"
[histogram_test]: ./report/histogram_test.png "histogram test"
[histogram_validation]: ./report/histogram_validation.png "histogram validation"
[histogram_augmented]: ./report/histogram_train_post_augmentation.png "histogram augmented train"
[grayscale]: ./report/grayscale.png "Grayscaling"
[normalized]: ./report/normalized.png "normalized"
[perspective_transform]: ./report/perspective_transformation.png "perspective transform"
[rotation_transform]: ./report/rotation_transformation.png "rotation_transform"
[scaling_transform]: ./report/scaling_transformation.png
[translation_transform]: ./report/translation_transformation.png "translation_transform"
[image1]: ./traffic-signs-test/speed_limit_120.jpg "Traffic Sign 1"
[image2]: ./traffic-signs-test/roundabout.jpg "Traffic Sign 2"
[image3]: ./traffic-signs-test/stop.jpg "Traffic Sign 3"
[image4]: ./traffic-signs-test/no_entry.jpg "Traffic Sign 4"
[image5]: ./traffic-signs-test/workzone.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/thejerrytan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (RBG image)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are some photos which appear really dark, these might be taken under low light conditions (at night) and will likely pose a problem for both humans and machine.

This is a 8 x 6 grid showing thumbnails of the traffic signs in RBG format, before normalization. The labels are above each thumbnail.

![alt text][visualization]

Below is a histogram showing the distribution of labels for train set.

![alt text][histogram_train]

This is a histogram showing distribution of labels for validation set.

![alt text][histogram_validation]

This is a histogram showing distribution of labels for test set.

![alt text][histogram_test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it was mentioned in the paper by Yann LeCun and Pierre Sermanet that it yielded better results as compared to training with full RBG color information. A possible explanation is that the network cannot make full use of the overwhelming amount of color information.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscale]


As a last step, I normalized the image data to values between -1 and 1 because it is known to result in better numerical stability.

Here is an example of the same traffic sign before normalizing and after.

![alt text][normalized]

I decided to generate additional data because 
1) it is a way to oversample for under-represented classes. The train set is unevenly distributed, classes with more samples are going to have a disproportionate influence on the network's weights. 
2) an easy way to increase the training set and improve accuracy.
3. it also allows the model to generalize better to new images which are slightly rotated, translated or scaled.

To add more data to the the data set, I used the following techniques:
1. rotation - because the labels should be the same regardless of orientation of the photo, rotation invariance
2. translation in width and height - because the labels should be the same regardless of where the sign is, translation invariance
3. scaling up - because the labels should be the same regardless of size of the traffic sign, scale invariance. 
Since the images have to fit into a 32 x 32 image, parts of the resulting image will be cropped out. This allows the network to learn how to recognize the traffic sign even if parts of it is occluded.
There is no need to scale down because it is already being scaled down in the subsampling stage of the network.
4. perspective transform - the traffic sign looks different when viewed from different angles, but label should be the same


The difference between the original data set and the augmented data set is the following ... 

Perspective transform
![alt text][perspective_transform]

Rotation transforma
![alt text][rotation_transform]

Translation transform
![alt text][translation_transform]

Scaling transform
![alt text][scaling_transform]

Only classes with num of samples less than 1000 are augmented. The new training data distribution after augmentation is as follows:

![alt text][histogram_augmented]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32  				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x64 					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 	 				|
| Fully connected		| input 1600, output  120     					|
| RELU					|												|
| Dropout				| dropout 50%									|
| Fully connected		| input 120, output  84     					|
| Dropout				| dropout 50%									|
| Softmax				| 	        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer on cross-entropy error function. Adam stands for adaptive momentum estimation and incorporates elements of momentum and adagrad optimizer. It keeps track of the recent history of gradients to achieve a smoother descent along the error function surface.

After trying out various learning rates - 0.00001, 0.0001, 0.001 and various batch sizes - 32, 64, 128, as well as differnt epochs 20, 50, 100, I discovered that the training rate of 0.00001 and 0.0001 was too slow and training for 20 or 50 epochs resulted in pre-mature termination while the accuracy was still increasing. Hence learning rate of 0.001 was chosen. Batch size does not seem to matter much, except smaller batch size results in faster training times, hence i chose 64. Ideally training should stop at the peak of the accuracy vs epochs curve, where more training increases training accuracy but validation accuracy drops. That is when the model goes from underfitting to overfitting. After training 100 epochs, I realized that the training accuracy has reached its maximum of 0.999 but validation accuracy plateaus at 0.95. The first this happened is at around epoch 50, hence if we want minimal optimal epoch, it should be 50.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.940
* test set accuracy of 0.935

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?

The LeNet-5 architecture from our previous lab was chosen. It was originally made to classify MNIST handwritten digits, which were much simpler in terms of features. Since traffic signs are much more complicated - RBG images, with brightness, contrast, shapes and sizes, i increased the first convolution layer depth to 32 and second layer to 64 which means more feature maps available to help the network recognize salient features. This change greatly improved my validation accuracy from 0.89 to 0.95. For fear of overfitting, I added in 2 dropout layers at the 2 fully connected layers, which i believe, has helped the network to converge to 0.95 validation accuracy and stay constant even with more epochs.

* Why did you believe it would be relevant to the traffic sign application?

LeNet was built to recognize and classify images, the input here is in image form too.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 As training accuracy increases, the validation accuracy increases as well. Prolonged training does not improve nor decrease the validation accuracy which means the model is not overfitting. Final training and validation accuracy is high ~ 0.95 so we are not underfitting. Gap between train and validation accuracy is small ~ 0.4 i.e. they are converging and tracking each other. Test accuracy is 0.935 which is high and very very close to validation accuracy - only 0.005 difference.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The speed limit 120 km/h sign might be difficult to classify because it has a bright reflection glare at the top right hand corner.

The roundabout sign might be difficult to classify because its rich background might cause confusion for the network.

The stop sign might be difficult to classify because its red background, when converted to grayscale, might overwhelm the textual information on the sign.

The no entry sign might be difficult because of partial occlusion by a pole.

The road work sign might be difficult because its rich features cannot be fully captured in a low resolution 32 x 32 image, causing it to be confused with other traffic signs with rich features like bumpy road, animal crossing, children crossing etc.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| Go straight or right   						| 
| Roundabout mandatory 	| Roundabout mandatory 							|
| Speed limit (120km/h)	| speed limit (50 km/h)							|
| Stop sign	      		| speed limit (50 km/h)			 				|
| Road work 	   		| Road work 	      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This does not compare favourably to the test set accuracy of 0.935 but this is not a representative measure as it is just over a sample of 5, and the traffic signs were purposely chosen to be difficult.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For all the images, the model was very certain (close to 1) about its prediction, except for the first image - which was a partially occluded No entry sign, wrongly classified it as a Go Straight or Right sign but guessed correctly that it could be a no entry sign as its second most likely answer.

Actual label: No entry
Top 5 predicted label, probability:
1. 0.999, Go straight or right
2. 0.001, No entry
3. 0.000, Go straight or left
4. 0.000, End of no passing by vehicles over 3.5 metric tons
5. 0.000, Turn right ahead

Actual label: Roundabout mandatory
Top 5 predicted label, probability:
1. 1.000, Roundabout mandatory
2. 0.000, Speed limit (20km/h)
3. 0.000, Speed limit (30km/h)
4. 0.000, Speed limit (50km/h)
5. 0.000, Speed limit (60km/h)

Actual label: Speed limit (120km/h)
Top 5 predicted label, probability:
1. 1.000, Speed limit (50km/h)
2. 0.000, Speed limit (60km/h)
3. 0.000, Ahead only
4. 0.000, Stop
5. 0.000, Speed limit (30km/h)

Actual label: Stop
Top 5 predicted label, probability:
1. 1.000, Speed limit (50km/h)
2. 0.000, Speed limit (80km/h)
3. 0.000, No passing for vehicles over 3.5 metric tons
4. 0.000, Turn left ahead
5. 0.000, No vehicles

Actual label: Road work
Top 5 predicted label, probability:
1. 1.000, Road work
2. 0.000, Slippery road
3. 0.000, Bicycles crossing
4. 0.000, Speed limit (20km/h)
5. 0.000, Speed limit (30km/h)


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][activation_visualization]

The feature maps are currently of low contrast and lower resolution than expected, i believe it could be a mistake when calling the function with activation_min and activation_max.
