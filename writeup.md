# **Traffic Sign Recognition** 

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

[clz_dist_train]: ./writeup/clz_dist_train.png "Training Data Distribution"
[before_gray]: ./writeup/before_gray.png "Before Grayscaling"
[after_gray]: ./writeup/after_gray.png "After Grayscaling"
[cfs_mat]: ./writeup/cfs_mat.png "Confusion Matrix"
[new_1]: ./writeup/new_1.png "Traffic Sign 1"
[new_2]: ./writeup/new_2.png "Traffic Sign 2"
[new_3]: ./writeup/new_3.png "Traffic Sign 3"
[new_4]: ./writeup/new_4.png "Traffic Sign 4"
[new_5]: ./writeup/new_5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/magicyuli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [a sample run of my code](https://github.com/magicyuli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html) (might be too big to be viewed in git directly..)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of the classes of the training data.

![Training Data Distribution][clz_dist_train]

I did the above for the validation and test data sets as well, and found that uneven distribution is present across all 3 sets.

I also wrote functions (see in code and html) to display 100 sample randomly drawn from the training set, to display rgb/grayscale images for a specific class for both training and validation sets. I found these functions especially useful when I was trying to find out why some samples were classified wrong and improving the classifier.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the beginning I trained on rgb images, so the only normalization I employed is `subtracting mean and dividing with standard deviation`. But I'm submitting the grayscale model as the final model.

So the first step I did there was converting the images into grayscale by averaging across the 3 channels.

I did grayscaling because it seems to be able to improve the model performance by reducing the influence of lighting, especially for dark images, where after grayscaling different parts of the images contrast more from each other. Another point of grayscaling is to reduce the amount of unnecessary features so that we can use smaller model, and the model won't try to fit the noises in the training data, in turn we avoid overfitting. Smaller model also trains faster.

Here is an example of a traffic sign image before and after grayscaling.

![Before Grayscaling][before_gray]

![After Grayscaling][after_gray]

Second I did the simpler normalization which is `(X - 128)/128`.

The reason behind normalization is to help the optimizer converge easier and faster. I didn't choose the `subtracting mean and dividing with standard deviation` method because I felt that for images, the adjacent features (pixels) were dependent/related, and this method could potentially disrupt the dependency/relation between them. But maybe I was wrong.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 3x3 patch with 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x32    |
| RELU					|												|
| Max pooling	      	| 3x3 patch with 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| 512 neurons				|
| RELU					|												|
| Fully connected		| 512 neurons				|
| RELU					|												|
| Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, because it converges fast due to its use of momentum and adaptive learning rate, where it applies different learning rate to different parameters (the more frequently updated, the less learning rate).

I used batch size of 128, I also tried batch size as large as 4096, but it always overfit the training set, whereas smaller batch size seems to produce model that generalizes better.

I didn't use fixed number of epochs. I learned the progressively reducing learning rate approach from [this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). I started with 0.001, as soon as the accuracy on the validation set dropped, I divided the learning rate by 10. After the learning rate shrank for 3 times, I trained the model with the final learning rate for 30 epochs.

I applied dropout with 0.5 keep probability to the fully connected layers.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were (calculated at the `validate` method of the `Model` class):
* training set accuracy of 0.998218
* validation set accuracy of 0.957370
* test set accuracy of 0.941093


##### What was the first architecture that was tried and why was it chosen?
The first architecture I used was of the same structure, but with more parameters, i.e. depths of the conv layes were 32, 64, and number of neurons in the fully connected layers were 1024, 1024. I chose it because I was training on the RGB images, and wanted more complex model to capture the features. Also the pooling layers were using patches of 2x2 instead of 3x3.

##### What were some problems with the initial architecture?
It overfit the training set more easily. And it trained slower.

Actually at the very beginning I didn't even have pooling layers and dropout in my model, which easily achieves 0.99 plus accuracy on the training set, but with only 0.6 on the validation set. After adding regularization, I got 0.85 on the validation set.

##### How was the architecture adjusted and why was it adjusted?
First I changed the patch size in the pooling layers to 3x3 after reading [this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), which led to `overlapping pooling`, i.e. the patch is 3x3 while the stride is 2x2. The paper claims this approach reduces the chances of overfitting, and it improved my validation accuracy to 0.929925.

Then I tried training model with grayscale images because after looking at the images that were classified wrong, I found most of them were extremely dark while the images of the same class in the training set were normally lightened. I also looked at the visualizations of the conv layers, and it seemed that the second conv layer barely activated, so I reduced the depth of the second conv layer to 32. I also reduced the fully connected layers to 512 neurons each, which gave me the final validation accuracy of 0.957370.


##### Which parameters were tuned? How were they adjusted and why?
The learning rate were dynamically reduced because when the model is closer to the optimum, large learning rate led to overshoot and bouncing. Batch size was reduced because large batch size led to serious overfitting.

##### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Max pooling and dropout greatly helped reduce overfitting. Convolution layer works well with images because of the statistical invariance existing in images. In other words, it can find the object of interest no mattern where it locates in the image.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (converted to 32x32):

![New Image 1][new_1] ![New Image 2][new_2] ![New Image 3][new_3] 
![New Image 4][new_4] ![New Image 5][new_5]

The stop sign image might be difficult to classify because it's really blurred and distorted out of normal ratio of aspects.

The 70m/h speed limit image might be difficult to classify because it's not properly lighted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Stop      		| Stop   									| 
| Go straight or left     			| Go straight or left 										|
| 50 km/h	      		| 50 km/h				 				|
| 70 km/h			| 70 km/h     							|


The model was able to correctly classify 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.10%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the `Stop` image, the model is relatively sure that this is a stop sign (probability of 0.63). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .63     				| Stop 										|
| .14					| 60 km/h											|
| .12	      			| Turn left ahead					 				|
| .017				    | End of all speed and passing limits      							|
| .012				    | Children crossing      							|

For other images, the model were absolutely sure about the answer and they were all correct. Below are the top 1 probabilities of the five images.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield   									| 
| .63     				| Stop 										|
| 1.00					| Go straight or left											|
| .99	      			| 50 km/h					 				|
| .99				    | 70 km/h      							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
From learning, the conv layers can learn lines, shapes, color contrast, as we can see from the activation visualizations of the conv layers at the bottom of [this](https://github.com/magicyuli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html).

#### 2. Possible improvements
Introducing blur, random noise, and random distortion to the training samples might improve the performance further. We can see this might be true if we look at the validation set classification result of images from class 24 in the confusion matrix below (last column is precisions and last row is recalls). The recall for class 24 is pretty low.

![Validation Set confusion matrix][cfs_mat]

And if we see the wrongly classified samples in the [html](https://github.com/magicyuli/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html), they're really blurred and distorted compared to their training set conterparts.
