# Landscape-image-colorisation
# Objective
Produce color images given grayscale input image.

# Dataset
Use the download link below to download the dataset. You are expected to split your dataset to 
create a validation set for initial testing. Your final model can use the entire dataset for training. Note that 
this model will be evaluated on a test dataset not visible to you.

# Baseline Model

A baseline model is available in basic_model.py. You may use this model to kickstart 
this assignment. 
1) Fill in the dataloader, (colorize_data.py)
2) Fill in the loss function and optimizer. (train.py)
3) Complete the training loop, validation loop (train.py)
4) Determine model performance using appropriate metric. Describe your metric and why the metric 
works for this model?
5) Prepare an inference script that takes as input grayscale image, model path and produces a color 
image.

The network available in model.py is a very simple network. How would you improve the overall image 
quality for the above system?

# Bonus

You are tasked to control the average mood (or color temperature) of the image that you are 
colorizing. What are some ideas that come to your mind?
