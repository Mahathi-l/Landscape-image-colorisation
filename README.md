# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  
## Background
- In image colorization, the goal is to produce a color image given a greyscale image. As a single greyscale image may correspond to many plasible coloured versions,
- traditional models rely on significant user input along side grayscale image. A full-coloured image has 3 values per pixel ( lightness, saturation & hue) associated 
- with it whereas greyscale has only 1 value per pixel(lightness). 
- The inputs are of size 256 x 256 x1 (lightness) and outputs are 25 x256 x 2.
- Since LAB colorspace format retains the same info as RBG images, as it makes it easy to seperate lightness channel from A* & B*, RBG images are converted to LAB colorspace
- for ease.
- The base model shared uses a simple Resnent-18 architecture where regression has  been employed to perform classification.

- In the last layer of basic_model.py, the model takes 3 neurons corresponding to RBG channels. I had tweaed it to only take 2 channels corresponding to A* & B*.
- To improvize the model, AverageMeter() method was added to the base model to monitor the model metrics such as training loss, validation loss & computation time.
- Incorporated GPU flag to check for GPU availability to make sure the model tracks the availability otherwise runs on CPU.
- Passed the training and validation datasets via dataloaders in the Train.py file.
- Defined the following model parameters:
Learning rate = 0.0001
Batch size = 64
Epochs = 25
Training/Val split = 75% train and 25% validation
Criteria = Mean Squared Error loss function
Optimizer = Adam optimizer
- The training and validation methods convert the incoming input and output into a pytorch Variable, that helps in conversion of the incoming data to a
  pytorch wrapper around the tensor.
- To enhance the final input image, I tried playing around with different paramters and functions that OpenCV has to offer. I ended up using cv2.detailEnhance,
  which worked much better than adding any kind of filter, dilation or erosion of the image.

## Running the model

- Please run the model to train as: python execute_model.py <data_dir> or
  by default landscape_images will be picked
- Please run the inference script as: python inference.py <checkpoint_file> <grayscale_image> -> Stores a file called output.jpg,
  which will be the colored image.

