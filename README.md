# ** Virtual Self-Driving Simulator**

This is a Self-Driving Car steering simulator based on Sully Chen's [model](https://github.com/SullyChen/Autopilot-TensorFlow/blob/master/model.py) implementation of the NVIDIA End to End Learning for Self-Driving Cars **(DAVE-2)** [paper](https://arxiv.org/pdf/1604.07316.pdf).

[//]: # (Image References)
[image0]: ./Img/simulator_image.jpg
[image1]: ./Img/dave2.jpg
[image2]: ./Img/aug/sat_var.png
[image3]: ./Img/aug/light_var.png
[image4]: ./Img/aug/shad.png
[image5]: ./Img/aug/trans.png
[image6]: ./Img/aug/eq.png

![][image0]

#  _**How to run**_
  * _**To drive simply type the following command in while in the project directory** (I have made the project using tensorflow such that there is no need to type `model.json` in front of it):_
    > **`python drive.py`**

  * _**To train type the following:**_
    > **`python train_on_game.py`**

    _**In order to train there need to be two metatdata(csv) files in the project folder:**_
    * **`driving_log.csv`** (_used for training and validation_)
    * **`test_driving_log.csv`** (_used for testing_)

### Model
The model has five convolutional layers, four fully connected layers and one output layer. It applies dropout in all of the fully connected layers. The following diagram from the NVIDIA paper illustrates the model.

![][image1]

<p></p>
<p></p>
<p></p>A complete table of the structure of the DAVE-2 Architecture.
<p></p>
<table>
  <tr>
    <td colspan="4"><b>Convolutional Layers</b></td>
  </tr>
  <tr>
  <tr>
    <td><i>Layer No.</i></td>
    <td><i>Kernel Size</i></td>
    <td><i>No. of Kernels</i></td>
    <td><i>Stride</i></td>
  </tr>
    <td><i>1st</i></td>
    <td>5x5</td>
    <td>24</td>
    <td>2x2</td>
  </tr>
  <tr>
    <td><i>2nd</i></td>
    <td>5x5</td>
    <td>36</td>
    <td>2x2</td>
  </tr>
  <tr>
    <td><i>3rd</i></td>
    <td>5x5</td>
    <td>48</td>
    <td>2x2</td>
  </tr>
  <tr>
    <td><i>4th</i></td>
    <td>3x3</td>
    <td>64</td>
    <td>1x1</td>
  </tr>
  <tr>
    <td><i>5th</i></td>
    <td>3x3</td>
    <td>64</td>
    <td>1x1</td>
  </tr>

  <tr>
    <td colspan="4"><b>Fully Connected Layers</b></td>
  </tr>

  <tr>
    <td colspan="2"><i>Layer No.</i></td>
    <td colspan="2"><i>Width</i></td>
  </tr>
  <tr>
    <td colspan="2"><i>6th</i></td>
    <td colspan="2">1164</td>
  </tr>
  <tr>
    <td colspan="2"><i>7th</i></td>
    <td colspan="2">100</td>
  </tr>
  <tr>
    <td colspan="2"><i>8th</i></td>
    <td colspan="2">50</td>
  </tr>
  <tr>
    <td colspan="2"><i>9th</i></td>
    <td colspan="2">10</td>
  </tr>

  <tr>
    <td colspan="4"><b>Output Layer</b></td>
  </tr>
  <tr>
    <td colspan="1"><i>10th</i></td>
    <td colspan="3">1 Neuron followed by <b>2*atan(x)</b> activation</td>
  </tr>
</table>

### Training
The model was trained on a [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity. The dataset contain **~8000** examples of center, right and left camera images along with steering angles. I used **80%** of this data for training and **20%** for validation. I also generated some additional test data by driving around on **track 1** of the Udacity Beta simulator.

### Reflections

**Looking back, the model can be improved even further by applying a few small modifications**

* **Using ELU instead of RELU to improve convergence rates. (_Suggested by Udacity reviewer_)**
* **Adding better shadows to the model so it can work on _track_ 2 in Fast, Simple, Good, Beautiful, Fantastic modes.**
* **Applying better offsets to the angels in order to get a better distribution of angles and avoid zig-zag behaviour**

For testing I generated some additional data from the simulator.

After about 30 epochs the model started working on track1.

After 35 epochs the model works on both track 1 and track 2 with full throttle.

## **Documentation**
**`trainer.py`**

  ```python
  class trainer
  ```
  ```python
  # The class has a constructor and two functions
  __init__(
  self,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2,
  tune_model = True,
  L2NormConst = 0.001,
  left_and_right_images = False,
  left_right_offset = 0.2,
  root_path = '',
  test_root_path ='',
  stop_gradient_at_conv = False,
  test_left_and_right_images = False
  )
  ```

  * **epochs:** Number of epochs
  * **validation_split:** The fraction of the data to use for validation_split
  * **tune_model:** Should we tune the model or start from scratch.
  * **L2NormConst:** The constant for amount of L2 regularization to apply.
  * **left_and_right_images:** Should we include left and right images?
  * **left_right_offset:** Amount of offset in angle for the left and right images.
  * **root_path:** The root path of the image.
  * **test_root_path:** The root path of the test images.
  * **stop_gradient_at_conv:** Should we stop the gradient at the conv layers.
  * **test_left_and_right_images:** Should we include left and right images during testing.


  ```python
  train(self) # Call this function to train
  ```
  ```python
  test(self) # Call this function to test
  ```

**`simulation_data.py`**

