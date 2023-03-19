# Deep_learning-
Assignment_01_cs910


Question-1:
There are ten classes in the Fashion-MNIST data set and here is a dictionary relating the model's numerical labels and corresponding class names.
Class_labels_names = { "0": "T-shirt/Top", "1": "Trouser",
"2": "Pullover", "3": "Dress",
"4": "Coat", "5": "Sandal",
"6": "Shirt", "7": "Sneaker",
"8": "Bag", "9": "Ankle Boot", }

Solution Approach:
Import data from fashion_mnist.
Sort in sorted_arr until first occurrence of a class is found.
Plot all the classes with associated class names.
Integrate wandb to log the images and keep track of the experiment using wandb.ai.


Question-2:Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

Solution Approach:
Feed-forward neural network feed_forward() has been implemented which takes in the training dataset(X_train), weights, biases, and activation function.
Initialize the randomized weights, biases as per the number of inputs, hidden & output layer specification using param_inint.
Implement loss functions such as: 1. cross entropy 2. Mean squared error
Implement Activation functions such as:
sigmoid, tanh, relu...etc
our code provides the flexibility in choosing the above mentioned parameters.
It also provides flexibility in choosing the number of neurons in each hidden layer.


Question-3:
Back propagation algorithm implemented with the support of the following optimization function and the code works for any batch size:
SGD
Momentum based gradient descent
Nesterov accelerated gradient descent
RMS Prop
ADAM
NADAM

Solution Approach:
Make use of output of the feed-forward neural network in the previous question.
Initialize one_hot function to encode the labels of images.
Implement the activation functions and their gradients.
sgd
softmax
Rel
tanh
Initialize the randomized parameters using the 'random' in python.
Initialize predictions, accuracy and loss functions.
loss functions are:
Mean squared Error
Cross entropy
Initialize the gradient descent classes.
and Initialize the train function to use the above functions.

Question-4:Use the sweep functionality provided by wandb to find the best
values for the hyperparameters listed below. Use the standard
train/test split of fashion_mnist (use (X_train, y_train), (X_test,
y_test) = fashion_mnist.load_data() ). Keep 10% of the training
data aside as validation data for this hyperparameter search. Here
are some suggestions for different values to try for hyperparameters.
As you can quickly see that this leads to an exponential number of
combinations. You will have to think about strategies to do this
hyperparameter search efficiently. Check out the options provided
by wandb.sweep and write down what strategy you chose and why.

Solution Approach:
Split the training data in the ratio of 9:1.
The standard training & test split of fashion_mnist has been used with 60000 training images and 10000 test images & labels.
10% shuffled training data was kept aside as validation data for the hyperparameter search i.e, 6000 images.
wandb.sweeps() provides an functionality to comapre the different combinations of the hyperparameters for the training purpose.
we are avail with 3 types of search strategies which are:
grid
random
Bayes
By considering the number of parameters given, there are totally 5400 combinations are possible.
grid : It checks through all the possible combinations of hyperparameters. If there are n hyperparameters and m options of each hyperparameter. There will be m^n number of runs to see the final picture, hence grid search strategy wont work beacause it would be a computationally intensive.
There are 2 options left to choose.
we chose random search. and we obtained a maximum validation accuracy of 88.443% after picking the sweep function, set the sweep function of wandb by setting up the different parameters in sweep configuration


Question-5:We would like to see the best accuracy on the validation set across all
the models that you train.
wandb automatically generates this plot which summarises the test
accuracy of all the models that you tested. Please paste this plot
below using the "Add Panel to Report" feature

Solution Approach:
The best accuracy across all the models is a validation accuracy of 88.443%.
The graph containing a summary of validation accuracies for all the models is shown in the wandb report.

Question-10:Based on your learnings above, give me 3 recommendations for what
would work for the MNIST dataset (not Fashion-MNIST). Just to be
clear, I am asking you to take your learnings based on extensive
experimentation with one dataset and see if these learnings help on
another dataset. If I give you a budget of running only 3
hyperparameter configurations as opposed to the large number of
experiments you have run above then which 3 would you use and
why. Report the accuracies that you obtain using these 3
configurations.

Solution Approach:
Since MNIST is a much simpler dataset, and a very similar image classification task with the same number of classes, the configurations of hyperparameters that worked well for Fashion-MNIST is worked well for MNIST too.
Although transfer learning from the pre trained Fashion MNIST dataset's best model configuration for the digits MNIST dataset is an extremely viable option for faster training and better initialization of the network, in the current implementation of the code, transfer learning has not been used.
