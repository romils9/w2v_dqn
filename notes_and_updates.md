# Notes about the various steps in this project
* dqn_atari.py is used to train the model initially
* Next collect_trajectories.py loads the saved model and generates and collects trajectories using the saved model.
* Next, cnn_trajectory_vecs.py loads the saved model and transfers the CNN related weights into a new CNN only model and converts the trajectory states into vectors obtained at the output of the CNN.
* Then, w2v.py converts these CNN output vector representations into vector representations for the given state using w2v.
    * Here we first pass the CNN output vecs into an AE. We take the output of the middle layer of this AE as the new vector representations so that the initial vector size is condensed.
        * CNN vector output is of size = 3136
        * AE vector output is of size = 128 (hyperparameter)
    * Next we take the AE output vectors and pass them through the Skip-Gram model of w2v using -ve Sampling.
        * **Other details about the implementation to be included here**
    * Finally the w2v middle layer holds the new and improved representations for each state.
* The hope is that by using these w2v based vector representations for each state, we will be able to improve the performance of agent on the Atari environment set-up.




# Current updates and information
* As of now the model being used was trained for 10M time-steps
* Using Breakout-v4
* We will keep the CNN weights frozen since w2v inputs depend upon the CNN.
* Hyperparameters include <w2v vector size>, <AE vector size>, <number of steps of training for the CNN before freezing weights>
