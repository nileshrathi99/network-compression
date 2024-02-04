# Network Compression Using SVD - Implementation

## Part 1

### Training the Baseline Model
- Implemented a fully-connected neural network for MNIST classification with 5 hidden layers, each containing 1024 hidden units.
- Achieved a test accuracy above 98% through diligent training and experimentation.

### Singular Value Decomposition (SVD) Compression
1. Employed TensorFlow or PyTorch SVD implementations to approximate weight matrices using SVD: W = U S V.T
2. Varied the number of singular values D from 10, 20, 50, 100, 200 to D_full to create differently compressed versions.
3. Reported and analyzed test accuracies for each compressed network.
4. Compared the number of parameters in SVDed networks with the baseline, considering the diagonal nature of S.

## Part 2

### Low-Rank Approximation with Fixed \(D\)
1. Fixed D = 20 for low-rank approximation.
2. Defined a new network with factorized weight matrices: W = U V.T
3. Initialized factor matrices using SVD results from Part 1 with D = 20.
4. Finetuned the network with backpropagation, adjusting learning rates as necessary.
5. Reported and analyzed the test-time classification accuracy of the new network.

## Part 3

### Dynamic SVD during Training
1. Initialized weight matrices using the baseline model.
2. Implemented dynamic SVD at every epoch during training.
3. Fed forward using SVD'ed weights: W = U S V.T
4. Updated weights using backpropagation with the assumption of a perfect identity derivative.
5. Reported and analyzed the test-time classification accuracy, observing the impact of dynamic SVD on compression and performance.


### Dynamic SVD Performance Boost
- Achieved a notable performance boost for the D=20 compressed network, achieving an accuracy of around 97%.
- Remarkable memory savings, with the compressed network utilizing only about 2% of the original memory footprint.
- Demonstrated the effectiveness of dynamic SVD during training as a powerful technique for compressing neural networks while maintaining performance.


### Implementation Details
- Used PyTorch SVD implementations during the feedforward process.
- Experimented with hyperparameters to strike a balance between compression and performance.
- Results and analyses are provided in the accompanying code and documentation.

Feel free to explore the code, check the results, and modify hyperparameters for further experimentation. If you encounter any issues or have suggestions for improvements, please don't hesitate to open an issue or submit a pull request. Your feedback is highly appreciated!
