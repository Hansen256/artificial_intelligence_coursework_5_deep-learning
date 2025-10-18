# MNIST CNN Classification - Deep Learning Coursework

A comprehensive implementation and analysis of Convolutional Neural Networks (CNNs) for handwritten digit classification using the MNIST dataset. This project demonstrates the theoretical foundations and practical applications of deep learning for computer vision tasks.

## Project Overview

This project implements and compares various neural network architectures for MNIST digit classification, focusing on:

- **CNN vs Fully Connected Networks**: Comparative analysis of architectural approaches
- **Regularization Techniques**: Dropout and batch normalization effects
- **Optimizer Comparison**: Performance analysis of Adam, SGD, and RMSprop
- **Feature Visualization**: Understanding what CNNs learn through filter and feature map analysis
- **Theoretical Understanding**: Mathematical foundations and practical insights

## Key Results

- **CNN Test Accuracy**: 98.5-99.0%
- **Fully Connected Accuracy**: 97.5-98.0%
- **Training Time**: 15-20 minutes on CPU for full CNN training
- **Model Size**: ~1.2M parameters for CNN vs ~100K for FC network

## Project Structure

``` txt
├── mnist_cnn_classification.ipynb    # Main implementation notebook
├── CNN_Theoretical_Report.md         # Mathematical foundations and theory
├── CNN_Practical_Report.md          # Empirical observations and insights
├── .venv/                           # Python virtual environment
└── README.md                        # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- CPU with sufficient memory (4GB+ recommended)

### Installation

1. **Clone the repository**

   ``` bash
   git clone https://github.com/Hansen256/artificial_intelligence_coursework_5_deep-learning.git
   cd artificial_intelligence_coursework_5_deep-learning
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install required packages**

   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

### Running the Project

1. **Start Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Navigate to `mnist_cnn_classification.ipynb`
   - Run all cells to reproduce the complete analysis

3. **Alternative: Run specific sections**
   - Each section in the notebook can be run independently
   - Results and visualizations will be displayed inline

## Project Components

### 1. Main Implementation (`mnist_cnn_classification.ipynb`)

#### Data Exploration & Preprocessing

- MNIST dataset loading and analysis
- Pixel value normalization and data preparation
- Class distribution visualization

#### Model Architectures

- Basic CNN with two convolutional layers
- Fully connected baseline network
- CNN with dropout regularization
- CNN with batch normalization

#### Training & Evaluation

- Model training with progress monitoring
- Comprehensive performance evaluation
- Confusion matrix analysis
- Classification reports

#### Feature Analysis

- Learned filter visualization
- Feature map analysis
- Understanding CNN representations

### 2. Theoretical Report (`CNN_Theoretical_Report.md`)

#### Mathematical Foundations

- Convolution operation mathematics
- Backpropagation in CNNs
- Hierarchical feature learning theory

#### Architecture Analysis

- Why CNNs excel at image classification
- Parameter efficiency comparisons
- Inductive biases and regularization

#### Advanced Concepts

- Batch normalization theory
- Dropout mathematical formulation
- Optimization algorithm analysis

### 3. Practical Report (`CNN_Practical_Report.md`)

#### Empirical Observations

- Real-world performance analysis
- Training dynamics and convergence
- Resource usage and optimization

#### Comparative Analysis

- CNN vs Fully Connected performance
- Regularization technique effects
- Optimizer behavior comparison

#### Best Practices

- Practical recommendations
- Debugging and monitoring strategies
- Performance optimization tips

## Experimental Results

### Model Comparison

| Model Type | Test Accuracy | Parameters | Training Time |
|------------|---------------|------------|---------------|
| Basic CNN | 98.5-99.0% | ~1.2M | 15-20 min |
| CNN + Dropout | 99.3% | ~1.2M | 15-20 min |
| CNN + BatchNorm | 99.1-99.4% | ~1.2M | 12-15 min |
| Fully Connected | 97.5-98.0% | ~100K | 5-10 min |

### Regularization Effects

- **Dropout (0.25 rate)**: Reduces overfitting gap from 0.6% to 0.1%
- **Batch Normalization**: 20-30% faster convergence, improved stability
- **Combined Techniques**: Best generalization performance

### Optimizer Performance

- **Adam**: Fastest convergence, most stable training
- **SGD + Momentum**: Sometimes higher final accuracy, requires tuning
- **RMSprop**: Good middle ground, reliable performance

## Visualizations

The project includes comprehensive visualizations:

- **Dataset Analysis**: Sample images, pixel distributions, class balance
- **Training Curves**: Accuracy and loss progression over epochs
- **Confusion Matrices**: Detailed error analysis by digit class
- **Feature Maps**: Visualization of learned representations
- **Filter Analysis**: Understanding what CNN filters detect

## Key Insights

### Theoretical Insights

1. **Spatial Inductive Bias**: CNNs naturally encode spatial relationships
2. **Parameter Sharing**: Reduces overfitting while maintaining expressiveness
3. **Hierarchical Learning**: Features build from simple to complex patterns
4. **Translation Invariance**: Objects recognized regardless of position

### Practical Insights

1. **Architecture Matters**: Even simple CNNs outperform complex FC networks
2. **Regularization is Key**: Dropout and batch norm significantly improve generalization
3. **Optimizer Choice**: Adam provides reliable performance with minimal tuning
4. **Feature Learning**: Visualizing filters provides valuable debugging information

## Customization and Extension

### Hyperparameter Tuning

- Modify `batch_size`, `epochs`, `learning_rate` in the notebook
- Experiment with different dropout rates (0.1-0.5)
- Try various CNN architectures (deeper networks, different filter sizes)

### Adding New Experiments

- Implement data augmentation techniques
- Try different optimizers or learning rate schedules
- Compare with pre-trained models or transfer learning
- Extend to other datasets (Fashion-MNIST, CIFAR-10)

### Performance Optimization

- **CPU Optimization**: Adjust batch sizes for your hardware
- **Memory Management**: Reduce model size if memory is limited
- **Training Speed**: Use mixed precision or distributed training for larger experiments

## Educational Value

This project serves as a comprehensive educational resource for:

- **Deep Learning Students**: Complete CNN implementation with theory
- **Computer Vision Practitioners**: Practical insights and best practices
- **Machine Learning Engineers**: Performance optimization and debugging techniques
- **Researchers**: Foundation for more advanced CNN architectures

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

- Ensure Python 3.8+ is installed
- Use virtual environment to avoid dependency conflicts
- Install TensorFlow CPU version for CPU-only training

#### Memory Issues

- Reduce batch size (try 64 or 32)
- Close other applications during training
- Use smaller model architectures if needed

#### Training Problems

- Check data preprocessing steps
- Verify model compilation parameters
- Monitor for NaN values in loss function

### Performance Issues

#### Slow Training

- Reduce model complexity for faster iteration
- Use smaller subset of data for experimentation
- Consider cloud computing for intensive experiments

#### Poor Accuracy

- Check data normalization
- Verify label encoding
- Adjust learning rate or optimizer

## Additional Resources

### Further Reading

- **Deep Learning Book** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Pattern Recognition and Machine Learning** by Christopher Bishop
- **Convolutional Neural Networks for Visual Recognition** (CS231n Stanford Course)

### Online Resources

- [TensorFlow Official Documentation](https://tensorflow.org/tutorials)
- [Keras Examples and Tutorials](https://keras.io/examples/)
- [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)
