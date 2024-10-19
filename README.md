# Univariate Time Series Forecasting

This project was developed for the course of **Artificial Neural Networks and Deep Learning** for the MSc. in Computer, Mathematical, and High-Performance Computing Engineering at Politecnico di Milano, A.Y. 2023/2024.

## Overview
The challenge aims to build a model capable of effectively generalizing to predict future samples of 60 time series in the test set.

### Dataset
The dataset consists of monovariate time series, i.e., composed of a single feature, belonging to six different domains: Demography, Finance, Industry, Macroeconomy, Microeconomy, and Others. Each row of the dataset corresponds to a time and feature-independent time series.

**Dataset Details:**
- **Time Series Length:** Varies, with a maximum of 2776. Series have been padded with zeros to reach this uniform length.
- **File Format:** .npy
- **Categories:** Six distinct categories, indicated by {'A', 'B', 'C', 'D', 'E', 'F'}.
- **Dataset Structure:** The dataset contains the following files:
  - `training_data.npy`: Numpy array of shape (48000, 2776).
  - `valid_periods.npy`: Numpy array of type (48000, 2) with the start and end indices of the original series.
  - `categories.npy`: Numpy array of shape (48000,) with category codes for each series.

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/federicoschermi/time-series-training-dataset?utm_medium=social&utm_campaign=kaggle-dataset-share).

### Data Exploration and Preprocessing
The dataset consists of 48000 time series across six categories (A to F). The time series have been zero-padded to ensure consistent lengths, resulting in a 48000 by 2776 matrix. We analyzed the distribution of the series lengths and found that most are under 1000 time points, with class F being underrepresented. Detailed figures are available in the dataset_analysis and further_dataset_exploration notebooks.

We computed the value distributions by class and observed that category A had a slightly higher median compared to the others, though overall distributions were largely consistent. After experimenting with models both including and excluding category labels, we found comparable performance, and decided not to include them further.

To capture seasonality, we relied on autocorrelation, revealing patterns every 7 and 12 steps, indicating potential weekly and monthly cycles. These insights led us to use a minimum window size of 12 to help models learn seasonal components. Additionally, sequences shorter than 72 (4 times the predicted length of 18) were excluded, resulting in 34876 final time series.

We split the data into 90% training and 10% validation sets. The training sequences were generated with a window of 200 and a stride of 6, with padding applied as needed.

### Model Requirements
- **Input/Output:** The input X will be a numpy array with shape [BS, 200], where 200 is the length of the sequences. The returned output must be a numpy array with shape [BS, 18].

## Model Architectures and Performance

We experimented with several neural network architectures to identify the best model for our task. The models we tested included:

- **Vanilla LSTM (128)**
- **LSTM (256) seq2seq + attention**
- **LSTM (64) seq2seq + attention + Time2Vec (dim=3)**
- **CONV1D (32) + LSTM (128) seq2seq + attention**
- **Transformer + multi-head attention (4 heads)**

**Performance Comparison:**

| Model | Validation MAE |
|-------|----------------|
| Vanilla LSTM | 0.0574 |
| LSTM seq2seq + attention | 0.0554 |
| LSTM seq2seq + attention + Time2Vec | 0.0564 |
| CONV1D + LSTM seq2seq + attention | 0.0561 |
| Transformer + multi-head attention | 0.0605 |

The best performing model during validation was the LSTM seq2seq with attention mechanism, though the CONV1D LSTM model also showed competitive performance.

### Attention Mechanisms in Sequence Modeling
We introduced attention mechanisms to allow models to selectively focus on important segments of input sequences and to improve the robustness of predictions, especially for sequences with a high proportion of zero-padding. The Transformer model used a multi-head attention mechanism to enhance the capacity for handling complex temporal dependencies.

### Time2Vec Embedding
The Time2Vec embedding method was applied to enhance the representation of time in our models. For LSTM-based architectures, we used a Time2Vec embedding dimension of 3. Implementing Time2Vec in the Transformer model proved challenging due to increased training time, and this aspect would benefit from further optimization.

### Data Augmentation
To improve model robustness, we applied data augmentation, adding jitter to the original time series with a standard deviation proportional to 0.01 times the standard deviation of each time series. This resulted in tripling the number of sequences. Augmentation results were mixed, with the CONV1D LSTM model showing an MAE reduction of 0.006, while other models saw slight performance declines.

### Final Model Selection
After evaluating the models, we selected the **CONV1D LSTM seq2seq + attention** as the final model due to its performance and stability during testing.

**Final Model Architecture**:
- Masking
- 2x (CONV1D(32) + BatchNormalization + MaxPooling1D)
- LSTM (128, return sequences, return state)
- RepeatVector(1)
- LSTM (128, return sequences)
- Bi-LSTM (64, return sequences)
- Attention Mechanism (Dot product)
- Dropout (0.5)
- Context Vector (Dot product)
- Concatenate
- Flatten
- Dense (18)

### Results and Performance
The performance of the final model is as follows:

| Dataset | MAE |
|---------|-----|
| Train   | 0.0478 |
| Validation | 0.0564 |
| Test    | 0.0752 |

We observed some overfitting in the training set, suggesting that a simpler model or stronger regularization might further improve performance.

### Future Improvements
Several potential improvements could enhance model performance:
- **Data Augmentation**: More robust exploration of data augmentation techniques such as time warping or permutation.
- **Category Analysis**: Deeper investigation of the impact of categories, including the use of category-specific weights.
- **Regularization**: Implementing robust scaling techniques based on the median and interquartile range could help mitigate overfitting.

## Authors
- Luigi Pagani ([@LuigiPagani](https://github.com/LuigiPagani))
- Flavia Petruso ([@fl-hi1](https://github.com/fl-hi1))
- Federico Schermi ([@federicoschermi](https://github.com/federicoschermi))

## Output
Check out the final [report.pdf](./report_final.pdf).

## References
1. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate". In: arXiv preprint arXiv:1409.0473 (2014). [Link](https://arxiv.org/abs/1409.0473)
2. Seyed Mehran Kazemi et al. "Time2vec: Learning a vector representation of time". In: arXiv preprint arXiv:1907.05321 (2019). [Link](https://arxiv.org/abs/1907.05321)
3. Ashish Vaswani et al. "Attention is all you need". In: Advances in neural information processing systems 30 (2017). [Link](https://arxiv.org/abs/1706.03762)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
