# Univariate Time Series Forecasting

This project was developed for the course of **Artificial Neural Networks and Deep Learning** for the MSc. in Computer, Mathematical, and High-Performance Computing Engineering at Politecnico di Milano, A.Y. 2023/2024.

## Overview
The challenge aims to build a model capable of effectively generalizing to predict future samples of 60 time series in the test set.

### Dataset
The dataset consists of monovariate time series, i.e., composed of a single feature, belonging to six different domains: Demography, Finance, Industry, Macroeconomy, Microeconomy, and Others. Each row of the dataset corresponds to a time and feature-independent time series.

**Dataset Details:**
- **Time Series Length:** Varies, with a maximum of 2776. Series have been padded with zeros to reach this uniform length.
- **File Format:** `.npy`
- **Categories:** Six distinct categories, indicated by {'A', 'B', 'C', 'D', 'E', 'F'}.
- **Dataset Structure:** The dataset contains the following files:
  - `training_data.npy`: Numpy array of shape (48000, 2776).
  - `valid_periods.npy`: Numpy array of type (48000, 2) with the start and end indices of the original series.
  - `categories.npy`: Numpy array of shape (48000,) with category codes for each series.

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/federicoschermi/time-series-training-dataset?utm_medium=social&utm_campaign=kaggle-dataset-share).

### Model Requirements
- **Input/Output:** The input `X` will be a numpy array with shape [BS, 200], where 200 is the length of the sequences. The returned output must be a numpy array with shape [BS, 18].

## Authors

- Luigi Pagani ([@LuigiPagani](https://github.com/LuigiPagani))
- Flavia Petruso ([@fl-hi1](https://github.com/fl-hi1))
- Federico Schermi ([@federicoschermi](https://github.com/federicoschermi))

## Output

Check out the final [`report.pdf`](./report_final.pdf).

## References
1. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural machine translation by jointly learning to align and translate”. In: arXiv preprint arXiv:1409.0473 (2014). [Link](https://arxiv.org/abs/1409.0473)
2. Seyed Mehran Kazemi et al. “Time2vec: Learning a vector representation of time”. In: arXiv preprint arXiv:1907.05321 (2019). [Link](https://arxiv.org/abs/1907.05321)
3. Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing systems 30 (2017). [Link](https://arxiv.org/abs/1706.03762)

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
