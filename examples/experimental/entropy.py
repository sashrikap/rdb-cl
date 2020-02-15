import numpy as np
from scipy.stats import gaussian_kde


def sample_data(min_=0.0, max_=10.0, num=20, dim=2):
    return np.random.random((num, dim)) * (max_ - min_) + min_


if __name__ == "__main__":
    data1 = sample_data(max_=5.0)
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 1 {entropy1:.3f}")

    data1 = sample_data()
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 2 {entropy1:.3f}")

    data1 = sample_data(max_=20)
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 3 {entropy1:.3f}")

    data1 = sample_data(max_=30)
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 4 {entropy1:.3f}")

    data1 = sample_data(max_=30, num=30)
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 4 {entropy1:.3f}")

    data1 = sample_data(max_=30, num=40)
    kernel1 = gaussian_kde(dataset=data1.T)
    N = data1.shape[0]
    entropy1 = -(1.0 / N) * np.sum(np.log(kernel1(data1.T)))
    print(f"Entropy 4 {entropy1:.3f}")
