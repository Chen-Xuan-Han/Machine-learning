import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 

def bayes_discriminant(mu1, sigma1, mu2, sigma2):
    """
    計算貝氏判別點

    Args:
        mu1, sigma1: 第一類常態分佈的均值和標準差
        mu2, sigma2: 第二類常態分佈的均值和標準差

    Returns:
        float: 貝氏判別點
    """
    # 假設先驗概率相等，則貝氏判別點為使得兩個類別似然比等於1的點
    def f(x):
        return norm.pdf(x, mu1, sigma1) / norm.pdf(x, mu2, sigma2) - 1

    # 使用scipy.optimize.fsolve求解方程f(x) = 0
    from scipy.optimize import fsolve
    return fsolve(f, (mu1 + mu2) / 2)

# 自定義參數
mu1, sigma1 = 0, 1 #第一類常態分佈的均值和標準差
mu2, sigma2 = 2, 0.5#第二類常態分佈的均值和標準差

# 生成數據（僅用於繪圖，實際分類時不需要生成大量數據）
x = np.linspace(-3, 6, 1000)

y1 = norm.pdf(x, mu1, sigma1)/2
y2 = norm.pdf(x, mu2, sigma2)/2

# 計算貝氏判別點
discriminant_point = bayes_discriminant(mu1, sigma1, mu2, sigma2)

# 繪圖

plt.plot(x, y1, label='Class 1')
plt.plot(x, y2, label='Class 2')
plt.axvline(discriminant_point, color='k', linestyle='--', label='Bayes Discriminant')
plt.legend()
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Two Normal Distributions and Bayes Discriminant')
plt.show()