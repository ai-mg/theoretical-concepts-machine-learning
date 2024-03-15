# Theoretical Concepts of Machine Learning

## Central Limit Theorem

The Central Limit Theorem (CLT) is a fundamental result in the field of statistics and probability theory. It provides a foundation for understanding why many distributions in nature tend to approximate a normal distribution under certain conditions, even if the original variables themselves are not normally distributed. The theorem states that, given a sufficiently large sample size, the distribution of the sample means will be approximately normally distributed, regardless of the shape of the population distribution, provided the population has a finite variance.

The formula or the mathematical formulation of the CLT can be derived from the concept of convergence in distribution of standardized sums of independent random variables. Let's consider the classical version of the CLT to understand where the formula comes from:

Consider a sequence of $n$ independent and identically distributed (i.i.d.) random variables, $X_1, X_2, ..., X_n$, each with a mean $\mu$ and a finite variance $\sigma^2$. The sample mean $\bar{X}$ of these $n$ random variables is given by:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

The Central Limit Theorem tells us that as $n$ approaches infinity, the distribution of the standardized sample means (i.e., how many standard deviations away the sample mean is from the population mean) converges in distribution to a standard normal distribution. The standardized sample mean is given by:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}
$$

This formula ensures that $Z$ has a mean of 0 and a standard deviation of 1:

- **Mean of \(Z\)**: $E(Z) = E\left(\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}\right) = \frac{E(\bar{X}) - \mu}{\frac{\sigma}{\sqrt{n}}} = 0$

- **Variance of $Z$**: $Var(Z) = Var\left(\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}\right) = \frac{Var(\bar{X})}{\left(\frac{\sigma}{\sqrt{n}}\right)^2} = 1$

Here, $Z$ converges in distribution to a standard normal distribution $N(0, 1)$ as $n$ becomes large.

<!-- This means:

$$
\lim_{n \to \infty} P\left(a < \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} < b\right) = \Phi(b) - \Phi(a)
$$

where $P$ denotes probability, $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution, and $a$ and $b$ are any two points on the real line with $a < b$. -->

The CLT is derived from the properties of characteristic functions or moment-generating functions of probability distributions. In essence, the CLT can be proven by showing that the characteristic function of $Z$ converges to the characteristic function of a standard normal distribution as $n$ approaches infinity ([see Appendix A.3](#A.3.-Derivation-of-normal-distribution-from-central-limit-theorem)).

The significance of the CLT lies in its ability to justify the use of the normal distribution in many practical situations, including hypothesis testing, confidence interval construction, and other inferential statistics procedures, even when the underlying population distribution is unknown or non-normal.

### 1. Why $\sigma$ is divided by $\sqrt{n}$:

We know that that Z-Score or Standardization is the deviation of the data point from mean in units of standard deviation ([see Appendix A.2. on Standardization](#A.2.-Standardization-or-$Z$-Score)). Here, the deviation is of the sample mean ($\bar{X}$) from the population mean ($\mu$). Therefore, we derive the standard deviation of the sample mean ($\bar{X}$) as follows.

#### Variance of the Sample Mean $\bar{X}$

The variance of the sample mean $\bar{X}$ is derived as follows. Since $\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$, its variance is:

$$
Var(\bar{X}) = Var\left(\frac{1}{n} \sum_{i=1}^{n} X_i\right)
$$

Because the $X_i$ are i.i.d., the variances add up, and we get:

$$
Var(\bar{X}) = \frac{1}{n^2} \sum_{i=1}^{n} Var(X_i) = \frac{1}{n^2} \cdot n \cdot \sigma^2 = \frac{\sigma^2}{n}
$$

This shows that the variance of the sample mean decreases as the sample size $n$ increases.


#### Standard Deviation of $\bar{X}$

The standard deviation is the square root of the variance. Therefore, the standard deviation of the sample means, also known as the standard error of the mean (SEM), is:

$$
SEM = \sqrt{Var(\bar{X})} = \sqrt{\frac{\sigma^2}{n}} = \frac{\sigma}{\sqrt{n}}
$$

#### Mathematical Explanation for Dividing by $\sqrt{n}$

- **Reducing Spread**: Dividing by $\sqrt{n}$ reduces the spread of the sampling distribution of the sample mean as the sample size increases. This reflects the fact that larger samples are likely to yield means closer to the population mean ($\mu$), thus decreasing variability among the sample means.
  
- **Normalization**: The process of dividing the population standard deviation ($\sigma$) by $\sqrt{n}$ normalizes the scale of the sample means' distribution. This normalization ensures that no matter the sample size, the scale (spread) of the distribution of sample means is consistent and comparable.

#### Role in the Central Limit Theorem

The CLT states that as $n$ approaches infinity, the distribution of the standardized sample means:

$$
Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}
$$

converges to a standard normal distribution $N(0, 1)$. Here, $\bar{X}$ is the mean of a sample of size $n$, $\mu$ is the population mean, and $\sigma$ is the population standard deviation. The denominator $\frac{\sigma}{\sqrt{n}}$ standardizes the distribution of $\bar{X}$ by adjusting for the size of the sample, allowing the theorem to hold across different sample sizes and population variances.

#### Conclusion

Mathematically, dividing by $\sqrt{n}$ in the calculation of the SEM and the standardization of sample means under the CLT ensures that the variability among sample means decreases with increasing sample size. This adjustment is fundamental to the convergence of the distribution of sample means to a normal distribution, a cornerstone of statistical inference.

### 2. Derivation of the Variance of $Z$

Given the definition of $Z$, to find its variance, we use the property that the variance operator $Var(aX) = a^2 Var(X)$ for any random variable $X$ and constant $a$ (see proof in Appendix A.1.). Applying this to the definition of $Z$, we get:

$$
Var(Z) = Var\left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}\right)
$$

Since $\mu$ is a constant, subtracting it from $\bar{X}$ does not affect the variance, so we focus on the scaling factor. Applying the variance operator:

$$
Var(Z) = \left(\frac{1}{\sigma / \sqrt{n}}\right)^2 Var(\bar{X}) = \left(\frac{\sqrt{n}}{\sigma}\right)^2 \cdot \frac{\sigma^2}{n} = \frac{n}{\sigma^2} \cdot \frac{\sigma^2}{n} = 1
$$

This calculation shows that the variance of $Z$ is 1. Here's the breakdown:

- $\left(\frac{1}{\sigma / \sqrt{n}}\right)^2$ is the square of the inverse of the standard deviation of $\bar{X}$, which is $\sigma / \sqrt{n}$.
- $Var(\bar{X}) = \frac{\sigma^2}{n}$ is the variance of the sample mean.
- Multiplying these together, the $\sigma^2$ and $n$ terms cancel out, leaving $Var(Z) = 1$.

The derivation shows that the process of standardizing the sample mean $\bar{X}$ results in a new variable $Z$ with a variance of 1. This is a crucial step in the application of the CLT because it ensures that $Z$ is scaled appropriately to have a standard normal distribution with mean 0 and variance 1 as $n$ becomes large. This standardization allows us to use the properties of the standard normal distribution for statistical inference and hypothesis testing.

#### Properties of Variance

Variance is a fundamental statistical measure that quantifies the spread or dispersion of a set of data points or a random variable's values around its mean. Understanding the properties of variance is crucial for statistical analysis, as these properties often underpin the manipulation and interpretation of statistical data. Here are some key properties of variance:

#### 1. Non-negativity
Variance is always non-negative ($Var(X) \geq 0$). This is because variance is defined as the expected value of the squared deviation from the mean, and a square is always non-negative.

#### 2. Variance of a Constant
The variance of a constant ($c$) is zero ($Var(c) = 0$). Since a constant does not vary, its spread around its mean (which is the constant itself) is zero.

#### 3. Scaling Property
Scaling a random variable by a constant factor scales the variance by the square of that factor: $Var(aX) = a^2 Var(X)$, where $a$ is a constant and $X$ is a random variable. This property was detailed in a previous explanation.

#### 4. Variance of a Sum of Random Variables
For any two random variables $X$ and $Y$, the variance of their sum is given by $Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)$, where $Cov(X, Y)$ is the covariance of $X$ and $Y$. If $X$ and $Y$ are independent, $Cov(X, Y) = 0$, and the formula simplifies to $Var(X + Y) = Var(X) + Var(Y)$.

#### 5. Linearity of Variance (for Independent Variables)
While the expectation operator is linear ($E[aX + bY] = aE[X] + bE[Y]$), variance is not linear except in specific cases. For independent random variables $X$ and $Y$, and constants $a$ and $b$, $Var(aX + bY) = a^2 Var(X) + b^2 Var(Y)$. However, for dependent variables, you must also consider the covariance term.

#### 6. Variance of the Difference of Random Variables
Similar to the sum, the variance of the difference of two random variables is $Var(X - Y) = Var(X) + Var(Y) - 2Cov(X, Y)$. For independent variables, this simplifies to $Var(X - Y) = Var(X) + Var(Y)$, as their covariance is zero.

#### 7. Zero Variance Implies a Constant
If a random variable $X$ has a variance of zero ($Var(X) = 0$), then $X$ is almost surely a constant. This is because no variation from the mean implies that $X$ takes on its mean value with probability 1.

These properties are widely used in statistical modeling, data analysis, and probability theory, especially in the derivation of statistical estimators, hypothesis testing, and in the study of the distributional properties of sums and transformations of random variables.


### 3. CLT simulation in Python for a centered continous uniform random distribution

```python=9

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm

%matplotlib inline

# Set the random seed for reproducibility
np.random.seed(42)

# Parameters
sample_sizes = [1, 2, 10]
num_realizations = 100000
a, b = -np.sqrt(3), np.sqrt(3)

# Plot setup
fig, axes = plt.subplots(len(sample_sizes), 1, figsize=(10, 6), sharex=True)
plt.subplots_adjust(hspace=0.5)

# Perform simulations and plotting
for ax, n in zip(axes, sample_sizes):
    # Generate realizations of Z
    Z = (1 / np.sqrt(n)) * np.sum(np.random.uniform(a, b, size=(num_realizations, n)), axis=1)
    print(Z.shape)
    # Plot histogram of the realizations
    sns.histplot(Z, bins=30, kde=True, ax=ax, stat='density', label=f'n={n}')
    ax.set_title(f'Histogram of Z with n={n}')
    
    # Overlay the standard normal density for comparison
    x = np.linspace(min(Z), max(Z), 100)
    ax.plot(x, norm.pdf(x), 'r--', label='Standard Normal')
    ax.legend()

plt.suptitle('Density/Histogram of Z and Standard Normal Distribution')
plt.show()

```

![Screenshot 2024-03-15 at 3.50.42 AM](https://hackmd.io/_uploads/r1KmON-0T.png)

The plots demonstrate the Central Limit Theorem (CLT) in action for different sample sizes $n \in \{1, 2, 10\}$. Each subplot shows the distribution of $Z = \frac{1}{\sqrt{n}} \sum_{i=1}^n X_i$ for 100,000 realizations, where $X_i$ are drawn from the uniform distribution over $[-\sqrt{3},\sqrt{3})$. This distribution is overlayed with the density of the standard normal distribution (dotted red curve) for comparison.

Here's what we observe:

- For $n=1$, the histogram of $Z$ resembles the uniform distribution itself, since there's no averaging involved, and the transformation simply scales the distribution.
- For $n=2$, the histogram starts to show a more bell-shaped curve, indicating the beginning of the convergence towards a normal distribution as predicted by the CLT.
- For $n=10$, the histogram closely resembles the standard normal distribution, showcasing a significant convergence towards normality. This demonstrates the CLT, which states that the sum (or average) of a large number of independent, identically distributed random variables, regardless of the original distribution, will tend towards a normal distribution.

These observations align with the CLT, highlighting its power: as the number of samples increases, the distribution of the sum (or mean) of these samples increasingly approximates a normal distribution, even when the original variables are not normally distributed.

#### How does the number of realizations affect the histogram?

The number of realizations, or samples, you choose for generating a histogram affects the smoothness and accuracy of the representation of the underlying distribution. This principle applies not just to histograms of samples from a standard normal distribution, but to histograms of samples from any distribution. Here's how changing the number of realizations impacts the histogram:

#### More Realizations

- **Smoothness**: Increasing the number of realizations tends to produce a smoother histogram. This is because more data points allow for a more detailed and accurate approximation of the distribution's shape.
- **Accuracy**: With more samples, the histogram better approximates the true underlying distribution (in this case, the standard normal distribution). You're more likely to observe the classic bell curve shape of the normal distribution with a mean of 0 and a standard deviation of 1.
- **Stability**: The histogram becomes more stable with respect to random fluctuations. This means that if you were to repeat the experiment multiple times, the shape of the histogram would be more consistent across trials.

#### Fewer Realizations

- **Roughness**: Decreasing the number of realizations can lead to a rougher, more jagged histogram. This is because there are fewer data points to outline the distribution's shape, leading to a less accurate representation.
- **Inaccuracy**: With fewer samples, the histogram might not accurately capture the true characteristics of the underlying distribution. It might miss certain features or exaggerate others due to the limited data.
- **Instability**: The histogram becomes more susceptible to random fluctuations. Small sample sizes can lead to significant variability in the histogram's shape across different realizations of the experiment.

#### Practical Implications

In practical terms, choosing the number of realizations depends on the balance between computational resources and the need for accuracy. For exploratory data analysis or when computational resources are limited, a smaller number of realizations might suffice. However, for precise statistical analysis or when the goal is to closely approximate the underlying distribution, using a larger number of realizations is preferable.

It's also important to note that while increasing the number of realizations improves the approximation to the underlying distribution, it does not change the distribution itself. The Central Limit Theorem (CLT) ensures that, given enough samples, the distribution of sample means will approximate a normal distribution, independent of the number of realizations used to construct each individual histogram.

### Further links and resources

- On convergence speeds for different common distributions in CLT: https://david-salazar.github.io/posts/fat-vs-thin-tails/2020-05-30-central-limit-theorem-in-action.html

## Appendix

### A.1. Derivation of $Var(Y)=a^2 Var(X)$

The variance of a random variable measures the dispersion of that variable's values around its mean. The formula for the variance of a random variable $X$ is defined as:

$$
Var(X) = E[(X - E[X])^2]
$$

where $E[X]$ is the expected value (or mean) of $X$, and $E$ denotes the expectation operator.

Now, let's consider a new random variable $Y = aX$, where $a$ is a constant. We want to derive the variance of $Y$, denoted as $Var(Y)$ or $Var(aX)$.

#### Step 1: Define $Y = aX$

Given $Y = aX$, we apply the variance formula:

$$
Var(Y) = E[(Y - E[Y])^2]
$$

Since $Y = aX$, we have $E[Y] = E[aX]$.

#### Step 2: Calculate $E[Y]$

The expected value of $Y$ is:

$$
E[Y] = E[aX] = aE[X]
$$

This is because the expectation operator is linear, and the constant $a$ can be factored out of the expectation.

#### Step 3: Plug $E[Y]$ into the Variance Formula

Substituting $Y = aX$ and $E[Y] = aE[X]$ into the variance formula, we get:

$$
Var(Y) = E[((aX) - aE[X])^2] = E[(a(X - E[X]))^2]
$$

#### Step 4: Simplify the Expression

Since $a$ is a constant, we can factor it out of the squared term:

$$
Var(Y) = E[a^2(X - E[X])^2] = a^2 E[(X - E[X])^2]
$$

Noting that $E[(X - E[X])^2]$ is the definition of $Var(X)$, we have:

$$
Var(Y) = a^2 Var(X)
$$


This derivation shows that the variance of $Y = aX$, where $a$ is a constant, is $a^2$ times the variance of $X$. The key takeaway is that scaling a random variable by a constant $a$ scales its variance by $a^2$, reflecting the squared nature of variance as a measure of dispersion.

### A.2. Standardization or $Z$-Score

Standardization is a statistical method used to transform random variables into a standard scale without distorting differences in the ranges of values. The process converts original data into a format where the mean of the transformed data is 0 and the standard deviation is 1. This transformation is achieved by subtracting the expected value (mean) from each data point and then dividing by the standard deviation.

#### The Formula

The formula for standardizing a random variable $X$ is:

$$
Z = \frac{X - \mu}{\sigma}
$$

It is basically the deviation of data point from mean (i.e. how far the data point is from the mean) per unit standard deviation. For e.g, $Z = 2$ means that the data point is 2 standard deviation away from the mean. 

where:
- $X$ is the original random variable,
- $\mu$ is the mean of $X$,
- $\sigma$ is the standard deviation of $X$, and
- $Z$ is the standardized variable.

#### Why Use Standardization?

The rationale behind standardization and the specific form of the standardization formula involves several key statistical principles:

1. **Comparability**: Standardization allows data from different sources or distributions to be compared directly. Because the standardized data has a mean of 0 and a standard deviation of 1, it removes the units of measurement and normalizes the scale, making different datasets or variables comparable.

2. **Normalization**: Many statistical methods and machine learning algorithms assume or perform better when the data is normally distributed or similarly scaled. Standardization can help meet these assumptions or improve performance by giving every variable an equal weight, preventing variables with larger scales from dominating those with smaller scales.

3. **Understanding Z-scores**: The standardized value, or **Z-score, tells you how many standard deviations away from the mean a data point is.** This can be useful for identifying outliers, understanding the distribution of data, and performing statistical tests.

4. **Mathematical Foundation**: The formula is grounded in the properties of the normal distribution. In a standard normal distribution, the mean ($\mu$) is 0, and the standard deviation ($\sigma$) is 1. The standardization process transforms the data so that it can be described in terms of how far each observation is from the mean, in units of the standard deviation. This transformation is particularly useful in the context of the Central Limit Theorem, which states that the distribution of the sample means tends towards a normal distribution as the sample size increases, regardless of the shape of the population distribution.

#### Conclusion

The act of subtracting the mean and dividing by the standard deviation in standardization serves to "normalize" the scale of different variables, enabling direct comparison, simplifying the interpretation of data, and preparing data for further statistical analysis or machine learning modeling. This process leverages the fundamental statistical properties of mean and standard deviation to achieve a standardized scale, where the effects of differing magnitudes among original data values are neutralized.

### A.3. Derivation of Normal Distribution from Central Limit Theorem

Deriving the normal distribution mathematically from the Central Limit Theorem (CLT) in a simple, non-technical explanation is challenging due to the advanced mathematical concepts involved, particularly the use of characteristic functions or moment-generating functions. However, I'll outline a basic approach using characteristic functions to give you a sense of how the derivation works. This explanation simplifies several steps and assumes some familiarity with concepts from probability theory.

#### Step 1: Understanding Characteristic Functions

The characteristic function $\phi_X(t)$ of a random variable $X$ is defined as the expected value of $e^{itX}$, where $i$ is the imaginary unit and $t$ is a real number:

$$
\phi_X(t) = E[e^{itX}]
$$

Characteristic functions are powerful tools in probability theory because they uniquely determine the distribution of a random variable, and they have properties that make them particularly useful for analyzing sums of independent random variables.

#### Step 2: The Characteristic Function of the Sum of Independent Variables

Consider $n$ independent and identically distributed (i.i.d.) random variables $X_1, X_2, ..., X_n$, each with mean $\mu$ and variance $\sigma^2$. Let $S_n = X_1 + X_2 + ... + X_n$ be their sum. The characteristic function of $S_n$ is:

$$
\phi_{S_n}(t) = \left(\phi_X(t)\right)^n
$$

This is because the characteristic function of a sum of independent variables is the product of their individual characteristic functions.


#### Step 3: Standardizing $S_n$

First, standardize $S_n$ to get $Z_n$:

$$
Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}
$$

#### Characteristic Function of $Z_n$

We want to find the characteristic function of $Z_n$, $\phi_{Z_n}(t)$. The characteristic function for $S_n$ is $\phi_{S_n}(t) = (\phi_X(t))^n$, since the variables are i.i.d.

Given the Taylor expansion of $\phi_X(t)$ around 0:

$$
\phi_X(t) \approx 1 + it\mu - \frac{t^2\sigma^2}{2} + \text{higher-order terms}
$$

#### Adjusting for $Z_n$

To adjust this for $Z_n$, note that we're interested in the effect of $t$ on $Z_n$, not on the original variables. The transformation involves a shift and scaling of $t$, considering the definition of $Z_n$. So, we replace $t$ with $\frac{t}{\sigma\sqrt{n}}$ to reflect the scaling in $Z_n$ and consider the subtraction of $n\mu$, which shifts the mean to 0:

$$
\phi_{Z_n}(t) = E\left[e^{it\frac{S_n - n\mu}{\sigma\sqrt{n}}}\right] = \left(\phi_X\left(\frac{t}{\sigma\sqrt{n}}\right)\right)^n \cdot e^{-it\mu\sqrt{n}/\sigma}
$$

Substituting the approximation for $\phi_X\left(\frac{t}{\sigma\sqrt{n}}\right)$ and simplifying, we aim to show that this converges to $e^{-t^2/2}$ as $n \rightarrow \infty$.

#### Applying the Approximation and Taking the Limit

When you substitute the Taylor expansion into the expression for $\phi_{Z_n}(t)$ and simplify, focusing on terms up to the second order, you essentially deal with:

$$
\left(1 + i\frac{t}{\sigma\sqrt{n}}\mu - \frac{\left(\frac{t}{\sigma\sqrt{n}}\right)^2\sigma^2}{2}\right)^n
$$

Since $\mu$ is the mean of the original distribution, and we're considering the sum $S_n$ minus $n\mu$, adjusted by $\sigma\sqrt{n}$, this simplifies to:

$$
\left(1 - \frac{t^2}{2n}\right)^n
$$

As $n \rightarrow \infty$, this expression converges to $e^{-t^2/2}$, by the limit definition of the exponential function:

$$
\lim_{n \to \infty} \left(1 - \frac{t^2}{2n}\right)^n = e^{-t^2/2}
$$

#### Final Step: Connection to the Standard Normal Distribution

This result, $e^{-t^2/2}$, is the characteristic function of a standard normal distribution $N(0,1)$ (i.e., a mean $(\mu$) of 0 and a standard deviation ($\sigma$) of 1). The inverse Fourier transform (or the characteristic function inversion theorem) tells us that the probability density function corresponding to this characteristic function is the PDF of the standard normal distribution:

$$
f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$

This formula describes the distribution of values that $Z$ can take, where $Z$ represents the number of standard deviations away from the mean a particular observation is. The factor $\frac{1}{\sqrt{2\pi}}$ normalizes the area under the curve of the PDF to 1, ensuring that the total probability across all possible outcomes is 1, as required for any probability distribution.

This demonstrates how, under the Central Limit Theorem, the distribution of the standardized sum (or average) of a large number of i.i.d. random variables, regardless of their original distribution, converges to a normal distribution, provided the original variables have a finite mean and variance.

This derivation, while not delving into the full technical rigor of the proofs involving characteristic functions, provides a conceptual bridge from the CLT to the emergence of the normal distribution.

