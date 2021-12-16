# Polynomial_Regression_From_Scratch

I implemented a linear regression model and a polynomial regression model with a single input variable using gradient descent algorithm. My polynomial regression model can be used for any number of degree. Since linear functions are basically a polynomial function with the degree of 1, implementing the linear regression model on its own was not necessary at all.

I randomly chose float values between -1 and 1 to use as initial weights for all the parameters in the function. The correctness of the predicted line is calculated with the mean squared error (MSE) function. The program breaks the loop when the difference between two consecutive error values are lower than the threshold 10^(-7) for linear regression and 10^(-8) for polynomial regression. It also stops iterating when the maximum number of iterations have been reached, this only happens when the learning rate  and the threshold are too low. 

# Analytical Solution

The linear regression can also be calculated with the analytical solution. The formula for analytical solution is as follows: 
<img width="125" alt="image" src="https://user-images.githubusercontent.com/54302889/146243719-db95c761-ea7d-454d-8982-eeedc0ead078.png">	
θ represents the analytical solution function, X represents the x values of our dataset, <img width="26" alt="image" src="https://user-images.githubusercontent.com/54302889/146254048-c1f2e83a-cb97-425d-b3ad-4bdb2f6d53e7.png">
represents the transpose of the x matrix, <img width="65" alt="image" src="https://user-images.githubusercontent.com/54302889/146254212-0b427165-5e58-430b-acc8-e2cca53403d5.png">
represents the inverse of the dot product of <img width="26" alt="image" src="https://user-images.githubusercontent.com/54302889/146254048-c1f2e83a-cb97-425d-b3ad-4bdb2f6d53e7.png"> and X matrices, <img width="17" alt="image" src="https://user-images.githubusercontent.com/54302889/146255520-6d28b93e-6640-48b7-93f6-7db5acea4e75.png"> represents the y values in our dataset.

# Plotting Linear Regression Models

The following 6 figures show results for linear regression with various learning rates. Green lines represent the analytical solution while the black lines represent the gradient descent solution

<br>

*Figure 1. LR=0.1*

![image](https://user-images.githubusercontent.com/54302889/146252344-7fecb358-7234-4746-8f33-ddb234a5e6fb.png)

<br>

*Figure 2.*

![image](https://user-images.githubusercontent.com/54302889/146253095-e3bca369-a199-43c5-bb4e-e18ad3b455d6.png)

<br>

*Figure 3. Bonus*

![image](https://user-images.githubusercontent.com/54302889/146253187-fed214f6-2184-4413-98b1-651ac925c561.png)

<br>

<br>

*Figure 4. LR=0.7*

![image](https://user-images.githubusercontent.com/54302889/146253266-3fe88342-5a66-4b76-b3c6-849ed748b68e.png)

<br>

*Figure 5.*

![image](https://user-images.githubusercontent.com/54302889/146253411-401e6741-9030-4700-87d3-be3bd25d4f67.png)

<br>

*Figure 6. Bonus*

![image](https://user-images.githubusercontent.com/54302889/146253527-ab585422-0ef4-4417-806a-60ece5cb32dd.png)

<br>

It is observed that the analytical solutions and the gradient descent solutions are very similar. Low learning rates take more iterations to complete. Additionally, they are not as accurate because the termination threshold is reached more easily, this is due to parameters change very little with a low learning rate. It makes more sense to keep the learning rate high for quicker and more accurate results. However, the functions converge and are not able to find a solution when the learning rate is too large. The best learning rate I found was 0.7, it gave the lowest loss function value in minimum number of iterations, anything larger than 0.7 resulted in convergence.

The next figure show the result of the seconds dataset using linear regression model, while the other two figures use polynomial regression models with learning rates 0.001 and 0.01, with the degree of 3.

<br>

*Figure 7. LR=0.05*

![image](https://user-images.githubusercontent.com/54302889/146325097-5eaac5b8-fdb4-4958-a328-70d0e5e4b3ef.png)

<br>

*Figure 8. LR=0.001*

![image](https://user-images.githubusercontent.com/54302889/146325167-df4e42c9-52ae-4be0-8538-c6b2800ea5e9.png)

<br>

*Figure 9. LR=0.01*

![image](https://user-images.githubusercontent.com/54302889/146325210-7e9d3b34-809f-4120-8a0c-f49be266fa7b.png)

*Figure 9. LR=0.01*

<br>

I accidentally set the degree to 33 instead of 3 and got this graph, unfortunately the rather complicated function does not fit the screen.

*Figure 10. LR=0.01*

![image](https://user-images.githubusercontent.com/54302889/146325317-2ca37d60-bd72-4b5d-b80c-6248adc41639.png)
