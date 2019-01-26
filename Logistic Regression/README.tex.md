# Logistic Regression

This is a dummy example that you can play with and modify to integrate in your prediction algorithms. The situation is as follows:

* the model is: 
                                            $$y_t = 2 \, x_t^1 - 3 \, x_t^2 +1$$
* Our task is: given the observations $(x_t, y_t)_t$, recover the weights and the bias.

This can be easily solved using a linear regression, but we will add an additional layer of complexity by supposing that we only observe the $Z$ variable defined as:
                                                  $$Z_t = Ber(\sigma(y_t))$$
Where $\sigma$ is the sigmoid function:
                                            $$\sigma(y) = \frac{1}{1 + e^{-y}}$$                                            
The task is now to recover the weights from the obervations $(x_t, Z_t)$.
