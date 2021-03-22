# Custom ML Implementation
Package for Python >3.5 with Linear Regression and Logistic Regression modules.

## Project structure
- custom_ml_implementation: source code of the package
- tests: unit tests
- dist: buildfiles (wheels and tar.gz) files

## How to install
1. Copy the `custom_ml_implementation-{VERSION}-py3-none-any.wheel` file into your project root folder

2. Run the pip install command:  
`python -m pip install custom_ml_implementation-{VERSION}-py3-none-any.wheel`

3. Verify the installation: \
`python -m pip freeze | grep custom`

## Examples

### How to train a linear regression model

```python

from custom_ml_implementation.LinearRegression import LinearRegression

linear_regression = LinearRegression() # Create LinearRegression instance

# Modify training parameters
linear_regression.LEARNING_RATE = 0.1
linear_regression.MAX_ITER = 1000
linear_regression.TARGET_ERROR = 0.001

#Train 
train_x = [[0],[1],[2],[3],[4],[5],[6],[7]]
train_y = [2,3,4,5,6,7,8,9]
resulting_theta = linear_regression.train(train_x,train_y)
```