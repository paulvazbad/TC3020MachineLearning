from custom_ml_implementation.LinearRegression import LinearRegression
from custom_ml_implementation.LogisticRegression import LogisticRegression

def main():
    linear_regression = LinearRegression() 
    logistic_regression = LogisticRegression()
    print(logistic_regression.J([[0],[1]],[0,1],[1,1]))

if __name__ == "__main__":
    main()