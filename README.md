# Forecasting S&P 500 Realized Volatility: GARCH vs. MLP vs. LSTM

*07/06/2026*

---

## 1. The Question
I am predicting future volatility, the question I want to know is if the signal is in the input. This is why the project tests on volatility, in which case the answer is yes, over direction, in which case the answer is mostly no. This project is a comparison of a neural network (MLP and LSTM) to an equation (GARCH).

## 2. The Target
Realized volatility is the square root of sum of returns squared over the window h. H equals one day, here. This experiment actually uses log realized volatility rather than just realized volatility because volatility is positive and right-skewed, log ensures symmetry and prevents the model from expecting a negative volatility. The cardinal sin here is ensuring we do not contaminate the data with the target, or else we introduce a lookahead bias.

## 3. The Data
S&P500 data is taken from yfinance starting from 1962. The reason is data prior to 1962 had all OHLC values share the close price of the asset. For the best results I wanted to ensure data was as clean as possible so I decided to exclude those from the experiment.

## 4. The Three Models
GARCH is the classical baseline for predicting volatility. It takes 1 input which is log returns, and learns 3 parameters, omega, alpha, beta. Omega is the baseline variance, alpha is how strongly volatility reacts to a fresh shock, beta is the time a spike lasts. From these values it can estimate tomorrow's volatility.
MLP or multi-layer perceptron takes in 8 features of my choosing. These features run through a network of three layers, this includes two hidden layers and one output layer. I used ReLU activations and returned a linear output.
LSTM or long short term memory takes plain 30 day returns, the network is in charge of deciding what to focus on. The LSTM is a one layer network using a dropout of 20% to regularize the network and prevent overfitting.


## 5. Keeping It Honest (Methodology)
Out of the roughly 16,000 days of data, I used a 70% train, 15% validate, and 15% test chronological split. That works out to roughly 11k train and 2.5k for validate and test. Models were fine tuned on their validation data, test was only touched once. The numbers below represent each model on their test split, it is the average gap between predicted and actual volatility in percentage points, lower is better. A 1.00 score represents a gap of 1 percentage point from predicted to actual, whereas 0.00 represents a direct hit. 

## 6. Results

| Model | What it sees | h=1 RMSE | h=5 RMSE |
|---|---|---|---|
| GARCH(1,1) | returns only, 3 params | 0.81 | 1.26 |
| MLP | 8 engineered features | ~0.82 | ~1.20 |
| LSTM | raw 30-day sequences | 0.81 | 1.20 |

*RMSE is in percentage points; lower is better. The h=1 and h=5 columns are on different scales (a 5-day vol is ~√5× larger), so compare **within** a column, not across.*

## 7. What It Means
The three different models roughly tie. Although I would not say that means they are equal. In terms of pure results neither one has an edge over another, but in other metrics like ease of use, consistency, and reliability, the GARCH model wins with MLP in second and LSTM in third. GARCH is the simplest, most reliable, and consistent model since it is just an equation, MLP is second place, it does require to be trained, however, it is simpler than the LSTM and gets roughly the same results. LSTM is the worst of the three being the most complex and requiring the most technical knowledge to use, its results are equal to GARCH and roughly equal to MLP. The results being so similar actually showcases that the bottleneck is the data, plain price action plateaus here.

## 8. H=5 Results
Increasing h to 5, the model is now predicting the total volatility over the next 5 days. Same as with h = 1, I am testing the same 3 models and training splits remain the same, models can only make predictions on the future using past data. As a result of the smoother, less noisy target, the networks were able to outperform GARCH on some occasions, I should add however, that was over a couple of trials, I cannot confidently say the networks beat GARCH. Without further testing, I must conclude the rankings stay the same, GARCH in 1st, MLP in 2nd, and LSTM in 3rd. I showcased the best scores of the two networks, both of them vary massively from that. LSTM skewed 0.12 points off, and MLP varied 0.07 points. Although, the models were more accurate, loss was significantly lower for both models with both of them reaching below 0.3 as before when they were at 1.5 for h = 1. This drop indicates the the h=1 result was capped by target noise instead of model weakness.