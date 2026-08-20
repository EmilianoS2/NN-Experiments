# Pricing Basket Options: Monte Carlo vs. MLP vs. Black–Scholes

*08/20/2026*

---

## 1. The Question

This experiment asks whether an MLP is useful as a fast surrogate for Monte Carlo option pricing. The idea is to run an expensive simulator ahead of time, train the MLP on its results, and then use the trained network to price new contracts without rerunning thousands of simulated paths.

My goal was to find a problem where a neural network was not merely usable, but the best available option. By that standard, this experiment failed. A moment-matched Black–Scholes approximation was both simpler and more accurate than the MLP on the held-out test set.

The failure still produced the most important lesson of the project: Monte Carlo and neural networks can form a powerful combination when no sufficiently accurate equation exists. Monte Carlo handles the complicated pricing problem; the MLP compresses those expensive calculations into a fast reusable function. This particular option was simply not complicated enough to require that combination.

## 2. The Contract

The contract is a European call option on an equally weighted basket of five assets. It can only be exercised at expiration. At that point, the five asset prices are averaged and the payoff is:

```text
payoff = max(average terminal asset price - strike, 0)
```

If the basket finishes below the strike, the option expires worthless. If it finishes above the strike, the option pays the difference.

The model receives 14 inputs:

- Five current asset prices
- Five asset volatilities
- Strike price
- Time to maturity
- Risk-free interest rate
- One shared correlation value

The target is one number: the Monte Carlo estimate of the option's current price.

## 3. The Data

This project does not use actual market option prices. It generates 2,000 synthetic contracts with a fixed random seed.

| input | sampled range |
|---|---:|
| each asset price | $80–$120 |
| each volatility | 10%–50% |
| strike | $80–$120 |
| maturity | 0.25–2 years |
| risk-free rate | 1%–5% |
| shared asset correlation | 0.00–0.80 |

The ranges define the domain in which the MLP is intended to be reliable. A neural network should not be trusted to extrapolate far outside them.

An early version sampled strikes from $400 to $600 even though the equally weighted basket was worth roughly $100. Almost every option was therefore worthless, which created mostly zero labels and a misleadingly tiny loss. Correcting the strike range to $80–$120 produced a meaningful distribution of option prices.

## 4. Monte Carlo Labels

For each synthetic contract, Monte Carlo generates 2,000 possible terminal outcomes for the five correlated assets. Every path produces an option payoff. The pricer averages those payoffs and discounts the average back to today:

```text
random asset outcomes
        ↓
payoff from each outcome
        ↓
average payoff
        ↓
discounted Monte Carlo price
```

No individual simulated path is the output. The output is the discounted average payoff across all paths. The notebook also records the Monte Carlo standard error, which measures the sampling uncertainty caused by using a finite number of paths.

These prices are theoretical labels generated under correlated geometric Brownian motion. They are not observed market prices, and they only reflect the assumptions built into the simulator.

## 5. The Models

### MLP surrogate

The MLP receives the 14 contract and market inputs and returns one nonnegative option price:

```text
14 inputs → 64 ReLU → 64 ReLU → 1 Softplus output
```

It has 5,185 trainable parameters. Softplus is used on the final layer because an option price cannot be negative. The network is trained for 2,000 epochs with Adam and mean squared error.

The MLP is not simulating paths at prediction time. It is learning the function connecting contract inputs to the final prices previously produced by Monte Carlo.

### Moment-matched Black–Scholes

Standard Black–Scholes prices an option on one lognormal asset, not an arithmetic basket of five correlated assets. The baseline therefore replaces the basket with one synthetic lognormal asset whose first two terminal-price moments match the basket. Black–Scholes is then applied to that approximation.

This is not an exact closed-form price for the original basket. It is an equation-based approximation. It is nevertheless fast, deterministic, and well matched to this experiment because the simulated assets already follow the same geometric Brownian-motion assumptions that support Black–Scholes.

## 6. Keeping It Honest (Methodology)

The 2,000 scenarios are split 70% / 15% / 15%:

- 1,400 training contracts
- 300 validation contracts
- 300 test contracts

Input normalization statistics are calculated from the training split only and then applied to validation and test. The test set is held out from MLP training.

Both the MLP and Black–Scholes approximation are evaluated against the same 2,000-path Monte Carlo labels. This comparison answers, "Which method best reproduces these Monte Carlo estimates?" It does not prove which method is closest to real market prices, or even to an infinitely precise Monte Carlo value. A stronger follow-up would use a much larger number of paths to create a high-precision test benchmark.

MAE is the average absolute pricing error in dollars. RMSE penalizes large errors more heavily. Lower is better for both.

## 7. Results

| model | MAE | RMSE |
|---|---:|---:|
| **moment-matched Black–Scholes** | **0.3133** | **0.4206** |
| MLP | 0.5289 | 0.6472 |

The MLP generally follows the Monte Carlo prices and produces reasonable predictions, so it did learn the pricing relationship. However, the Black–Scholes approximation is better on both metrics while requiring no dataset, training loop, optimizer, or learned parameters.

By the original success criterion, the MLP lost. An equation-based approximation solved this version of the problem more accurately and with less complexity.

## 8. What It Means

The mistake was not that the MLP failed to learn. The mistake was choosing a problem whose structure was already captured by a strong mathematical model.

The simulator assumes:

- Geometric Brownian motion
- Constant volatility for each asset
- Constant correlation
- A European payoff determined only at expiration
- No jumps, barriers, early exercise, or changing volatility

Those assumptions make the option unusually friendly to a Black–Scholes-style approximation. The MLP has flexibility that the problem does not need, and that flexibility introduces training error without adding useful information.

This leads to a better rule for choosing neural-network projects:

> Do not ask whether a neural network can solve the problem. Ask whether it adds something that the strongest simpler method cannot provide.

For this contract, it does not. Black–Scholes is faster, simpler, more interpretable, and more accurate.

## 9. The Successful Lesson

The useful idea discovered here is the Monte Carlo–MLP surrogate pattern:

```text
No useful pricing equation
        ↓
Monte Carlo produces accurate but expensive labels
        ↓
MLP learns the input-to-price relationship once
        ↓
MLP produces fast prices for many new scenarios
```

This becomes valuable when all three conditions are true:

1. No sufficiently accurate closed-form or cheap numerical approximation exists.
2. Each high-quality simulation is expensive.
3. Many related contracts or risk scenarios must be priced repeatedly.

A better follow-up experiment would use a genuinely path-dependent exotic option, such as an arithmetic Asian basket option with stochastic volatility or barriers. Monte Carlo can represent those features, while ordinary Black–Scholes cannot. The MLP would then compete against low-path Monte Carlo and simpler approximations under an equal runtime budget, with high-path Monte Carlo serving as the reference.

The MLP should be considered successful only if it offers the best accuracy available at the required prediction speed. That would demonstrate a real need for the neural network rather than merely showing that one can be trained.

## 10. Files

| file | what it does |
|---|---|
| [01_data_gathering.ipynb](01_data_gathering.ipynb) | generates the 2,000 synthetic market and contract scenarios |
| [02_data_preparation.ipynb](02_data_preparation.ipynb) | validates the inputs and creates tensor-ready features |
| [03_monte_carlo.ipynb](03_monte_carlo.ipynb) | simulates correlated terminal prices and creates pricing labels |
| [04_train_mlp.ipynb](04_train_mlp.ipynb) | trains the MLP and compares it with Monte Carlo and Black–Scholes |
| [data/raw_scenarios.csv](data/raw_scenarios.csv) | synthetic inputs before pricing |
| [artifacts/labeled_scenarios.csv](artifacts/labeled_scenarios.csv) | synthetic inputs with Monte Carlo prices and standard errors |

