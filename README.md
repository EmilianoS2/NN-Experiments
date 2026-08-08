# NN-Experiments

A running set of small, honest machine-learning experiments built around one governing question:

> **Is the signal actually in the input?**

A neural network can only learn what is genuinely there. Each project tests that idea on a problem where the answer differs — pairing every model against a brutal baseline, respecting the discipline that separates real ML from backtested noise (no leakage, honest splits), and writing up *why* the result came out the way it did.

## Projects

### [Volatility Forecasting](volatility/) — *signal: yes*
Forecasting S&P 500 realized volatility with **GARCH vs. an MLP vs. an LSTM**. Volatility clusters, so the signal exists — and all three models converge to nearly the same accuracy. The lesson: when the signal is shallow, a 3-parameter classical model ties a neural net, and complexity has to earn its place. → **[Read the write-up](volatility/README.md)**

### [Prime Gap Prediction](primes/) — *signal: present but unusable*
The counter-example. Six models predict the gap to the next prime, from a dumb constant up to a transformer. Linear regression wins — 6 parameters beating 8,801 — and every model explains under 3% of the variance. The next prime is *fully determined* by the current one, yet almost none of it is reachable from recent gaps. The lesson: the question isn't whether the information is in the input, it's whether it's there in a form the model's inductive bias can use. → **[Read the write-up](primes/README.md)**

---

*Each project stands alone in its own folder with its own write-up. Built as an exercise in doing ML honestly: real baselines, leak-free evaluation, and a written explanation of the result — not just working code.*
