# portfolio_models.py

import numpy as np
import pandas as pd
import cvxpy as cp

def mean_variance_opt(mu, Sigma, allow_short=False):
    """
    Mean-Variance Optimizer: maximize mu^T w - 0.5*w^T Sigma w
    """
    n = len(mu)
    w = cp.Variable(n)
    gamma = 1.0  # Risk aversion parameter
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints += [w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

def min_variance_opt(Sigma, allow_short=False):
    """
    Minimum-Variance Portfolio: minimize w^T Sigma w
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints += [w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

def risk_parity_opt(Sigma, allow_short=False):
    """
    Risk Parity Portfolio (Equal Risk Contribution)
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    portfolio_var = cp.quad_form(w, Sigma)
    rc = [w[i] * (Sigma @ w)[i] for i in range(n)]
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints += [w >= 0]
    objective = cp.Minimize(cp.sum_squares(rc - portfolio_var / n))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

def sentiment_adjusted_mu(mu, sentiment, sentiment_beta=0.1):
    """
    Integrate sentiment: boost expected returns by a multiple of the sentiment index.
    """
    return mu + sentiment_beta * sentiment

# ------------------------
# Main script
# ------------------------
if __name__ == '__main__':
    # Load your features
    df = pd.read_csv('./data/model_input/features_with_sentiment.csv')

    # Automatically find all coins based on columns: return_*
    returns_cols = [col for col in df.columns if col.startswith('return_')]
    coins = [c.replace('return_', '') for c in returns_cols]

    df = df.dropna(subset=returns_cols + ['mean_sentiment'])
    rebalance_freq = 7  # Change as needed (e.g. 30 for monthly)

    weights_all = []
    dates = []
    for i in range(0, len(df) - rebalance_freq, rebalance_freq):
        window = df.iloc[i:i+rebalance_freq]
        mu = window[returns_cols].mean().values
        Sigma = np.cov(window[returns_cols].values, rowvar=False)
        sentiment = window['mean_sentiment'].mean()
        mu_adj = sentiment_adjusted_mu(mu, sentiment)
        # Choose optimizer: mean-variance, min-var, risk parity
        w_opt = mean_variance_opt(mu_adj, Sigma, allow_short=False)
        weights_all.append(w_opt)
        # Use the next rebalance date for the timestamp (i+rebalance_freq)
        dates.append(df.iloc[i+rebalance_freq]['date'])

    # Save weights for later use
    weights_df = pd.DataFrame(weights_all, columns=coins)
    weights_df['date'] = dates
    weights_df.to_csv('./data/model_input/portfolio_weights.csv', index=False)
    print("[💾] Optimized weights saved for coins:", coins)
