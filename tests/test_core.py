"""Unit tests for qrp package core modules."""
import unittest
import numpy as np
import pandas as pd
from qrp.weights import erc_weights, sp3_weights, target_leverage, compute_risk_contributions
from qrp.metrics import sharpe, max_drawdown, sortino, calmar


class TestWeightsERC(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a realistic covariance matrix (3 assets with different vols)
        vols = np.array([0.15, 0.05, 0.20])  # equity-like, bond-like, commodity-like
        corr = np.array([[1.0, -0.1, 0.4],
                         [-0.1, 1.0, 0.1],
                         [0.4, 0.1, 1.0]])
        D = np.diag(vols)
        self.cov = D @ corr @ D

    def test_erc_weights_sum_to_one(self):
        w = erc_weights(self.cov)
        self.assertAlmostEqual(w.sum(), 1.0, places=5)

    def test_erc_weights_all_positive(self):
        w = erc_weights(self.cov)
        self.assertTrue(np.all(w > 0))

    def test_erc_risk_contributions_balanced(self):
        w = erc_weights(self.cov)
        rc = compute_risk_contributions(w, self.cov)
        # Risk contributions should be approximately equal
        rc_pct = rc / rc.sum()
        expected = 1.0 / len(w)
        for pct in rc_pct:
            self.assertAlmostEqual(pct, expected, delta=0.05)


class TestWeightsSP3(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 6 assets: 2 equity, 2 FI, 2 commodity
        vols = np.array([0.16, 0.18, 0.04, 0.06, 0.22, 0.25])
        n = len(vols)
        corr = np.eye(n)
        # Add some correlations
        corr[0, 1] = corr[1, 0] = 0.8  # equity pair
        corr[2, 3] = corr[3, 2] = 0.6  # FI pair
        corr[4, 5] = corr[5, 4] = 0.5  # commodity pair
        D = np.diag(vols)
        self.cov = D @ corr @ D
        self.ac_indices = {
            'Equity': [0, 1],
            'Fixed Income': [2, 3],
            'Commodities': [4, 5],
        }

    def test_sp3_weights_all_positive(self):
        w = sp3_weights(self.cov, self.ac_indices, 0.10)
        self.assertTrue(np.all(w >= 0))

    def test_sp3_weights_fi_dominant(self):
        """Fixed income should have highest total weight (lowest vol asset class)."""
        w = sp3_weights(self.cov, self.ac_indices, 0.10)
        fi_weight = sum(w[i] for i in self.ac_indices['Fixed Income'])
        eq_weight = sum(w[i] for i in self.ac_indices['Equity'])
        self.assertGreater(fi_weight, eq_weight)

    def test_sp3_leveraged(self):
        """Total weight should exceed 1 (leveraged portfolio)."""
        w = sp3_weights(self.cov, self.ac_indices, 0.10)
        self.assertGreater(w.sum(), 1.0)

    def test_target_leverage_positive(self):
        w = erc_weights(self.cov)
        lev = target_leverage(w, self.cov, 0.10)
        self.assertGreater(lev, 0)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.0004, 0.01, 1000))

    def test_sharpe_positive_for_positive_returns(self):
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        self.assertGreater(sharpe(rets), 0)

    def test_max_drawdown_non_positive(self):
        dd = max_drawdown(self.returns)
        self.assertLessEqual(dd, 0)

    def test_max_drawdown_known(self):
        """Create series with known drawdown."""
        rets = pd.Series([0.1, 0.1, -0.3, 0.05])
        dd = max_drawdown(rets)
        # After +10%, +10% (cumulative 1.21), then -30% (cumulative 0.847), DD = 0.847/1.21 - 1 = -0.3
        self.assertAlmostEqual(dd, -0.3, places=1)

    def test_sortino_positive(self):
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        self.assertGreater(sortino(rets), 0)

    def test_calmar_positive(self):
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        self.assertGreater(calmar(rets), 0)


if __name__ == '__main__':
    unittest.main()
