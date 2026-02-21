# src/game.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import os
import csv


@dataclass(frozen=True)
class PayoffMatrix:
    """
    Symmetric 2x2 game payoff parameters (row player perspective).
      R: payoff for (C,C)
      T: payoff for (D,C)  (temptation)
      S: payoff for (C,D)  (sucker)
      P: payoff for (D,D)
    """
    R: float
    T: float
    S: float
    P: float

    def payoff(self, a_i: int, a_j: int) -> Tuple[float, float]:
        """
        a=1 -> C, a=0 -> D
        Return (r_i, r_j).
        """
        if a_i == 1 and a_j == 1:
            return self.R, self.R
        if a_i == 1 and a_j == 0:
            return self.S, self.T
        if a_i == 0 and a_j == 1:
            return self.T, self.S
        return self.P, self.P

    def pi_C(self, x: float) -> float:
        """Expected payoff of choosing C against population cooperation rate x."""
        return x * self.R + (1.0 - x) * self.S

    def pi_D(self, x: float) -> float:
        """Expected payoff of choosing D against population cooperation rate x."""
        return x * self.T + (1.0 - x) * self.P

    def replicator_equilibrium(self) -> Optional[float]:
        """
        Interior equilibrium x* in (0,1) where pi_C(x*) = pi_D(x*):
          x(R-S) + S = x(T-P) + P
          x* = (P-S) / ((R-S) - (T-P))
        """
        denom = (self.R - self.S) - (self.T - self.P)
        if abs(denom) < 1e-12:
            return None
        x_star = (self.P - self.S) / denom
        if 0.0 < x_star < 1.0:
            return x_star
        return None


def default_payoff_hawk_dove() -> PayoffMatrix:
    """
    A Hawk–Dove style firm co-opetition game with stable interior x* = 0.5:
      (C,C)=(3,3), (C,D)=(1,4), (D,C)=(4,1), (D,D)=(0,0)
    """
    return PayoffMatrix(R=3.0, T=4.0, S=1.0, P=0.0)


def load_payoff_time_series(
    csv_path: str,
    year_col: str = "year",
    cols: Tuple[str, str, str, str] = ("R", "T", "S", "P"),
) -> Dict[int, PayoffMatrix]:
    """
    Optional: load time-varying payoff matrices from data/cordis_processed/payoff_time_series.csv
    Expected columns: year, R, T, S, P
    Returns {year: PayoffMatrix}.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Payoff time series not found: {csv_path}")

    out: Dict[int, PayoffMatrix] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row[year_col])
            R = float(row[cols[0]])
            T = float(row[cols[1]])
            S = float(row[cols[2]])
            P = float(row[cols[3]])
            out[y] = PayoffMatrix(R=R, T=T, S=S, P=P)
    return out


def payoff_from_series(series: Dict[int, PayoffMatrix], year: int) -> PayoffMatrix:
    """
    Get payoff for a given year. If the exact year doesn't exist, use nearest previous year.
    """
    if year in series:
        return series[year]
    years = sorted(series.keys())
    prev = [y for y in years if y <= year]
    if prev:
        return series[prev[-1]]
    return series[years[0]]
