
from dataclasses import dataclass, field

import pandas as pd
import pulp

REQUIRED_COLUMNS = ["Product", "Material_Cost", "Labor_Hours", "Selling_Price", "Max_Units"]


@dataclass
class OptimizationResult:
    status: str
    total_profit: float = 0.0
    total_revenue: float = 0.0
    budget_used: float = 0.0
    hours_used: float = 0.0
    budget: float = 0.0
    hours_limit: float = 0.0
    unused_budget: float = 0.0   # zero-waste: money left idle
    idle_hours: float = 0.0      # zero-waste: working time left idle
    plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    shadow_price_budget: float = 0.0
    shadow_price_time: float = 0.0
    lp_upper_bound: float = 0.0


def _validate(df: pd.DataFrame, budget: float, hours_limit: float) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if budget < 0 or hours_limit < 0:
        raise ValueError("Budget and available hours must be non-negative.")
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True).copy()
    for col in ["Material_Cost", "Labor_Hours", "Selling_Price", "Max_Units"]:
        if (df[col] < 0).any():
            raise ValueError(f"'{col}' cannot contain negative values.")
    return df


def _build_model(df: pd.DataFrame, budget: float, hours_limit: float, cat: str):
    """cat = 'Integer' (real problem) or 'Continuous' (LP relaxation)."""
    prob = pulp.LpProblem("Artisano_Daily_Profit", pulp.LpMaximize)
    x = {
        i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=int(df.at[i, "Max_Units"]), cat=cat)
        for i in df.index
    }
    profit = {i: df.at[i, "Selling_Price"] - df.at[i, "Material_Cost"] for i in df.index}

    prob += pulp.lpSum(profit[i] * x[i] for i in df.index), "Total_Profit"
    prob += pulp.lpSum(df.at[i, "Material_Cost"] * x[i] for i in df.index) <= budget, "Budget"
    prob += pulp.lpSum(df.at[i, "Labor_Hours"] * x[i] for i in df.index) <= hours_limit, "Time"
    return prob, x


def optimize_production(df: pd.DataFrame, budget: float, hours_limit: float) -> OptimizationResult:
    """Return the profit-maximising daily production plan."""
    df = _validate(df, budget, hours_limit)
    if df.empty:
        return OptimizationResult(status="No products", budget=budget, hours_limit=hours_limit)

    # 1) Exact integer solution
    prob, x = _build_model(df, budget, hours_limit, "Integer")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return OptimizationResult(status=status, budget=budget, hours_limit=hours_limit)

    plan = df.copy()
    plan["Units_To_Make"] = [int(round(x[i].value() or 0)) for i in df.index]
    plan["Unit_Profit"] = plan["Selling_Price"] - plan["Material_Cost"]
    plan["Total_Cost"] = plan["Units_To_Make"] * plan["Material_Cost"]
    plan["Total_Hours"] = plan["Units_To_Make"] * plan["Labor_Hours"]
    plan["Revenue"] = plan["Units_To_Make"] * plan["Selling_Price"]
    plan["Profit"] = plan["Units_To_Make"] * plan["Unit_Profit"]

    
    relaxed, _ = _build_model(df, budget, hours_limit, "Continuous")
    relaxed.solve(pulp.PULP_CBC_CMD(msg=False))
    sp_budget = relaxed.constraints["Budget"].pi or 0.0
    sp_time = relaxed.constraints["Time"].pi or 0.0

    budget_used = float(plan["Total_Cost"].sum())
    hours_used = float(plan["Total_Hours"].sum())
    return OptimizationResult(
        status=status,
        total_profit=float(plan["Profit"].sum()),
        total_revenue=float(plan["Revenue"].sum()),
        budget_used=budget_used,
        hours_used=hours_used,
        budget=budget,
        hours_limit=hours_limit,
        unused_budget=max(budget - budget_used, 0.0),
        idle_hours=max(hours_limit - hours_used, 0.0),
        plan=plan,
        shadow_price_budget=float(sp_budget),
        shadow_price_time=float(sp_time),
        lp_upper_bound=float(pulp.value(relaxed.objective) or 0.0),
    )
