import pytest

from models.models import FundamentalsData, PriceData, UserPrefs
from return_engine import combine, evaluate_return
from risk_engine import evaluate_risk


def price(**kw):
    base = dict(ticker="T.NS", latest_close=90, high_52w=100, low_52w=60, volatility_pct=20,
                max_drawdown_pct=-15, current_drawdown_pct=-5, avg_volume=1_000_000)
    base.update(kw)
    return PriceData(**base)


def fund(**kw):
    base = dict(ticker="T.NS", sector="Industrials", beta=0.8, trailing_pe=14, forward_pe=11,
                roe_pct=22, profit_margin_pct=20, revenue_growth_pct=18, earnings_growth_pct=25,
                debt_to_equity=0.3, free_cash_flow=1e9, dividend_yield_pct=3)
    base.update(kw)
    return FundamentalsData(**base)


def weak_fund():
    return fund(trailing_pe=60, forward_pe=70, roe_pct=2, profit_margin_pct=-2,
                revenue_growth_pct=-5, earnings_growth_pct=-20, dividend_yield_pct=0)


P = lambda profile="balanced", horizon="medium": UserPrefs(profile=profile, horizon=horizon)


def test_strong_company_is_high():
    r = evaluate_return(price(), fund(), P())
    assert r.level == "high" and r.score >= 65


def test_weak_company_is_low():
    r = evaluate_return(price(latest_close=62), weak_fund(), P())
    assert r.level == "low"


def test_empty_data_no_verdict():
    r = evaluate_return(PriceData(ticker="X"), FundamentalsData(ticker="X"), P())
    assert r.level is None and r.score is None


def test_none_inputs_do_not_crash():
    assert evaluate_return(None, None, P()).score is None


def test_horizon_changes_weights():
    # strong momentum, weak fundamentals: short horizon should score higher than long
    p, f = price(latest_close=99), weak_fund()
    short = evaluate_return(p, f, P(horizon="short")).score
    long_ = evaluate_return(p, f, P(horizon="long")).score
    assert short > long_


def test_combine_strong_fit():
    prefs = P("balanced", "medium")
    risk = evaluate_risk(price(), fund(), prefs)
    ret = evaluate_return(price(), fund(), prefs)
    c = combine(risk, ret, prefs)
    assert c.label == "strong_fit" and c.score >= 70


def test_poor_risk_fit_is_a_gate():
    prefs = P("conservative", "short")
    bad_price = price(volatility_pct=70, max_drawdown_pct=-60, current_drawdown_pct=-45)
    risk = evaluate_risk(bad_price, fund(), prefs)
    ret = evaluate_return(bad_price, fund(), prefs)
    assert risk.result == "does_not_fit"
    assert combine(risk, ret, prefs).label == "weak_fit"


def test_low_return_blocks_strong_fit():
    prefs = P("conservative", "long")
    risk = evaluate_risk(price(), fund(), prefs)
    ret = evaluate_return(price(latest_close=62), weak_fund(), prefs)
    c = combine(risk, ret, prefs)
    assert c.label != "strong_fit"


def test_profile_changes_return_weight():
    prefs_a, prefs_c = P("aggressive"), P("conservative")
    risk = evaluate_risk(price(), fund(), prefs_a)
    ret = evaluate_return(price(), fund(), prefs_a)
    assert combine(risk, ret, prefs_a).return_weight > combine(risk, ret, prefs_c).return_weight


def test_missing_side_gives_no_combined_score():
    prefs = P()
    risk = evaluate_risk(None, None, prefs)
    ret = evaluate_return(price(), fund(), prefs)
    c = combine(risk, ret, prefs)
    assert c.score is None and c.label == "watch"