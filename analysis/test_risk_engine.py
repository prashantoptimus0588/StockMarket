import itertools

import pytest

from models.models import FundamentalsData, PriceData, UserPrefs
from risk_engine import HORIZONS, PROFILES, evaluate_risk


def good_price(**kw):
    base = dict(
        ticker="TEST.NS", latest_close=100, volatility_pct=20, max_drawdown_pct=-15,
        current_drawdown_pct=-5, avg_volume=1_000_000,
    )
    base.update(kw)
    return PriceData(**base)


def good_fund(**kw):
    base = dict(
        ticker="TEST.NS", sector="Industrials", beta=0.8, trailing_pe=22, roe_pct=15,
        profit_margin_pct=12, earnings_growth_pct=10, debt_to_equity=0.3, free_cash_flow=1e9,
    )
    base.update(kw)
    return FundamentalsData(**base)


def status(verdict, name):
    return next(r.status for r in verdict.rules if r.name == name)


def test_strong_stock_fits_conservative_long():
    v = evaluate_risk(good_price(), good_fund(), UserPrefs(profile="conservative", horizon="long"))
    assert v.result == "fits" and v.score == 100.0


@pytest.mark.parametrize("profile,horizon", list(itertools.product(PROFILES, HORIZONS)))
def test_all_nine_combinations_run(profile, horizon):
    v = evaluate_risk(good_price(), good_fund(), UserPrefs(profile=profile, horizon=horizon))
    assert v.result == "fits"
    assert v.rules_total == 11


def test_volatile_stock_rejected_for_conservative_short():
    p = good_price(volatility_pct=60, max_drawdown_pct=-60, current_drawdown_pct=-45)
    v = evaluate_risk(p, good_fund(), UserPrefs(profile="conservative", horizon="short"))
    assert v.result == "does_not_fit"


def test_profile_changes_threshold():
    p = good_price(volatility_pct=45)
    cons = evaluate_risk(p, good_fund(), UserPrefs(profile="conservative", horizon="long"))
    aggr = evaluate_risk(p, good_fund(), UserPrefs(profile="aggressive", horizon="long"))
    assert status(cons, "Volatility") == "fail"
    assert status(aggr, "Volatility") == "pass"


def test_empty_data_gives_no_verdict():
    v = evaluate_risk(PriceData(ticker="X"), FundamentalsData(ticker="X"),
                      UserPrefs(profile="balanced", horizon="medium"))
    assert v.score is None and v.result == "borderline"
    assert v.rules_evaluated == 0


def test_none_inputs_do_not_crash():
    v = evaluate_risk(None, None, UserPrefs(profile="balanced", horizon="medium"))
    assert v.score is None


def test_bank_exempt_from_debt_rule():
    prefs = UserPrefs(profile="balanced", horizon="long")
    bank = evaluate_risk(good_price(), good_fund(sector="Financial Services", debt_to_equity=15), prefs)
    other = evaluate_risk(good_price(), good_fund(debt_to_equity=15), prefs)
    assert status(bank, "Debt/equity") == "skipped"
    assert status(other, "Debt/equity") == "fail"


def test_heavy_failure_caps_at_borderline():
    # volatility 30 > 28 limit; weight 3 for short horizon -> score 83.3 but capped
    v = evaluate_risk(good_price(volatility_pct=30), good_fund(),
                      UserPrefs(profile="conservative", horizon="short"))
    assert v.score >= 70
    assert v.result == "borderline"


def test_missing_fundamentals_still_scores_if_coverage_ok():
    v = evaluate_risk(good_price(), FundamentalsData(ticker="TEST.NS"),
                      UserPrefs(profile="balanced", horizon="short"))
    assert v.score == 100.0
    assert status(v, "Beta") == "skipped"