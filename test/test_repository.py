from database.repository import save_assessment, get_assessment

assessment_id = save_assessment(
    thread_id="test-sbin-session-001",
    ticker="SBIN.NS",
    profile="moderate",
    horizon="medium",
    risk_verdict="WATCH",
    return_potential="medium",
    combined_verdict="Sample assessment for persistence testing",
    report_markdown="# SBIN.NS Assessment\nTest report",
    report_json={"test": True},
)

print(get_assessment(assessment_id))