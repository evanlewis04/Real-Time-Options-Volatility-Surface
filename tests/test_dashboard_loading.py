from src.dashboard.loading import LoadingState, render_empty_state, render_loading_state


def test_render_loading_state_includes_dense_skeleton_markup():
    html = render_loading_state(
        LoadingState(
            title="AAPL option chain",
            detail="Fetching yfinance expirations and bid/ask fields.",
            stage="option chain",
            rows=5,
        )
    )

    assert "AAPL option chain" in html
    assert "option chain" in html
    assert "skeleton-line" in html
    assert html.count("skeleton-line") >= 5
    assert "FETCH" in html


def test_render_loading_state_escapes_user_supplied_text():
    html = render_loading_state(
        LoadingState(
            title="<script>alert(1)</script>",
            detail="x < y",
            stage="history",
            rows=1,
        )
    )

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "x &lt; y" in html


def test_render_empty_state_has_recovery_action():
    html = render_empty_state(
        "Correlation matrix unavailable",
        "Not enough historical closes were returned.",
        "Refresh data or reduce the universe.",
    )

    assert "Correlation matrix unavailable" in html
    assert "Not enough historical closes" in html
    assert "Refresh data or reduce the universe." in html
    assert "empty-action" in html
