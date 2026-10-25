from app.analyst.answer import Answer, render_markdown
from app.analyst.semantic import get_semantic
from app.analyst.sim import run_sim


def test_only_single_selects_run():
    s = get_semantic()
    for sql in ["DROP TABLE prices", "DELETE FROM prices", "SELECT 1; DROP TABLE prices",
                "COPY prices TO 'x.csv'", "CREATE TABLE x AS SELECT 1"]:
        assert s.run(sql, "q").error.startswith("rejected"), sql
    assert s.run("SELECT * FROM read_csv('pyproject.toml')", "q").error  # no file access
    assert s.run("SELECT COUNT(*) FROM prices", "q").rows[0][0] > 0  # tables survived


def test_canonical_metrics_and_the_simpsons_trap():
    s = get_semantic()
    city = s.cities[0]
    idx = s.run(f"SELECT * FROM chain_price_index('{city}', NULL)", "q").frame()
    assert {"chain", "n_products", "index", "pct_vs_median"} <= set(idx.columns) and len(idx) >= 2
    assert list(idx["index"]) == sorted(idx["index"])  # ordered cheapest first
    spread = s.run(f"SELECT * FROM price_spread('{city}', NULL) LIMIT 5", "q").frame()
    assert (spread.max_price >= spread.min_price).all()
    assert s.run(f"SELECT * FROM promo_share('{city}')", "q").error is None


def test_errors_are_returned_not_raised():
    r = get_semantic().run("SELECT nope FROM prices", "q7")
    assert r.id == "q7" and "nope" in r.error and r.rows == []


def test_render_and_sim():
    a = Answer(headline="X is cheapest.", claims=[], method="m", assumptions=["a"], caveats=[], chart=None,
               declined=False)
    md = render_markdown(a)
    assert "X is cheapest." in md and "Assumptions" in md
    r = run_sim("Which chain is cheapest in Haifa?", "a1")
    assert r.answer and not r.answer["declined"] and r.queries[0]["error"] is None
    assert all(c["query_id"] == "q1" for c in r.answer["claims"])
    assert run_sim("tell me a joke", "a1").answer["declined"]
