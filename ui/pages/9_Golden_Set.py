import json

import pandas as pd
import streamlit as st

from app.analyst.semantic import get_semantic
from app.db import GoldenItem, SessionLocal, utcnow
from app.evals import golden
from components import owner_unlock, role
from lib import setup, tiles

setup("Golden set", "🏅")
st.markdown("## 🏅 Golden set · review queue")
st.caption("The benchmark that gates releases: fixed questions with reference SQL. Expected answers are computed "
           "live from today's data, so prices can change without the references going stale. It grows from three "
           "places: this queue, the Trace explorer's 'Add to golden set', and every disagreement in Chat. Those "
           "drafts start with the agent's own SQL as the reference, so check it before approving. A draft gates CI "
           "only after a human approves it here.")
owner_unlock()
is_owner = role() == "owner"

with SessionLocal() as s:
    items = [golden.as_dict(g) for g in s.query(GoldenItem).order_by(GoldenItem.id)]
if not items:
    st.info("No golden items yet. Run `python -m app.evals.golden build` (or `import`).")
    st.stop()
df = pd.DataFrame(items)
counts = df.status.value_counts().to_dict()
tiles([("Approved", str(counts.get("approved", 0)), f"of {len(df)} · gate CI", False),
       ("Drafts to review", str(counts.get("draft", 0)), "not gating yet", False),
       ("Rejected", str(counts.get("rejected", 0)), "excluded", False),
       ("From traces & chat", str(df.source.isin(["promoted", "trace", "feedback"]).sum()),
        "Review page · Trace explorer · Chat", False)])
st.progress(counts.get("approved", 0) / len(df), text=f"{counts.get('approved', 0)}/{len(df)} reviewed and approved")

f1, f2, f3 = st.columns(3)
status = f1.selectbox("Status", ["draft", "approved", "rejected", "any"])
qtype = f2.selectbox("Type", ["any"] + sorted(df.qtype.unique()))
lang = f3.selectbox("Language", ["any", "en", "he"])
view = df
if status != "any":
    view = view[view.status == status]
if qtype != "any":
    view = view[view.qtype == qtype]
if lang != "any":
    view = view[view.language == lang]
if view.empty:
    st.success("Nothing here. 🎉")
    st.stop()

ids = view.id.tolist()
# save() can't move the selectbox once it's drawn, so it leaves the next item here for the rerun
if "golden_next" in st.session_state:
    st.session_state.golden_sel = st.session_state.pop("golden_next")
if toast := st.session_state.pop("golden_toast", None):
    st.toast(toast)
if st.session_state.get("golden_sel") not in ids:
    st.session_state.golden_sel = ids[0]
sel = st.selectbox("Item", ids, key="golden_sel",
                   format_func=lambda i: f"{i} · {df.set_index('id').loc[i, 'question'][:90]}")
it = df.set_index("id").loc[sel].to_dict() | {"id": sel}

st.divider()
st.markdown(f"#### {it['question']}")
st.caption(f"{it['id']} · {it['qtype']} · {it['difficulty']} · answer type **{it['answer_type']}** · "
           f"{it['language']} · status **{it['status']}** · source {it['source']}"
           + (f" · reviewed by {it['reviewed_by']}" if it.get("reviewed_by") else ""))
if it.get("notes"):
    st.info(it["notes"])

guide = {
    "scalar": "Correct if the answer states the value in the key column (±1%).",
    "ranking": "Correct if the first entity named in the headline is the top row (ties on value_column all count).",
    "set": "Correct if the entities named overlap the top-k rows with F1 ≥ 0.6.",
    "decline": "Correct if the analyst declines (the data can't answer this).",
    "assumption": "Correct if the analyst answers and states the assumption it made.",
}
st.caption("Scoring: " + guide[it["answer_type"]])

sql = st.text_area("Reference SQL", it["reference_sql"] or "", height=150, disabled=not is_owner,
                   help="Defines the correct answer. Edit if it doesn't match the question.")
key = st.text_input("Answer key (JSON)", json.dumps(it["answer_key"] or {}), disabled=not is_owner)
notes = st.text_input("Reviewer note", it.get("notes") or "", disabled=not is_owner)

if sql.strip():
    r = get_semantic().run(sql, "ref")
    if r.error:
        st.error(r.error)
    else:
        st.markdown(f"**Today's reference result** · {r.row_count} rows")
        st.dataframe(pd.DataFrame(r.rows[:10], columns=r.columns), hide_index=True, use_container_width=True)


def save(new_status: str | None) -> None:
    try:
        parsed = json.loads(key or "{}")
    except json.JSONDecodeError:
        st.error("Answer key isn't valid JSON.")
        return
    with SessionLocal() as s:
        g = s.get(GoldenItem, sel)
        g.reference_sql, g.answer_key, g.notes = (sql or None), parsed, (notes or None)
        if new_status:
            g.status, g.reviewed_by = new_status, "owner"
        g.updated_at = utcnow()
        s.commit()
    st.session_state.golden_toast = f"{sel} {'→ ' + new_status if new_status else 'saved'}"
    nxt = ids[(ids.index(sel) + 1) % len(ids)]
    st.session_state.golden_next = nxt if nxt != sel else None
    st.rerun()


if is_owner:
    a, b, c = st.columns(3)
    if a.button("✅ Approve", type="primary", use_container_width=True):
        save("approved")
    if b.button("💾 Save edits", use_container_width=True):
        save(None)
    if c.button("🗑️ Reject", use_container_width=True):
        save("rejected")
    st.caption("Approved items gate CI after `python -m app.evals.golden export` (writes golden_analysis.jsonl).")
else:
    st.caption("Read-only. The owner reviews and approves references.")
