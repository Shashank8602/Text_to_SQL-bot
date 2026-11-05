"""Upload wizard and dataset management (docs/DESIGN.md section 15.5).

Upload -> review schema (types, relationships, hidden columns) -> load -> descriptions -> ready.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # make `app` importable under `streamlit run`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.datasets import descriptions as desc_store  # noqa: E402
from app.datasets import registry  # noqa: E402
from app.datasets.loader import DatasetLoadError, delete_dataset, load_pg_dump, slugify  # noqa: E402
from app.datasets.relationships import detect  # noqa: E402
from app.datasets.tabular import PG_TYPES, TabularError  # noqa: E402
from app.datasets.tabular_loader import load_tabular, parse_files  # noqa: E402

DUMP_SUFFIXES = (".sql", ".sql.gz")


def _is_dump(name: str) -> bool:
    return name.lower().endswith(DUMP_SUFFIXES)


def _save_uploads(files) -> list[Path]:
    """Write uploads to a fresh temp folder using only the base name (never a client-supplied path)."""
    folder = Path(tempfile.mkdtemp(prefix="upload_"))
    paths = []
    for f in files:
        path = folder / Path(f.name).name
        path.write_bytes(f.getbuffer())
        paths.append(path)
    return paths


def _reset_wizard() -> None:
    for k in ("wizard", "uploaded_slug"):
        st.session_state.pop(k, None)


# --------------------------------------------------------------------------- steps

def step_upload() -> None:
    st.subheader("1 · Upload")
    st.caption("CSV (one file = one table), Excel .xlsx (one sheet = one table), or a plain-SQL PostgreSQL "
               f"dump (.sql / .sql.gz). Up to {get_settings().max_upload_mb} MB per file. "
               "Files uploaded together form one dataset.")
    files = st.file_uploader("Files", type=["csv", "tsv", "xlsx", "sql", "gz"], accept_multiple_files=True)
    name = st.text_input("Dataset name", placeholder="e.g. My Shop")
    if not st.button("Read files", type="primary", disabled=not (files and name.strip())):
        return
    if registry.get(slugify(name)):
        st.error(f"A dataset called '{slugify(name)}' already exists. Pick another name or delete it below.")
        return
    paths = _save_uploads(files)
    try:
        if any(_is_dump(p.name) for p in paths):
            if len(paths) != 1:
                st.error("Upload a SQL dump on its own, not together with other files.")
                return
            st.session_state.wizard = {"kind": "dump", "name": name.strip(), "path": paths[0]}
        else:
            tables = parse_files(paths)
            st.session_state.wizard = {"kind": "tabular", "name": name.strip(), "tables": tables,
                                       "candidates": detect(tables), "paths": paths}
    except TabularError as e:
        st.error(str(e))
        return
    st.rerun()


def step_review_tabular(w: dict) -> None:
    st.subheader("2 · Review")
    overrides: dict[str, str] = {}
    for t in w["tables"]:
        with st.expander(f"{t.name} · {len(t.rows)} rows · from {t.source}", expanded=len(w["tables"]) <= 3):
            st.dataframe(pd.DataFrame(t.rows[:5], columns=[c.name for c in t.columns]), width="stretch")
            edited = st.data_editor(
                pd.DataFrame({"column": [c.name for c in t.columns], "type": [c.type for c in t.columns]}),
                key=f"types-{t.name}", hide_index=True, disabled=["column"], width="stretch",
                column_config={"type": st.column_config.SelectboxColumn(options=list(PG_TYPES), required=True)},
            )
            for c, new_type in zip(t.columns, edited["type"]):
                if new_type != c.type:
                    overrides[f"{t.name}.{c.name}"] = new_type

    accepted = []
    if w["candidates"]:
        st.markdown("**Detected relationships** (confirmed ones become foreign keys and guide joins)")
        for i, rel in enumerate(w["candidates"]):
            if st.checkbox(rel.label(), value=True, key=f"rel-{i}"):
                accepted.append(rel)
    else:
        st.caption("No relationships detected between these tables.")

    all_cols = [f"{t.name}.{c.name}" for t in w["tables"] for c in t.columns]
    hide = st.multiselect("Hide extra columns from the bot", all_cols,
                          help="Columns that look sensitive (password, token, ...) are hidden automatically.")
    if st.button("Load dataset", type="primary"):
        with st.spinner("Loading..."):
            try:
                source = "excel" if any(p.suffix.lower() == ".xlsx" for p in w["paths"]) else "csv"
                report = load_tabular(w["tables"], w["name"], type_overrides=overrides,
                                      relationships=accepted, hide=hide, source_type=source)
            except (DatasetLoadError, TabularError, registry.DatasetExists, ValueError) as e:
                st.error(str(e))
                return
        st.session_state.uploaded_slug = report.slug
        st.session_state.pop("wizard")
        st.rerun()


def step_review_dump(w: dict) -> None:
    st.subheader("2 · Review")
    st.write(f"Ready to restore **{w['path'].name}** as dataset **{w['name']}**. The dump is parsed by the "
             "app (never run through `psql`), restored into its own database, and sensitive-looking columns "
             "are hidden.")
    hide = st.text_input("Extra columns to hide (comma-separated `table.column`)", "")
    if st.button("Load dataset", type="primary"):
        with st.spinner("Restoring dump..."):
            try:
                report = load_pg_dump(w["path"], w["name"],
                                      hide=[h.strip() for h in hide.split(",") if h.strip()])
            except (DatasetLoadError, registry.DatasetExists, ValueError) as e:
                st.error(str(e))
                return
        st.session_state.uploaded_slug = report.slug
        st.session_state.pop("wizard")
        st.rerun()


def step_descriptions(slug: str) -> None:
    ds = registry.get(slug)
    if ds is None:
        _reset_wizard()
        st.rerun()
    st.success(f"Dataset **{ds.name}** is loaded"
               + (f"; hidden from the bot: {', '.join(ds.hidden_columns)}" if ds.hidden_columns else "") + ".")
    st.subheader("3 · Descriptions")
    st.caption("Descriptions help the bot pick the right tables and columns. Draft them with the AI "
               "(uses a little LLM quota), edit, then save. You can skip this and come back later.")
    key = f"yaml-{slug}"
    if st.button("Draft descriptions with AI"):
        from app.datasets.describer import draft
        with st.spinner("Drafting..."):
            try:
                st.session_state[key] = desc_store.to_yaml(draft(slug))
            except Exception as e:
                st.error(f"Could not draft descriptions: {e}")
    if key in st.session_state:
        text = st.text_area("Descriptions (YAML)", st.session_state[key], height=320)
        if st.button("Save and build search index", type="primary"):
            _save_descriptions(slug, text)
    c1, c2 = st.columns(2)
    if c1.button("Finish: build search index without descriptions"):
        _index(slug)
    if c2.button("Upload another dataset"):
        _reset_wizard()
        st.rerun()


def _save_descriptions(slug: str, text: str) -> None:
    from app.database.catalog import load_catalog
    try:
        desc = desc_store.parse_yaml_text(text)
        problems = desc_store.validate(desc, load_catalog(registry.get(slug).db_name))
    except Exception as e:
        st.error(f"Invalid YAML: {e}")
        return
    if problems:
        st.error("These names do not exist in the dataset: " + "; ".join(problems))
        return
    desc_store.save(slug, desc)
    _index(slug)


def _index(slug: str) -> None:
    from app.retrieval.indexer import build_index
    embedder = get_settings().retrieval_embedder
    if embedder == "none":
        st.info("Retrieval is off (RETRIEVAL_EMBEDDER=none); nothing to index.")
        return
    with st.spinner("Building the search index (the first run downloads the embedding model)..."):
        n = build_index(slug, embedder)
    st.success(f"Indexed {n} schema documents. Open the chat page and ask away.")


def manage_datasets() -> None:
    st.divider()
    st.subheader("Your datasets")
    datasets = registry.list_all()
    if not datasets:
        st.caption("None yet.")
    for d in datasets:
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown(f"**{d.name}** · `{d.slug}`")
        c2.caption(f"{d.source_type} · {d.status}")
        if c3.button("Delete", key=f"del-{d.slug}"):
            st.session_state.confirm_delete = d.slug
    slug = st.session_state.get("confirm_delete")
    if slug:
        st.warning(f"Delete **{slug}** and all its data? This cannot be undone.")
        a, b = st.columns(2)
        if a.button("Yes, delete", type="primary"):
            delete_dataset(slug)
            st.session_state.pop("confirm_delete")
            st.rerun()
        if b.button("Cancel"):
            st.session_state.pop("confirm_delete")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Upload data", page_icon="📤", layout="wide")
    st.title("📤 Bring your own data")
    w = st.session_state.get("wizard")
    slug = st.session_state.get("uploaded_slug")
    if slug:
        step_descriptions(slug)
    elif w is None:
        step_upload()
    else:
        (step_review_tabular if w["kind"] == "tabular" else step_review_dump)(w)
        if st.button("Start over"):
            _reset_wizard()
            st.rerun()
    manage_datasets()


main()
