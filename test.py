import os
import re
import io
import csv
import ssl
import time
import json
import sqlite3
import tempfile
from datetime import datetime, date, timedelta, timezone
from urllib.parse import urlparse

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib

# =========================================================
# ENV (local .env + Streamlit secrets)
# =========================================================
load_dotenv()

def env_get(key: str, default: str = "") -> str:
    """Read Streamlit Secrets first, then OS env. Always returns string."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return str(os.getenv(key, default)).strip()

def ensure_dir(p: str) -> None:
    if not p:
        return
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def ensure_google_application_credentials() -> str:
    """
    For Streamlit Cloud:
    - If GOOGLE_APPLICATION_CREDENTIALS path exists, use it.
    - Else read GOOGLE_SERVICE_ACCOUNT_JSON from secrets/env,
      write to temp file, set GOOGLE_APPLICATION_CREDENTIALS.
    """
    existing = env_get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if existing and os.path.exists(existing):
        return existing

    sa_json = env_get("GOOGLE_SERVICE_ACCOUNT_JSON", "") or env_get("GOOGLE_CREDENTIALS_JSON", "")
    if not sa_json:
        return ""

    try:
        info = json.loads(sa_json)
        sa_json_clean = json.dumps(info)
    except Exception:
        sa_json_clean = sa_json

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(sa_json_clean.encode("utf-8"))
    tmp.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    return tmp.name

SERPAPI_KEY = env_get("SERPAPI_KEY")
OPENAI_API_KEY = env_get("OPENAI_API_KEY")

# SMTP
SMTP_HOST = env_get("SMTP_HOST")
SMTP_PORT = int(env_get("SMTP_PORT", "587") or "587")
SMTP_USER = env_get("SMTP_USER")
SMTP_PASS = env_get("SMTP_PASS")
MAIL_FROM = env_get("MAIL_FROM")
REPORT_TO = [x.strip() for x in env_get("REPORT_TO").split(",") if x.strip()]
REPORT_CC = [x.strip() for x in env_get("REPORT_CC").split(",") if x.strip()]

# Google libs (optional for weekly mail)
GOOGLE_OK = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as gbuild
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Metric, Dimension, RunReportRequest
except Exception:
    GOOGLE_OK = False

GOOGLE_CRED_FILE = ensure_google_application_credentials()
GSC_SITE_URL = env_get("GSC_SITE_URL")
GA4_PROPERTY_ID = env_get("GA4_PROPERTY_ID")

# =========================================================
# DB (SQLite)
# =========================================================
# TIP: on Streamlit Cloud this may not persist across restarts.
DB = env_get("SQLITE_DB", "data/rank_history.db") or "data/rank_history.db"
ensure_dir(DB)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS daily_ranks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  checked_date TEXT NOT NULL,
  project_domain TEXT NOT NULL,
  keyword TEXT NOT NULL,
  location TEXT DEFAULT '',
  gl TEXT DEFAULT '',
  hl TEXT DEFAULT '',
  device TEXT DEFAULT 'desktop',
  position REAL DEFAULT 0,
  url TEXT DEFAULT '',
  source TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  UNIQUE(checked_date, project_domain, keyword, location, gl, hl, device)
);

CREATE INDEX IF NOT EXISTS idx_daily_ranks_date ON daily_ranks(checked_date);
CREATE INDEX IF NOT EXISTS idx_daily_ranks_domain ON daily_ranks(project_domain);
"""

def db_connect():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA_SQL)
    return con

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def upsert_daily_rank(con, checked_date, project_domain, keyword, location, gl, hl, device, position, url, source):
    con.execute(
        """
        INSERT INTO daily_ranks (checked_date, project_domain, keyword, location, gl, hl, device, position, url, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(checked_date, project_domain, keyword, location, gl, hl, device)
        DO UPDATE SET position=excluded.position, url=excluded.url, source=excluded.source, created_at=excluded.created_at
        """,
        (
            checked_date,
            project_domain,
            keyword,
            location or "",
            gl or "",
            hl or "",
            device or "desktop",
            float(position or 0),
            url or "",
            source or "",
            utc_now_iso(),
        ),
    )

def fetch_by_date(con, project_domain: str, d: date):
    rows = con.execute(
        """
        SELECT checked_date, keyword, location, gl, hl, device, position, url, source
        FROM daily_ranks
        WHERE checked_date=? AND project_domain=?
        ORDER BY keyword ASC
        """,
        (d.isoformat(), project_domain),
    ).fetchall()
    return [dict(r) for r in rows]

def fetch_week(con, project_domain, ws: date, we: date):
    rows = con.execute(
        """
        SELECT checked_date, keyword, location, gl, hl, device, position, url, source
        FROM daily_ranks
        WHERE checked_date>=? AND checked_date<=? AND project_domain=?
        ORDER BY checked_date ASC, keyword ASC
        """,
        (ws.isoformat(), we.isoformat(), project_domain),
    ).fetchall()
    return [dict(r) for r in rows]

def list_available_dates(con, project_domain: str):
    rows = con.execute(
        """
        SELECT checked_date, COUNT(*) as cnt
        FROM daily_ranks
        WHERE project_domain=?
        GROUP BY checked_date
        ORDER BY checked_date ASC
        """,
        (project_domain,),
    ).fetchall()
    return [(r["checked_date"], int(r["cnt"])) for r in rows]

def get_min_max_date(con, project_domain: str):
    row = con.execute(
        """
        SELECT MIN(checked_date) mn, MAX(checked_date) mx
        FROM daily_ranks
        WHERE project_domain=?
        """,
        (project_domain,),
    ).fetchone()
    if not row or not row["mn"] or not row["mx"]:
        return None, None
    return date.fromisoformat(row["mn"]), date.fromisoformat(row["mx"])

def closest_available_date(available_dates: list[str], picked: date) -> str | None:
    """Return exact match if exists, else nearest available date string."""
    if not available_dates:
        return None
    target = picked.isoformat()
    if target in available_dates:
        return target
    # nearest by absolute distance
    ad = [date.fromisoformat(x) for x in available_dates]
    best = min(ad, key=lambda d: abs((d - picked).days))
    return best.isoformat()

# =========================================================
# Helpers
# =========================================================
def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("https://", "").replace("http://", "").strip("/")
    d = d.replace("www.", "")
    return d

def domain_match(target_domain: str, url: str) -> bool:
    td = normalize_domain(target_domain)
    raw = (url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    netloc = (parsed.netloc or "").lower().replace("www.", "")
    if not netloc:
        netloc = normalize_domain(raw).split("/")[0]
    return netloc == td or netloc.endswith("." + td)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def fmt_pos(p: float) -> str:
    return f"{p:.0f}" if p and p > 0 else "NR"

def today_local() -> date:
    return datetime.now().date()

def last_full_week_range() -> tuple[date, date]:
    t = today_local()
    days_since_sun = (t.weekday() - 6) % 7
    last_sun = t - timedelta(days=days_since_sun or 7)
    last_mon = last_sun - timedelta(days=6)
    return last_mon, last_sun

def daterange(ws: date, we: date):
    d = ws
    out = []
    while d <= we:
        out.append(d)
        d += timedelta(days=1)
    return out

def google_domain_from_gl(gl: str) -> str:
    gl = (gl or "").lower().strip()
    special = {
        "us": "google.com",
        "ae": "google.ae",
        "in": "google.co.in",
        "uk": "google.co.uk",
        "gb": "google.co.uk",
        "ca": "google.ca",
        "au": "google.com.au",
        "sa": "google.com.sa",
        "qa": "google.com.qa",
        "kw": "google.com.kw",
        "om": "google.com.om",
        "bh": "google.com.bh",
    }
    return special.get(gl, "google.com")

# =========================================================
# SerpAPI
# =========================================================
def resolve_location_serpapi(api_key: str, query: str) -> dict:
    query = (query or "").strip()
    if not query:
        return {"ok": False, "location": "", "gl": ""}

    try:
        r = requests.get(
            "https://serpapi.com/locations.json",
            params={"q": query, "limit": 1, "api_key": api_key},
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return {"ok": False, "location": query, "gl": ""}
        top = data[0]
        canonical = top.get("canonical_name") or top.get("name") or query
        gl = (top.get("country_code") or "").lower().strip()
        return {"ok": True, "location": canonical, "gl": gl}
    except Exception:
        return {"ok": False, "location": query, "gl": ""}

def serpapi_get_rank(
    serpapi_key: str,
    keyword: str,
    target_domain: str,
    gl: str,
    hl: str,
    device: str,
    location: str,
    top_n: int = 100
) -> dict:
    google_domain = google_domain_from_gl(gl)
    params = {
        "api_key": serpapi_key,
        "engine": "google",
        "q": keyword,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        "num": 10,
        "device": device,
    }
    if location:
        params["location"] = location

    for start in range(0, min(top_n, 100), 10):
        params["start"] = start
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        organic = data.get("organic_results", []) or []
        for item in organic:
            pos = item.get("position")
            link = item.get("link") or item.get("url") or ""
            if pos and link and domain_match(target_domain, link):
                return {"position": float(pos), "url": link, "source": "organic"}

        time.sleep(0.35)

    return {"position": 0.0, "url": "", "source": "not_found"}

# =========================================================
# Google (GSC + GA4) - used only for Weekly Email
# =========================================================
def _google_creds(sa_file: str, scopes: list[str]):
    if not GOOGLE_OK:
        raise RuntimeError("Google libs missing. Install google-api-python-client google-auth google-analytics-data")
    return service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)

def gsc_totals(sa_file: str, site_url: str, start_date: str, end_date: str) -> dict:
    creds = _google_creds(sa_file, ["https://www.googleapis.com/auth/webmasters.readonly"])
    svc = gbuild("searchconsole", "v1", credentials=creds, cache_discovery=False)
    body = {"startDate": start_date, "endDate": end_date, "dimensions": [], "rowLimit": 1}
    resp = svc.searchanalytics().query(siteUrl=site_url, body=body).execute()
    rows = resp.get("rows", []) or []
    if not rows:
        return {"clicks": 0, "impressions": 0, "ctr": 0.0, "position": 0.0}
    r = rows[0]
    return {
        "clicks": int(r.get("clicks", 0)),
        "impressions": int(r.get("impressions", 0)),
        "ctr": float(r.get("ctr", 0.0)),
        "position": float(r.get("position", 0.0)),
    }

def gsc_top(sa_file: str, site_url: str, start_date: str, end_date: str, dim: str, limit: int = 10):
    creds = _google_creds(sa_file, ["https://www.googleapis.com/auth/webmasters.readonly"])
    svc = gbuild("searchconsole", "v1", credentials=creds, cache_discovery=False)
    body = {"startDate": start_date, "endDate": end_date, "dimensions": [dim], "rowLimit": int(limit), "startRow": 0}
    resp = svc.searchanalytics().query(siteUrl=site_url, body=body).execute()
    out = []
    for row in (resp.get("rows", []) or []):
        keys = row.get("keys", [])
        name = keys[0] if keys else ""
        out.append({
            dim: name,
            "clicks": int(row.get("clicks", 0)),
            "impressions": int(row.get("impressions", 0)),
            "ctr": float(row.get("ctr", 0.0)),
            "position": float(row.get("position", 0.0)),
        })
    return out

def ga4_totals(sa_file: str, property_id: str, start_date: str, end_date: str) -> dict:
    creds = _google_creds(sa_file, ["https://www.googleapis.com/auth/analytics.readonly"])
    client = BetaAnalyticsDataClient(credentials=creds)
    req = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        metrics=[
            Metric(name="totalUsers"),
            Metric(name="sessions"),
            Metric(name="engagedSessions"),
            Metric(name="engagementRate"),
            Metric(name="conversions"),
        ],
    )
    resp = client.run_report(req)
    if not resp.rows:
        return {"users": 0, "sessions": 0, "engaged_sessions": 0, "engagement_rate": 0.0, "conversions": 0.0}
    vals = resp.rows[0].metric_values
    return {
        "users": int(vals[0].value or 0),
        "sessions": int(vals[1].value or 0),
        "engaged_sessions": int(vals[2].value or 0),
        "engagement_rate": float(vals[3].value or 0.0),
        "conversions": float(vals[4].value or 0.0),
    }

def ga4_organic_sessions(sa_file: str, property_id: str, start_date: str, end_date: str) -> int:
    creds = _google_creds(sa_file, ["https://www.googleapis.com/auth/analytics.readonly"])
    client = BetaAnalyticsDataClient(credentials=creds)
    req = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name="sessions")],
    )
    resp = client.run_report(req)
    organic = 0
    for row in resp.rows or []:
        ch = row.dimension_values[0].value if row.dimension_values else ""
        s = int(row.metric_values[0].value or 0)
        if (ch or "").lower() == "organic search":
            organic = s
            break
    return organic

# =========================================================
# Email + LLM (kept)
# =========================================================
def send_email(subject: str, html_body: str, csv_bytes: bytes, csv_filename: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and REPORT_TO):
        raise RuntimeError("SMTP settings missing (SMTP_HOST/SMTP_USER/SMTP_PASS/MAIL_FROM/REPORT_TO)")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = ", ".join(REPORT_TO)
    if REPORT_CC:
        msg["Cc"] = ", ".join(REPORT_CC)

    msg.set_content("HTML email not supported in this client.")
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename=csv_filename)

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(context=context)
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def clean_llm_html(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^```(?:html)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def generate_llm_insights_strict(payload_text: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    try:
        from openai import OpenAI
    except Exception:
        return ""

    prompt = f"""
You are an SEO analyst. You MUST ONLY use the numbers and facts from the data provided.
DO NOT invent metrics, DO NOT assume improvements if not shown.
Write a weekly SEO summary in HTML, suitable for a business owner.

Output structure:
<h3>AI Insights</h3>
<p>Key highlights...</p>
<ul><li>...</li></ul>
<h3>Next Week Action Plan</h3>
<ul><li>...</li></ul>

DATA:
{payload_text}

Return ONLY HTML.
""".strip()

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        return clean_llm_html(resp.choices[0].message.content)
    except Exception:
        return ""

# =========================================================
# Weekly matrix helpers (same logic)
# =========================================================
def build_matrix(rows: list[dict], days: list[date]) -> dict:
    m = {}
    for r in rows:
        k = (r["keyword"], r.get("location", ""), r.get("device", "desktop"))
        m.setdefault(k, {})
        m[k][r["checked_date"]] = {"pos": safe_float(r.get("position")), "url": r.get("url", "") or "", "source": r.get("source", "") or ""}

    for k in list(m.keys()):
        for d in days:
            ds = d.isoformat()
            if ds not in m[k]:
                m[k][ds] = {"pos": 0.0, "url": "", "source": ""}
    return m

def best_pos_for_period(daymap: dict) -> float:
    vals = [v["pos"] for v in daymap.values() if v["pos"] and v["pos"] > 0]
    return min(vals) if vals else 0.0

def summarize_week(matrix: dict, expected_keywords: int) -> dict:
    best_positions = []
    for _, dm in matrix.items():
        bp = best_pos_for_period(dm)
        if bp > 0:
            best_positions.append(bp)
    return {
        "keywords_tracked": expected_keywords,
        "ranked_keywords": len(best_positions),
        "top3": sum(1 for p in best_positions if p <= 3),
        "top10": sum(1 for p in best_positions if p <= 10),
        "top20": sum(1 for p in best_positions if p <= 20),
        "avg_best_position": (sum(best_positions) / len(best_positions)) if best_positions else 0.0,
    }

def movers_vs_prev(curr_matrix: dict, prev_matrix: dict, limit=20):
    movers = []
    keys = set(curr_matrix.keys()) | set(prev_matrix.keys())
    for k in keys:
        cb = best_pos_for_period(curr_matrix.get(k, {})) if k in curr_matrix else 0.0
        pb = best_pos_for_period(prev_matrix.get(k, {})) if k in prev_matrix else 0.0
        if cb > 0 and pb > 0:
            delta = cb - pb
        elif cb > 0 and pb == 0:
            delta = -9999
        elif cb == 0 and pb > 0:
            delta = 9999
        else:
            delta = 0.0
        movers.append({"keyword": k[0], "location": k[1], "device": k[2], "prev_best": pb, "curr_best": cb, "delta": delta})
    ups = sorted([m for m in movers if m["delta"] < 0], key=lambda x: x["delta"])[:limit]
    downs = sorted([m for m in movers if m["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:limit]
    return ups, downs

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="SEO Rank Tracker", page_icon="📈", layout="wide")
st.title("📈 SEO Rank Tracker (Daily Store → Compare Any Stored Dates → Weekly Mail)")
st.caption("Key fix: Compare tab shows a CALENDAR + DB dates. You can pick any date, but only stored dates will compare.")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY missing (set in Streamlit Secrets or .env)")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Project Settings")
project_domain_input = st.sidebar.text_input("Brand URL (domain)", value="plumbersindubai.com")
project_domain = normalize_domain(project_domain_input)

project_name = st.sidebar.text_input("Project Name", value="SEO Project")
hl = st.sidebar.selectbox("Language (hl)", ["en", "hi", "ar", "fr", "es", "de"], index=0)
device = st.sidebar.selectbox("Device", ["desktop", "mobile"], index=1)

location_text = st.sidebar.text_input("Target Location", value="Dubai, United Arab Emirates")
auto_resolve = st.sidebar.checkbox("Auto-resolve location (recommended)", value=True)
gl_manual = st.sidebar.text_input("Country code (gl) [auto]", value="").strip().lower()

resolved_location = location_text.strip()
resolved_gl = gl_manual

if auto_resolve and location_text.strip():
    info = resolve_location_serpapi(SERPAPI_KEY, location_text)
    if info["ok"]:
        resolved_location = info["location"]
        if not resolved_gl:
            resolved_gl = info.get("gl", "") or ""
if not resolved_gl:
    resolved_gl = "us"

st.info(f"Using: location **{resolved_location or '(none)'}**, gl **{resolved_gl}**, hl **{hl}**, device **{device}**")

st.subheader("📝 Keywords")
keywords_text = st.text_area("Enter keywords (one per line)", height=160)
keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

tabs = st.tabs(["✅ Daily Fetch", "📊 Compare Dates", "📅 History", "📩 Weekly Email"])

# =========================
# Tab 1: Daily Fetch
# =========================
with tabs[0]:
    colA, colB = st.columns([1, 1])
    store_for_date = colA.date_input("Store for date", value=today_local())
    run_daily_btn = colB.button("✅ Run Daily Fetch & Store", type="primary")

    if run_daily_btn:
        if not project_domain or not keywords:
            st.error("Please enter Brand URL (domain) and keywords.")
        else:
            con = db_connect()
            try:
                progress = st.progress(0)
                rows_out = []
                for i, kw in enumerate(keywords):
                    st.write(f"🔍 Checking: **{kw}** ...")
                    try:
                        res = serpapi_get_rank(
                            serpapi_key=SERPAPI_KEY,
                            keyword=kw,
                            target_domain=project_domain,
                            gl=resolved_gl,
                            hl=hl,
                            device=device,
                            location=resolved_location,
                            top_n=100
                        )
                    except Exception as e:
                        res = {"position": 0.0, "url": "", "source": f"error:{type(e).__name__}"}

                    upsert_daily_rank(
                        con,
                        checked_date=store_for_date.isoformat(),
                        project_domain=project_domain,
                        keyword=kw,
                        location=resolved_location,
                        gl=resolved_gl,
                        hl=hl,
                        device=device,
                        position=res["position"],
                        url=res["url"],
                        source=res.get("source", ""),
                    )
                    con.commit()

                    rows_out.append({
                        "checked_date": store_for_date.isoformat(),
                        "keyword": kw,
                        "position": fmt_pos(res["position"]),
                        "url": res["url"] or "",
                        "source": res.get("source", ""),
                        "gl": resolved_gl,
                        "hl": hl,
                        "device": device,
                        "location": resolved_location,
                    })
                    progress.progress((i + 1) / len(keywords))
                    time.sleep(0.4)

                st.success("✅ Stored daily ranks successfully!")
                st.dataframe(pd.DataFrame(rows_out), width="stretch")
            finally:
                con.close()

# =========================
# Tab 2: Compare Dates (FIXED properly)
# =========================
with tabs[1]:
    con = db_connect()
    try:
        saved = list_available_dates(con, project_domain)
        if not saved:
            st.warning("No saved data for this domain yet. Run Daily Fetch first.")
        else:
            available = [d for d, _ in saved]
            label_map = {d: f"{d} — {cnt} rows" for d, cnt in saved}
            min_d, max_d = get_min_max_date(con, project_domain)

            st.write("### Choose any dates (Calendar) + DB dropdown")
            c1, c2, c3 = st.columns([1, 1, 1])

            # Calendar lets you pick ANY date, but we auto-snap to nearest available date in DB.
            picked1 = c1.date_input("Pick Date 1 (calendar)", value=max_d, min_value=min_d, max_value=max_d)
            picked2 = c2.date_input("Pick Date 2 (calendar)", value=(max_d - timedelta(days=1) if max_d and max_d > min_d else max_d),
                                    min_value=min_d, max_value=max_d)

            snap1 = closest_available_date(available, picked1)
            snap2 = closest_available_date(available, picked2)

            # DB dropdown (explicit)
            d1_str = c1.selectbox("OR Date 1 (from DB)", options=available, index=available.index(snap1),
                                  format_func=lambda x: label_map.get(x, x))
            d2_str = c2.selectbox("OR Date 2 (from DB)", options=available, index=available.index(snap2),
                                  format_func=lambda x: label_map.get(x, x))

            show_only_changed = c3.checkbox("Show only changed keywords", value=False)

            d1 = date.fromisoformat(d1_str)
            d2 = date.fromisoformat(d2_str)

            rows1 = fetch_by_date(con, project_domain, d1)
            rows2 = fetch_by_date(con, project_domain, d2)

            def key(r):
                return (r["keyword"], r.get("location",""), r.get("device","desktop"))

            m1 = {key(r): r for r in rows1}
            m2 = {key(r): r for r in rows2}
            all_keys = sorted(set(m1.keys()) | set(m2.keys()), key=lambda x: (x[0].lower(), x[1].lower(), x[2].lower()))

            out = []
            for k in all_keys:
                r1 = m1.get(k, {})
                r2 = m2.get(k, {})

                p1 = safe_float(r1.get("position", 0))
                p2 = safe_float(r2.get("position", 0))

                # Rank: smaller is better
                if p1 > 0 and p2 > 0:
                    change = int(p2 - p1)   # negative = improved
                    if change < 0:
                        trend, arrow = "UP", "▲"
                    elif change > 0:
                        trend, arrow = "DOWN", "▼"
                    else:
                        trend, arrow = "SAME", "●"
                    delta = f"{change:+d}"
                elif p1 == 0 and p2 > 0:
                    trend, arrow, delta = "NEW", "▲", ""
                elif p1 > 0 and p2 == 0:
                    trend, arrow, delta = "LOST", "▼", ""
                else:
                    trend, arrow, delta = "SAME", "●", ""

                if show_only_changed and trend == "SAME":
                    continue

                out.append({
                    "keyword": k[0],
                    "location": k[1],
                    "device": k[2],
                    f"pos_{d1_str}": fmt_pos(p1),
                    f"pos_{d2_str}": fmt_pos(p2),
                    "trend": trend,
                    "arrow": arrow,
                    "delta": delta,  # ALWAYS string -> no Arrow conversion errors
                })

            st.caption("UP = improved (rank number decreased). DOWN = dropped (rank number increased). NEW/LOST handled.")
            st.dataframe(pd.DataFrame(out), width="stretch")

    finally:
        con.close()

# =========================
# Tab 3: History
# =========================
with tabs[2]:
    con = db_connect()
    try:
        st.write(f"**DB:** `{DB}`")
        saved = list_available_dates(con, project_domain)
        if not saved:
            st.info("No history yet.")
        else:
            dfh = pd.DataFrame(saved, columns=["checked_date", "rows"])
            st.dataframe(dfh, width="stretch")

            # Optional: allow download DB
            with open(DB, "rb") as f:
                st.download_button("⬇️ Download SQLite DB (backup)", f.read(), file_name="rank_history.db")
    finally:
        con.close()

# =========================
# Tab 4: Weekly Email (kept; will fail if GSC permissions missing)
# =========================
with tabs[3]:
    st.write("### 📩 Weekly SEO Email")
    st.info("If you get GSC 403: Add the Service Account email as a user/owner in Search Console property.")

    send_weekly_btn = st.button("📨 Send Weekly Email Now", type="primary")

    if send_weekly_btn:
        missing = []
        if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and REPORT_TO):
            missing.append("SMTP settings")
        if not GOOGLE_OK:
            missing.append("Google libraries")
        if not (GOOGLE_CRED_FILE and os.path.exists(GOOGLE_CRED_FILE)):
            missing.append("Google credentials (GOOGLE_SERVICE_ACCOUNT_JSON)")
        if not GSC_SITE_URL:
            missing.append("GSC_SITE_URL")
        if not GA4_PROPERTY_ID:
            missing.append("GA4_PROPERTY_ID")

        if missing:
            st.error("Weekly report requires:\n- " + "\n- ".join(missing))
            st.stop()

        # NOTE: weekly builder omitted here to keep this file focused on your “compare” issue.
        # You can paste your existing weekly HTML builder functions as-is.
        st.error("Weekly email section: keep your existing HTML/CSV builders here (your compare + storage is now fixed).")
