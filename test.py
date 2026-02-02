import os
import re
import io
import csv
import ssl
import time
import json
import sqlite3
import tempfile
import requests
from datetime import datetime, date, timedelta
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib

# ------------------- ENV -------------------
# Local dev: loads .env (Streamlit Cloud: secrets are used instead)
load_dotenv()

def _get_setting(key: str, default: str = "") -> str:
    """
    Read from st.secrets first (Streamlit Cloud), fallback to env (local).
    """
    try:
        if key in st.secrets:
            return str(st.secrets.get(key, default)).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()

SERPAPI_KEY = _get_setting("SERPAPI_KEY")
OPENAI_API_KEY = _get_setting("OPENAI_API_KEY")

# SMTP
SMTP_HOST = _get_setting("SMTP_HOST")
SMTP_PORT = int(_get_setting("SMTP_PORT", "587") or "587")
SMTP_USER = _get_setting("SMTP_USER")
SMTP_PASS = _get_setting("SMTP_PASS")
MAIL_FROM = _get_setting("MAIL_FROM")
REPORT_TO = [x.strip() for x in _get_setting("REPORT_TO").split(",") if x.strip()]
REPORT_CC = [x.strip() for x in _get_setting("REPORT_CC").split(",") if x.strip()]

# Google (config)
GSC_SITE_URL = _get_setting("GSC_SITE_URL")
GA4_PROPERTY_ID = _get_setting("GA4_PROPERTY_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = _get_setting("GOOGLE_SERVICE_ACCOUNT_JSON")

# DB
DB = _get_setting("SQLITE_DB", "rank_history.db") or "rank_history.db"

# ------------------- GOOGLE IMPORTS -------------------
GOOGLE_OK = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as gbuild
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Metric, Dimension, RunReportRequest
except Exception:
    GOOGLE_OK = False

# ------------------- HELPERS -------------------
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

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def fmt_pos(p: float) -> str:
    return f"{p:.0f}" if p and p > 0 else "NR"

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

def today_local() -> date:
    return datetime.now().date()

def last_full_week_range() -> tuple[date, date]:
    """
    Last completed Mon-Sun week.
    """
    t = today_local()
    # Find last Sunday
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

# ------------------- STREAMLIT CLOUD GOOGLE CREDS -------------------
def ensure_google_application_credentials() -> str | None:
    """
    Streamlit Cloud doesn't have credential files.
    We store service account JSON in secrets/env and write to temp file.
    """
    # If a valid file path already exists (local machine case)
    existing = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if existing and os.path.exists(existing):
        return existing

    raw = (GOOGLE_SERVICE_ACCOUNT_JSON or "").strip()
    if not raw:
        return None

    # Validate it's JSON
    try:
        json.loads(raw)
    except Exception:
        # Sometimes users paste with trailing spaces; still invalid => return None
        return None

    tmp_path = os.path.join(tempfile.gettempdir(), "service-account.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(raw)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    return tmp_path

GOOGLE_CRED_FILE = ensure_google_application_credentials()

# ------------------- DB SCHEMA + MIGRATION -------------------
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

def _table_columns(con, table: str) -> set[str]:
    cols = set()
    for r in con.execute(f"PRAGMA table_info({table})").fetchall():
        cols.add(r[1])
    return cols

def migrate_db_if_needed(con: sqlite3.Connection):
    """
    Fix old DB that doesn't have 'project_domain' (or other missing columns).
    We'll rebuild table safely if required.
    """
    con.execute("CREATE TABLE IF NOT EXISTS __migrations (name TEXT PRIMARY KEY, applied_at TEXT)")
    # If no table yet, schema script handles it
    con.executescript(SCHEMA_SQL)

    cols = _table_columns(con, "daily_ranks")
    required = {
        "checked_date", "project_domain", "keyword", "location", "gl", "hl",
        "device", "position", "url", "source", "created_at"
    }

    if required.issubset(cols):
        return  # OK

    # Rebuild table
    con.execute("ALTER TABLE daily_ranks RENAME TO daily_ranks_old")

    con.executescript(SCHEMA_SQL)

    old_cols = _table_columns(con, "daily_ranks_old")
    # Map best-effort columns from old to new
    # If old DB didn't have project_domain, we set it to '' (it will still store new rows correctly).
    select_cols = []
    insert_cols = [
        "checked_date", "project_domain", "keyword", "location", "gl", "hl",
        "device", "position", "url", "source", "created_at"
    ]

    def pick(name, fallback_literal=None):
        if name in old_cols:
            return name
        if fallback_literal is not None:
            return fallback_literal
        return "''"

    select_cols = [
        pick("checked_date"),
        pick("project_domain", "''"),
        pick("keyword"),
        pick("location", "''"),
        pick("gl", "''"),
        pick("hl", "''"),
        pick("device", "'desktop'"),
        pick("position", "0"),
        pick("url", "''"),
        pick("source", "''"),
        pick("created_at", "''"),
    ]

    con.execute(
        f"""
        INSERT OR IGNORE INTO daily_ranks ({",".join(insert_cols)})
        SELECT {",".join(select_cols)} FROM daily_ranks_old
        """
    )
    con.execute("DROP TABLE daily_ranks_old")
    con.execute(
        "INSERT OR REPLACE INTO __migrations (name, applied_at) VALUES (?, ?)",
        ("rebuild_daily_ranks_v1", datetime.utcnow().isoformat()),
    )
    con.commit()

def db_connect():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    migrate_db_if_needed(con)
    return con

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
            datetime.utcnow().isoformat(),
        ),
    )

def fetch_by_date(con, project_domain, d: date):
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

def get_min_max_dates(con, project_domain):
    row = con.execute(
        "SELECT MIN(checked_date) as mn, MAX(checked_date) as mx FROM daily_ranks WHERE project_domain=?",
        (project_domain,),
    ).fetchone()
    if not row or not row["mn"]:
        return None, None
    return date.fromisoformat(row["mn"]), date.fromisoformat(row["mx"])

# ------------------- LOCATION RESOLVE (SerpAPI) -------------------
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

# ------------------- SERPAPI RANK -------------------
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

    # paginate 0..90 (top 100)
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

# ------------------- GOOGLE (GSC + GA4) -------------------
def _google_creds(sa_file: str, scopes: list[str]):
    if not GOOGLE_OK:
        raise RuntimeError("Google libs missing. Install: google-api-python-client google-auth google-analytics-data")
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

# ------------------- MATRIX / WEEK CALCS -------------------
def build_matrix(rows: list[dict], days: list[date]) -> dict:
    m = {}
    for r in rows:
        k = (r["keyword"], r.get("location", ""), r.get("device", "desktop"))
        if k not in m:
            m[k] = {}
        m[k][r["checked_date"]] = {
            "pos": safe_float(r.get("position")),
            "url": r.get("url", "") or "",
            "source": r.get("source", "") or ""
        }

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
            delta = -9999  # NEW
        elif cb == 0 and pb > 0:
            delta = 9999   # LOST
        else:
            delta = 0.0

        movers.append({
            "keyword": k[0], "location": k[1], "device": k[2],
            "prev_best": pb, "curr_best": cb, "delta": delta
        })

    ups = sorted([m for m in movers if m["delta"] < 0], key=lambda x: x["delta"])[:limit]
    downs = sorted([m for m in movers if m["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:limit]
    return ups, downs

# ------------------- EMAIL -------------------
def send_email(subject: str, html_body: str, csv_bytes: bytes, csv_filename: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and REPORT_TO):
        raise RuntimeError("SMTP settings missing (check Streamlit Secrets or .env).")

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

# ------------------- LLM INSIGHTS (STRICT) -------------------
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
DO NOT invent metrics. DO NOT assume improvements if not shown.
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

# ------------------- WEEKLY EMAIL BUILDERS -------------------
def build_weekly_email_html(
    project_name: str,
    domain: str,
    ws: date, we: date,
    prev_ws: date, prev_we: date,
    expected_keywords: int,
    curr_matrix: dict,
    prev_matrix: dict,
    curr_days: list[date],
    gsc_curr: dict,
    gsc_prev: dict,
    gsc_top_queries: list[dict],
    gsc_top_pages: list[dict],
    ga_curr: dict,
    ga_prev: dict,
    ga_curr_org: int,
    ga_prev_org: int,
    llm_html: str
) -> str:
    curr_sum = summarize_week(curr_matrix, expected_keywords)
    prev_sum = summarize_week(prev_matrix, expected_keywords)
    ups, downs = movers_vs_prev(curr_matrix, prev_matrix, limit=15)

    def kpi_row(label, prev_val, curr_val):
        return f"""
        <tr>
          <td style="border:1px solid #ddd;">{html_escape(label)}</td>
          <td style="border:1px solid #ddd;text-align:right;">{html_escape(str(prev_val))}</td>
          <td style="border:1px solid #ddd;text-align:right;"><b>{html_escape(str(curr_val))}</b></td>
        </tr>
        """

    def top_table(title: str, rows: list[dict], keyname: str) -> str:
        trs = ""
        for r in rows[:10]:
            name = r.get(keyname, "")
            trs += f"""
            <tr>
              <td style="border:1px solid #ddd;">{html_escape(name)}</td>
              <td style="border:1px solid #ddd;text-align:right;">{r.get("clicks",0)}</td>
              <td style="border:1px solid #ddd;text-align:right;">{r.get("impressions",0)}</td>
              <td style="border:1px solid #ddd;text-align:right;">{r.get("ctr",0.0)*100:.2f}%</td>
              <td style="border:1px solid #ddd;text-align:right;">{r.get("position",0.0):.2f}</td>
            </tr>
            """
        return f"""
        <h4 style="margin:12px 0 6px;">{html_escape(title)}</h4>
        <table cellpadding="7" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
          <tr style="background:#f4f4f4;">
            <th style="border:1px solid #ddd;text-align:left;">{html_escape(keyname.title())}</th>
            <th style="border:1px solid #ddd;text-align:right;">Clicks</th>
            <th style="border:1px solid #ddd;text-align:right;">Impr</th>
            <th style="border:1px solid #ddd;text-align:right;">CTR</th>
            <th style="border:1px solid #ddd;text-align:right;">Pos</th>
          </tr>
          {trs if trs else "<tr><td colspan='5' style='border:1px solid #ddd;color:#777;'>No data</td></tr>"}
        </table>
        """

    def movers_table(title: str, rows: list[dict]) -> str:
        trs = ""
        for m in rows[:10]:
            if m["delta"] == -9999:
                delta_txt = "NEW"
                trend = "▲"
            elif m["delta"] == 9999:
                delta_txt = "LOST"
                trend = "▼"
            else:
                delta_txt = f"{int(m['delta']):+d}"
                trend = "▲" if m["delta"] < 0 else ("▼" if m["delta"] > 0 else "●")
            trs += f"""
            <tr>
              <td style="border:1px solid #ddd;">{html_escape(m['keyword'])}</td>
              <td style="border:1px solid #ddd;">{html_escape(m['location'])}</td>
              <td style="border:1px solid #ddd;">{html_escape(m['device'])}</td>
              <td style="border:1px solid #ddd;text-align:center;">{fmt_pos(m['prev_best'])}</td>
              <td style="border:1px solid #ddd;text-align:center;">{fmt_pos(m['curr_best'])}</td>
              <td style="border:1px solid #ddd;text-align:center;"><b>{trend}</b></td>
              <td style="border:1px solid #ddd;text-align:center;">{html_escape(delta_txt)}</td>
            </tr>
            """
        return f"""
        <h3 style="margin:18px 0 8px;">{html_escape(title)}</h3>
        <table cellpadding="8" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
          <tr style="background:#f4f4f4;">
            <th style="border:1px solid #ddd;text-align:left;">Keyword</th>
            <th style="border:1px solid #ddd;text-align:left;">Location</th>
            <th style="border:1px solid #ddd;text-align:left;">Device</th>
            <th style="border:1px solid #ddd;">Prev Best</th>
            <th style="border:1px solid #ddd;">Curr Best</th>
            <th style="border:1px solid #ddd;">Trend</th>
            <th style="border:1px solid #ddd;">Δ</th>
          </tr>
          {trs if trs else "<tr><td colspan='7' style='border:1px solid #ddd;color:#777;'>No data</td></tr>"}
        </table>
        """

    day_headers = "".join([
        f"<th style='border:1px solid #ddd;background:#f4f4f4;'>{d.strftime('%a')}<br>{d.strftime('%d-%b')}</th>"
        for d in curr_days
    ])

    daywise_trs = ""
    if curr_matrix:
        def matrix_sort(item):
            (kw, loc, dev), dm = item
            cb = best_pos_for_period(dm)
            return (9999 if cb == 0 else cb, kw.lower())

        for (kw, loc, dev), dm in sorted(curr_matrix.items(), key=matrix_sort):
            cb = best_pos_for_period(dm)
            pb = best_pos_for_period(prev_matrix.get((kw, loc, dev), {})) if prev_matrix else 0.0
            if cb > 0 and pb > 0:
                dlt = cb - pb
                tr = "▲" if dlt < 0 else ("▼" if dlt > 0 else "●")
            elif cb > 0 and pb == 0:
                tr = "▲"
            elif cb == 0 and pb > 0:
                tr = "▼"
            else:
                tr = "●"

            cells = ""
            for d in curr_days:
                p = safe_float(dm.get(d.isoformat(), {}).get("pos", 0.0))
                cells += f"<td style='border:1px solid #ddd;text-align:center;'>{fmt_pos(p)}</td>"

            daywise_trs += f"""
            <tr>
              <td style="border:1px solid #ddd;">{html_escape(kw)}</td>
              <td style="border:1px solid #ddd;">{html_escape(loc)}</td>
              <td style="border:1px solid #ddd;">{html_escape(dev)}</td>
              <td style="border:1px solid #ddd;text-align:center;">{fmt_pos(pb)}</td>
              <td style="border:1px solid #ddd;text-align:center;">{fmt_pos(cb)}</td>
              <td style="border:1px solid #ddd;text-align:center;"><b>{tr}</b></td>
              {cells}
            </tr>
            """

    html = f"""
    <html><body style="font-family:Arial, sans-serif; color:#111;">
      <h2 style="margin:0 0 10px;">Weekly SEO Report — {html_escape(project_name)} ({html_escape(domain)})</h2>
      <div style="color:#555;margin-bottom:12px;">
        Current Week: <b>{ws.isoformat()}</b> to <b>{we.isoformat()}</b><br>
        Previous Week: <b>{prev_ws.isoformat()}</b> to <b>{prev_we.isoformat()}</b>
      </div>

      <h3 style="margin:14px 0 8px;">Rank Tracking (SerpAPI)</h3>
      <table cellpadding="8" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
        <tr>
          <td style="border:1px solid #ddd;"><b>Keywords Tracked</b><br>{curr_sum['keywords_tracked']}</td>
          <td style="border:1px solid #ddd;"><b>Ranked Keywords</b><br>{curr_sum['ranked_keywords']}</td>
          <td style="border:1px solid #ddd;"><b>Top 3</b><br>{curr_sum['top3']}</td>
          <td style="border:1px solid #ddd;"><b>Top 10</b><br>{curr_sum['top10']}</td>
          <td style="border:1px solid #ddd;"><b>Top 20</b><br>{curr_sum['top20']}</td>
          <td style="border:1px solid #ddd;"><b>Avg Best Pos</b><br>{curr_sum['avg_best_position']:.2f}</td>
        </tr>
      </table>

      <div style="margin-top:10px;color:#555;">
        <b>WoW (Rank Summary)</b> —
        Top10: {prev_sum['top10']} → {curr_sum['top10']},
        Top3: {prev_sum['top3']} → {curr_sum['top3']},
        Avg Best Pos: {prev_sum['avg_best_position']:.2f} → {curr_sum['avg_best_position']:.2f}
      </div>

      {"<div style='margin:16px 0;padding:14px;border-left:4px solid #0b6; background:#f6fffa;'>" + llm_html + "</div>" if llm_html else ""}

      <h3 style="margin:18px 0 8px;">Google Search Console (WoW)</h3>
      <table cellpadding="8" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
        <tr style="background:#f4f4f4;">
          <th style="border:1px solid #ddd;text-align:left;">Metric</th>
          <th style="border:1px solid #ddd;text-align:right;">Prev Week</th>
          <th style="border:1px solid #ddd;text-align:right;">Curr Week</th>
        </tr>
        {kpi_row("Clicks", gsc_prev.get("clicks",0), gsc_curr.get("clicks",0))}
        {kpi_row("Impressions", gsc_prev.get("impressions",0), gsc_curr.get("impressions",0))}
        {kpi_row("CTR", f"{gsc_prev.get('ctr',0.0)*100:.2f}%", f"{gsc_curr.get('ctr',0.0)*100:.2f}%")}
        {kpi_row("Avg Position", f"{gsc_prev.get('position',0.0):.2f}", f"{gsc_curr.get('position',0.0):.2f}")}

      </table>

      {top_table("Top Queries (Current Week)", gsc_top_queries, "query")}
      {top_table("Top Pages (Current Week)", gsc_top_pages, "page")}

      <h3 style="margin:18px 0 8px;">Google Analytics (GA4) (WoW)</h3>
      <table cellpadding="8" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
        <tr style="background:#f4f4f4;">
          <th style="border:1px solid #ddd;text-align:left;">Metric</th>
          <th style="border:1px solid #ddd;text-align:right;">Prev Week</th>
          <th style="border:1px solid #ddd;text-align:right;">Curr Week</th>
        </tr>
        {kpi_row("Users", ga_prev.get("users",0), ga_curr.get("users",0))}
        {kpi_row("Sessions", ga_prev.get("sessions",0), ga_curr.get("sessions",0))}
        {kpi_row("Organic Sessions", ga_prev_org, ga_curr_org)}
        {kpi_row("Engaged Sessions", ga_prev.get("engaged_sessions",0), ga_curr.get("engaged_sessions",0))}
        {kpi_row("Engagement Rate", f"{ga_prev.get('engagement_rate',0.0)*100:.2f}%", f"{ga_curr.get('engagement_rate',0.0)*100:.2f}%"))}
        {kpi_row("Conversions", ga_prev.get("conversions",0.0), ga_curr.get("conversions",0.0))}
      </table>

      {movers_table("Keywords Improved (Top movers)", ups)}
      {movers_table("Keywords Dropped (Top movers)", downs)}

      <h3 style="margin:18px 0 8px;">Day-wise Ranks (Current Week)</h3>
      <table cellpadding="7" cellspacing="0" border="0" style="border-collapse:collapse;width:100%;max-width:980px;">
        <tr style="background:#f4f4f4;">
          <th style="border:1px solid #ddd;text-align:left;">Keyword</th>
          <th style="border:1px solid #ddd;text-align:left;">Location</th>
          <th style="border:1px solid #ddd;text-align:left;">Device</th>
          <th style="border:1px solid #ddd;">Prev Best</th>
          <th style="border:1px solid #ddd;">Curr Best</th>
          <th style="border:1px solid #ddd;">Trend</th>
          {day_headers}
        </tr>
        {daywise_trs if daywise_trs else "<tr><td colspan='50' style='border:1px solid #ddd;color:#777;'>No day-wise rank data found for this week. Run Daily Fetch every day.</td></tr>"}
      </table>

      <div style="color:#777;margin-top:10px;font-size:12px;">
        Notes: NR = Not Ranked. For accurate day-wise ranks, run Daily Fetch once per day.
      </div>
    </body></html>
    """
    return html

def build_weekly_csv(ws: date, we: date, prev_ws: date, prev_we: date, curr_days: list[date], curr_matrix: dict, prev_matrix: dict):
    output = io.StringIO()
    writer = csv.writer(output)
    headers = [
        "prev_week_start", "prev_week_end",
        "curr_week_start", "curr_week_end",
        "keyword", "location", "device",
        "prev_best", "curr_best", "trend", "delta"
    ] + [d.isoformat() for d in curr_days]
    writer.writerow(headers)

    keys = sorted(set(curr_matrix.keys()) | set(prev_matrix.keys()), key=lambda x: (x[0].lower(), x[1].lower(), x[2].lower()))
    for k in keys:
        kw, loc, dev = k
        curr_dm = curr_matrix.get(k, {})
        prev_dm = prev_matrix.get(k, {})

        curr_best = best_pos_for_period(curr_dm) if curr_dm else 0.0
        prev_best = best_pos_for_period(prev_dm) if prev_dm else 0.0

        if curr_best > 0 and prev_best > 0:
            delta = int(curr_best - prev_best)
            trend = "UP" if delta < 0 else ("DOWN" if delta > 0 else "SAME")
        elif curr_best > 0 and prev_best == 0:
            delta = "NEW"
            trend = "UP"
        elif curr_best == 0 and prev_best > 0:
            delta = "LOST"
            trend = "DOWN"
        else:
            delta = 0
            trend = "SAME"

        row = [
            prev_ws.isoformat(), prev_we.isoformat(),
            ws.isoformat(), we.isoformat(),
            kw, loc, dev,
            (int(prev_best) if prev_best > 0 else "NR"),
            (int(curr_best) if curr_best > 0 else "NR"),
            trend,
            delta,
        ]
        for d in curr_days:
            p = safe_float(curr_dm.get(d.isoformat(), {}).get("pos", 0.0)) if curr_dm else 0.0
            row.append(int(p) if p > 0 else "NR")

        writer.writerow(row)

    return output.getvalue().encode("utf-8")

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="SEO Rank + Weekly Report", page_icon="📈", layout="wide")

st.title("📈 SEO Rank Tracker + Weekly SEO Report (GSC + GA4 + AI)")
st.caption("Daily store → Compare → Weekly Mail with complete SEO report")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY missing (set it in Streamlit Secrets or .env).")
    st.stop()

# Sidebar Settings
st.sidebar.header("⚙️ Project Settings")

project_domain_input = st.sidebar.text_input("Brand URL (domain)", value="plumbersindubai.com")
project_domain = normalize_domain(project_domain_input)

project_name = st.sidebar.text_input("Project Name", value="SEO Project")

hl = st.sidebar.selectbox("Language (hl)", ["en", "hi", "ar", "fr", "es", "de"], index=0)
device = st.sidebar.selectbox("Device", ["desktop", "mobile"], index=1)

location_text = st.sidebar.text_input("Target Location (any worldwide)", value="Dubai, United Arab Emirates")
auto_resolve = st.sidebar.checkbox("Auto-resolve location (recommended)", value=True)

gl_manual = st.sidebar.text_input("Country code (gl) [auto]", value="").strip().lower()

st.sidebar.markdown("---")
st.sidebar.header("📩 Weekly Report Requirements (Mandatory)")

missing_weekly = []
if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and REPORT_TO):
    missing_weekly.append("SMTP settings")
if not GOOGLE_OK:
    missing_weekly.append("Google libraries")
if not GOOGLE_CRED_FILE:
    missing_weekly.append("Google service account JSON (GOOGLE_SERVICE_ACCOUNT_JSON)")
if not GSC_SITE_URL:
    missing_weekly.append("GSC_SITE_URL")
if not GA4_PROPERTY_ID:
    missing_weekly.append("GA4_PROPERTY_ID")

if missing_weekly:
    st.sidebar.error("Weekly report missing:\n- " + "\n- ".join(missing_weekly))

st.sidebar.info(
    "GSC_SITE_URL must match property exactly:\n"
    "- Domain property: sc-domain:example.com\n"
    "- URL prefix: https://example.com/"
)

# Keywords input
st.subheader("📝 Keywords")
keywords_text = st.text_area(
    "Enter keywords (one per line)",
    height=180,
    placeholder="emergency plumber dubai\n24 hour plumber near me\nac repair dubai",
)
keywords = [k.strip() for k in keywords_text.splitlines() if k.strip()]

colA, colB, colC = st.columns([1, 1, 2])
store_for_date = colA.date_input("Store for date", value=today_local())
run_daily_btn = colB.button("✅ Run Daily Fetch & Store")
compare_btn = colC.button("📊 Compare Two Dates")

# Resolve location
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

st.info(f"Using location: **{resolved_location or '(none)'}** | gl: **{resolved_gl}** | hl: **{hl}** | device: **{device}**")

# DAILY RUN
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
                    "date": store_for_date.isoformat(),
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
            st.dataframe(pd.DataFrame(rows_out), use_container_width=True)
        finally:
            con.close()

# COMPARE TWO DATES
if compare_btn:
    con = db_connect()
    try:
        min_d, max_d = get_min_max_dates(con, project_domain)
        if not min_d:
            st.warning("No data found yet. Run Daily Fetch first.")
        else:
            c1, c2 = st.columns(2)
            d1 = c1.date_input("Date 1", value=max_d)
            d2 = c2.date_input("Date 2", value=max_d - timedelta(days=1) if max_d > min_d else max_d)

            rows1 = fetch_by_date(con, project_domain, d1)
            rows2 = fetch_by_date(con, project_domain, d2)

            def key(r): return (r["keyword"], r.get("location",""), r.get("device","desktop"))
            m1 = {key(r): r for r in rows1}
            m2 = {key(r): r for r in rows2}

            all_keys = sorted(set(m1.keys()) | set(m2.keys()), key=lambda x: x[0].lower())
            out = []
            for k in all_keys:
                r1 = m1.get(k, {})
                r2 = m2.get(k, {})
                p1 = safe_float(r1.get("position", 0))
                p2 = safe_float(r2.get("position", 0))
                delta = int(p1 - p2) if (p1 > 0 and p2 > 0) else ""

                out.append({
                    "keyword": k[0],
                    "location": k[1],
                    "device": k[2],
                    f"pos_{d1.isoformat()}": fmt_pos(p1),
                    f"pos_{d2.isoformat()}": fmt_pos(p2),
                    "delta": delta
                })

            st.subheader("📊 Rank Comparison")
            st.dataframe(pd.DataFrame(out), use_container_width=True)
    finally:
        con.close()

# WEEKLY EMAIL SEND
st.subheader("📩 Weekly SEO Report")
send_weekly_btn = st.button("📨 Send Weekly Email Now (Complete Report)")

if send_weekly_btn:
    missing = []
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and MAIL_FROM and REPORT_TO):
        missing.append("SMTP settings")
    if not GOOGLE_OK:
        missing.append("Google libraries")
    if not GOOGLE_CRED_FILE:
        missing.append("GOOGLE_SERVICE_ACCOUNT_JSON (Streamlit Secrets)")
    if not GSC_SITE_URL:
        missing.append("GSC_SITE_URL")
    if not GA4_PROPERTY_ID:
        missing.append("GA4_PROPERTY_ID")

    if missing:
        st.error("Weekly report requires:\n- " + "\n- ".join(missing))
        st.stop()

    con = db_connect()
    try:
        ws, we = last_full_week_range()
        prev_ws = ws - timedelta(days=7)
        prev_we = we - timedelta(days=7)

        curr_days = daterange(ws, we)
        prev_days = daterange(prev_ws, prev_we)

        curr_rows = fetch_week(con, project_domain, ws, we)
        prev_rows = fetch_week(con, project_domain, prev_ws, prev_we)

        curr_matrix = build_matrix(curr_rows, curr_days)
        prev_matrix = build_matrix(prev_rows, prev_days)

        expected_keywords = len(keywords) if keywords else max(len(curr_matrix), 0)

        # ---- GSC + GA4 (mandatory) ----
        gsc_curr = gsc_totals(GOOGLE_CRED_FILE, GSC_SITE_URL, ws.isoformat(), we.isoformat())
        gsc_prev = gsc_totals(GOOGLE_CRED_FILE, GSC_SITE_URL, prev_ws.isoformat(), prev_we.isoformat())
        gsc_top_queries = gsc_top(GOOGLE_CRED_FILE, GSC_SITE_URL, ws.isoformat(), we.isoformat(), dim="query", limit=10)
        gsc_top_pages = gsc_top(GOOGLE_CRED_FILE, GSC_SITE_URL, ws.isoformat(), we.isoformat(), dim="page", limit=10)

        ga_curr = ga4_totals(GOOGLE_CRED_FILE, GA4_PROPERTY_ID, ws.isoformat(), we.isoformat())
        ga_prev = ga4_totals(GOOGLE_CRED_FILE, GA4_PROPERTY_ID, prev_ws.isoformat(), prev_we.isoformat())
        ga_curr_org = ga4_organic_sessions(GOOGLE_CRED_FILE, GA4_PROPERTY_ID, ws.isoformat(), we.isoformat())
        ga_prev_org = ga4_organic_sessions(GOOGLE_CRED_FILE, GA4_PROPERTY_ID, prev_ws.isoformat(), prev_we.isoformat())

        # ---- LLM insights (strict) ----
        curr_sum = summarize_week(curr_matrix, expected_keywords)
        prev_sum = summarize_week(prev_matrix, expected_keywords)
        ups, downs = movers_vs_prev(curr_matrix, prev_matrix, limit=10)

        payload = f"""
PROJECT: {project_name} ({project_domain})
CURRENT WEEK: {ws.isoformat()} to {we.isoformat()}
PREVIOUS WEEK: {prev_ws.isoformat()} to {prev_we.isoformat()}

RANK KPIs:
Tracked: {curr_sum['keywords_tracked']}
Ranked: {curr_sum['ranked_keywords']}
Top3: {curr_sum['top3']}
Top10: {curr_sum['top10']}
Top20: {curr_sum['top20']}
Avg Best Position: {curr_sum['avg_best_position']:.2f}

GSC KPIs (WoW):
Clicks: {gsc_prev['clicks']} -> {gsc_curr['clicks']}
Impressions: {gsc_prev['impressions']} -> {gsc_curr['impressions']}
CTR: {gsc_prev['ctr']*100:.2f}% -> {gsc_curr['ctr']*100:.2f}%
Avg Position: {gsc_prev['position']:.2f} -> {gsc_curr['position']:.2f}

GA4 KPIs (WoW):
Users: {ga_prev['users']} -> {ga_curr['users']}
Sessions: {ga_prev['sessions']} -> {ga_curr['sessions']}
Organic Sessions: {ga_prev_org} -> {ga_curr_org}
Engagement Rate: {ga_prev['engagement_rate']*100:.2f}% -> {ga_curr['engagement_rate']*100:.2f}%
Conversions: {ga_prev['conversions']} -> {ga_curr['conversions']}

TOP IMPROVED:
{chr(10).join([f"- {m['keyword']}: {fmt_pos(m['prev_best'])} -> {fmt_pos(m['curr_best'])}" for m in ups[:5]])}

TOP DROPPED:
{chr(10).join([f"- {m['keyword']}: {fmt_pos(m['prev_best'])} -> {fmt_pos(m['curr_best'])}" for m in downs[:5]])}
""".strip()

        llm_html = generate_llm_insights_strict(payload)

        html = build_weekly_email_html(
            project_name=project_name,
            domain=project_domain,
            ws=ws, we=we,
            prev_ws=prev_ws, prev_we=prev_we,
            expected_keywords=expected_keywords,
            curr_matrix=curr_matrix,
            prev_matrix=prev_matrix,
            curr_days=curr_days,
            gsc_curr=gsc_curr,
            gsc_prev=gsc_prev,
            gsc_top_queries=gsc_top_queries,
            gsc_top_pages=gsc_top_pages,
            ga_curr=ga_curr,
            ga_prev=ga_prev,
            ga_curr_org=ga_curr_org,
            ga_prev_org=ga_prev_org,
            llm_html=llm_html
        )

        csv_bytes = build_weekly_csv(ws, we, prev_ws, prev_we, curr_days, curr_matrix, prev_matrix)
        csv_name = f"weekly_seo_report_{project_domain}_{ws.isoformat()}_{we.isoformat()}.csv"
        subject = f"Weekly SEO Report: {project_domain} ({ws.isoformat()} to {we.isoformat()})"

        send_email(subject, html, csv_bytes, csv_name)

        st.success("✅ Weekly report email sent successfully!")
        st.download_button("⬇️ Download Weekly CSV", csv_bytes, file_name=csv_name, mime="text/csv")
        with st.expander("Preview Weekly Email HTML"):
            st.components.v1.html(html, height=800, scrolling=True)

    except Exception as e:
        st.error(f"❌ Weekly mail failed: {e}")
    finally:
        con.close()
