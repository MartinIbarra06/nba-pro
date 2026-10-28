"""
🏀 NBA ANALYTICS PRO — Scouting Suite
=====================================
Módulos:
  1. Matchup Predictor (modelo de spread/total/probabilidad + forma + H2H)
  2. Comparador de jugadores (TS%, eFG%, USG%, PIE, radar percentil, tendencias)
  3. Player Props Studio (líneas Over/Under, hit rate, splits, B2B, proyección)
  4. Shot Charts (mapa de tiros, calor, hexbin, eficiencia por zona vs liga)
  5. Liga (mapas de eficiencia de jugadores y equipos, líderes)
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguegamefinder,
    playergamelog,
    shotchartdetail,
)
from nba_api.stats.static import teams as static_teams

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Analytics Pro", page_icon="🏀", layout="wide")

ORANGE, BLUE, GREEN, RED = "#ff6b00", "#38bdf8", "#22c55e", "#ef4444"
TXT, MUTED, GRID = "#e2e8f0", "#94a3b8", "#2a3140"
TIMEOUT = 45

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a1208 0%, #0b0e14 45%); }
section[data-testid="stSidebar"] { background: #0e131b; border-right: 1px solid #1f2733; }
.hero { padding: 22px 28px; border-radius: 18px; margin-bottom: 14px;
        background: linear-gradient(120deg, #1c1208 0%, #11161f 60%);
        border: 1px solid #2b3445; border-left: 6px solid #ff6b00; }
.hero h1 { margin: 0; font-size: 2rem; font-weight: 900; color: #fff; letter-spacing: -.02em; }
.hero p { margin: 4px 0 0 0; color: #94a3b8; }
.badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size:.75rem;
         font-weight:700; background:#ff6b0022; color:#ff9a4d; border:1px solid #ff6b0055; margin-right:6px;}
.kpi { background: linear-gradient(160deg,#161b22,#10141b); border:1px solid #232a36;
        border-left: 4px solid var(--accent); border-radius: 14px; padding: 14px 16px; height: 100%; }
.kpi-label { font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; color:#94a3b8; font-weight:600;}
.kpi-value { font-size:1.85rem; font-weight:800; color:#f8fafc; line-height:1.2; }
.kpi-sub { font-size:.78rem; color:#94a3b8; margin-top:2px; }
.kpi-sub.pos { color:#22c55e; } .kpi-sub.neg { color:#ef4444; }
.chip { display:inline-block; width:28px; height:28px; line-height:28px; text-align:center;
        border-radius:8px; font-weight:800; margin-right:5px; color:#fff; font-size:.85rem; }
.chip.w { background:#16a34a; } .chip.l { background:#dc2626; }
.pname { font-size:1.35rem; font-weight:800; color:#fff; margin-bottom:2px; }
.psub { color:#94a3b8; font-size:.85rem; margin-bottom:10px; }
h2, h3, h4 { color:#f1f5f9 !important; }
button[data-baseweb="tab"] { font-weight:700; }
div[data-testid="stExpander"] { border:1px solid #232a36; border-radius:12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE UI
# ─────────────────────────────────────────────────────────────────────────────
def show_plot(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpi_html(label: str, value: str, sub: str = "", color: str = ORANGE, tone: str = "") -> str:
    sub_html = f'<div class="kpi-sub {tone}">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi" style="--accent:{color}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def kpi_row(items: list[dict], container=None) -> None:
    cols = (container or st).columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(
            kpi_html(it["label"], it["value"], it.get("sub", ""), it.get("color", ORANGE), it.get("tone", "")),
            unsafe_allow_html=True,
        )


def style_fig(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    kw.setdefault("margin", dict(l=10, r=10, t=50, b=10))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, sans-serif", color=TXT),
        legend=dict(orientation="h", y=-0.18),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def radar(series: list[tuple[str, list[float], str]], cats: list[str], height: int = 420, title: str = "") -> go.Figure:
    fig = go.Figure()
    for name, vals, color in series:
        fig.add_trace(
            go.Scatterpolar(
                r=list(vals) + [vals[0]], theta=cats + [cats[0]], fill="toself",
                name=name, line=dict(color=color, width=2), opacity=0.6,
            )
        )
    fig.update_layout(
        title=title,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], showticklabels=False, gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID),
        ),
    )
    return style_fig(fig, height)


def chips(wl: list[str]) -> str:
    return "".join(f'<span class="chip {"w" if x == "W" else "l"}">{x}</span>' for x in wl)


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS AVANZADAS
# ─────────────────────────────────────────────────────────────────────────────
def ts_pct(pts, fga, fta):
    denom = 2 * (np.asarray(fga, dtype=float) + 0.44 * np.asarray(fta, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, np.asarray(pts, dtype=float) / denom * 100, np.nan)


def efg_pct(fgm, fg3m, fga):
    fga = np.asarray(fga, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fga > 0, (np.asarray(fgm, dtype=float) + 0.5 * np.asarray(fg3m, dtype=float)) / fga * 100, np.nan)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pct_rank(series: pd.Series, value: float, invert: bool = False) -> float:
    s = series.dropna()
    if s.empty or pd.isna(value):
        return 0.0
    return float(((s >= value) if invert else (s <= value)).mean() * 100)


def fmt(v, d: int = 1, suffix: str = "") -> str:
    return "N/D" if v is None or pd.isna(v) else f"{v:.{d}f}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE DATOS (cacheada y blindada con HEADERS para evitar bloqueos)
# ─────────────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com"
}


def season_str(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


def current_season() -> str:
    t = date.today()
    return season_str(t.year if t.month >= 10 else t.year - 1)


def season_options(n: int = 6) -> list[str]:
    y = int(current_season()[:4])
    return [season_str(y - i) for i in range(n)]


@st.cache_data(ttl=3600, show_spinner=False)
def load_league_players(season: str) -> pd.DataFrame:
    kw = dict(season=season, season_type_all_star="Regular Season", per_mode_detailed="PerGame", timeout=TIMEOUT, headers=HEADERS)
    try:
        base = leaguedashplayerstats.LeagueDashPlayerStats(**kw).get_data_frames()[0]
        if base.empty:
            return base
        adv = leaguedashplayerstats.LeagueDashPlayerStats(measure_type_detailed_defense="Advanced", **kw).get_data_frames()[0]
        keep = ["PLAYER_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "AST_PCT", "REB_PCT",
                "TM_TOV_PCT", "USG_PCT", "PACE", "PIE", "TS_PCT"]
        adv = adv[[c for c in keep if c in adv.columns]].rename(columns={"TS_PCT": "TS_OFICIAL"})
        df = base.merge(adv, on="PLAYER_ID", how="left")
        for c in ["AST_PCT", "REB_PCT", "TM_TOV_PCT", "USG_PCT", "PIE", "TS_OFICIAL"]:
            if c in df.columns:
                df[c] = df[c] * 100
        df["TS_CALC"] = ts_pct(df["PTS"], df["FGA"], df["FTA"])
        df["EFG_CALC"] = efg_pct(df["FGM"], df["FG3M"], df["FGA"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_team_table(season: str) -> pd.DataFrame:
    kw = dict(season=season, season_type_all_star="Regular Season", per_mode_detailed="PerGame", timeout=TIMEOUT, headers=HEADERS)
    try:
        adv = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense="Advanced", **kw).get_data_frames()[0]
        if adv.empty:
            return adv
        ff = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense="Four Factors", **kw).get_data_frames()[0]
        keep = [c for c in ["TEAM_ID", "FTA_RATE", "OPP_EFG_PCT", "OPP_TOV_PCT", "OPP_OREB_PCT"] if c in ff.columns]
        df = adv.merge(ff[keep], on="TEAM_ID", how="left")
        for c in ["EFG_PCT", "TS_PCT", "TM_TOV_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT", "AST_PCT",
                  "FTA_RATE", "OPP_EFG_PCT", "OPP_TOV_PCT", "OPP_OREB_PCT"]:
            if c in df.columns:
                df[c] = df[c] * 100
        return df
    except Exception:
        return pd.DataFrame()


def _prep_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["WL"]).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["LOCAL"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["RIVAL"] = df["MATCHUP"].str.split().str[-1]
    df["PTS_RIVAL"] = df["PTS"] - df["PLUS_MINUS"]
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_team_games(team_id: int, season: str) -> pd.DataFrame:
    try:
        df = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id, season_nullable=season,
            season_type_nullable="Regular Season", league_id_nullable="00", timeout=TIMEOUT, headers=HEADERS,
        ).get_data_frames()[0]
        return _prep_games(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_h2h(team_id: int, rival_id: int) -> pd.DataFrame:
    try:
        df = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id, vs_team_id_nullable=rival_id, league_id_nullable="00", timeout=TIMEOUT, headers=HEADERS,
        ).get_data_frames()[0]
        return _prep_games(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_player_log(player_id: int, season: str) -> pd.DataFrame:
    try:
        df = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season", timeout=TIMEOUT, headers=HEADERS
        ).get_data_frames()[0]
        if df.empty:
            return df
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        for c in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["LOCAL"] = df["MATCHUP"].str.contains("vs.", regex=False)
        df["RIVAL"] = df["MATCHUP"].str.split().str[-1]
        df["GAME_NO"] = np.arange(1, len(df) + 1)
        df["LABEL"] = df["GAME_DATE"].dt.strftime("%d/%m") + np.where(df["LOCAL"], " vs ", " @ ") + df["RIVAL"]
        df["TS_G"] = ts_pct(df["PTS"], df["FGA"], df["FTA"])
        df["EFG_G"] = efg_pct(df["FGM"], df["FG3M"], df["FGA"])
        return df
    except Exception:
        return pd.DataFrame()


# ── Descanso / Back-to-back ──────────────────────────────────────────────────
def add_rest(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return g
    c = g.sort_values("GAME_DATE").copy()
    c["REST"] = c["GAME_DATE"].diff().dt.days - 1
    c["REST_BUCKET"] = pd.cut(c["REST"].fillna(3), bins=[-1, 0, 1, 2, 99], labels=["B2B", "1 día", "2 días", "3+ días"])
    return c


def rest_info(g: pd.DataFrame, match_date) -> dict:
    out = {"rest": None, "b2b": False, "g7": 0, "three_in_four": False, "last": None}
    if g.empty:
        return out
    md = pd.Timestamp(match_date)
    prev = g[g["GAME_DATE"] < md]
    if prev.empty:
        return out
    last = prev["GAME_DATE"].max()
    out["last"] = last
    out["g7"] = int((prev["GAME_DATE"] >= md - pd.Timedelta(days=7)).sum())
    out["three_in_four"] = int((prev["GAME_DATE"] >= md - pd.Timedelta(days=3)).sum()) >= 2
    rest = (md - last).days - 1
    if rest <= 10:
        out["rest"], out["b2b"] = int(rest), rest == 0
    return out


def rest_adjustment(info: dict, pen: float) -> float:
    r = info["rest"]
    if r is None:
        return 0.0
    base = -pen if r == 0 else (0.0 if r == 1 else (0.25 * pen if r == 2 else 0.4 * pen))
    return base - (0.35 * pen if info["three_in_four"] else 0.0)


def attach_rest(log: pd.DataFrame, team_id: int) -> pd.DataFrame:
    log = log.copy()
    rest = None
    try:
        tg = add_rest(load_team_games(team_id, SEASON))
        if not tg.empty:
            rest = log["Game_ID"].map(dict(zip(tg["GAME_ID"], tg["REST"])))
    except Exception:
        rest = None
    if rest is None or rest.isna().all():
        rest = log["GAME_DATE"].diff().dt.days - 1
    log["REST"] = rest
    log["B2B"] = log["REST"] == 0
    return log


# ── Lesiones ─────────────────────────────────────────────────────────────────
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _espn_team_id(name: str):
    if name in TEAMS:
        return TEAMS[name]["id"]
    alt = name.replace("LA ", "Los Angeles ", 1) if name.startswith("LA ") else name
    if alt in TEAMS:
        return TEAMS[alt]["id"]
    nick = name.split()[-1] if name else ""
    hits = [v["id"] for k, v in TEAMS.items() if k.split()[-1] == nick]
    return hits[0] if len(hits) == 1 else None


def status_group(status: str):
    s = str(status).lower()
    if "out" in s:
        return "out"
    if any(w in s for w in ("doubtful", "questionable", "day-to-day", "day to day", "game time")):
        return "duda"
    return None


@st.cache_data(ttl=900, show_spinner=False)
def load_injuries():
    try:
        r = requests.get(ESPN_INJURIES_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {}, str(e)
    out: dict = {}
    for t in data.get("injuries", []):
        tid = _espn_team_id(t.get("displayName", ""))
        if tid is None:
            continue
        for inj in t.get("injuries", []):
            ath = inj.get("athlete") or {}
            det = inj.get("details") or {}
            out.setdefault(tid, []).append({
                "name": ath.get("displayName", ""),
                "status": inj.get("status", ""),
                "detail": " ".join(x for x in [det.get("type", ""), det.get("detail", "")] if x).strip(),
                "return": det.get("returnDate", "") or "",
                "comment": inj.get("shortComment", "") or "",
            })
    return out, None


def injury_for_player(name: str):
    data, _ = load_injuries()
    nn = _norm(name)
    for lst in data.values():
        for inj in lst:
            if _norm(inj["name"]) == nn:
                return inj
    return None


def player_impact(row: pd.Series, team_gp: float, k: float) -> float:
    pie = row.get("PIE")
    if pie is None or pd.isna(pie):
        return 0.0
    share = min(1.0, float(row["GP"]) / team_gp) if team_gp else 1.0
    return float(k * (row["MIN"] / 48.0) * max(pie - 8.0, 0.0) * share)


# ── Shot charts ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_shots(player_id: int, season: str):
    try:
        res = shotchartdetail.ShotChartDetail(
            team_id=0, player_id=player_id, context_measure_simple="FGA",
            season_nullable=season, season_type_all_star="Regular Season", timeout=TIMEOUT, headers=HEADERS,
        )
        dfs = res.get_data_frames()
        return dfs[0], dfs[1]
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def summarize_log(log: pd.DataFrame) -> dict:
    g = len(log)
    t = log[["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST", "STL", "BLK", "TOV", "MIN", "PLUS_MINUS"]].sum()
    safe = lambda a, b: float(a / b) if b else float("nan")
    return {
        "PJ": g, "MIN": t.MIN / g, "PTS": t.PTS / g, "REB": t.REB / g, "AST": t.AST / g,
        "STL": t.STL / g, "BLK": t.BLK / g, "TOV": t.TOV / g, "FG3M": t.FG3M / g,
        "TS": float(ts_pct(t.PTS, t.FGA, t.FTA)), "EFG": float(efg_pct(t.FGM, t.FG3M, t.FGA)),
        "FGP": safe(t.FGM, t.FGA) * 100, "TPP": safe(t.FG3M, t.FG3A) * 100, "FTP": safe(t.FTM, t.FTA) * 100,
        "FTr": safe(t.FTA, t.FGA) * 100, "TPAr": safe(t.FG3A, t.FGA) * 100,
        "AST_TO": safe(t.AST, t.TOV), "PM": t.PLUS_MINUS / g, "PTS36": safe(t.PTS, t.MIN) * 36,
    }


def league_avgs(df: pd.DataFrame) -> dict:
    tot = lambda c: float((df[c] * df["GP"]).sum())
    return {
        "TS": float(ts_pct(tot("PTS"), tot("FGA"), tot("FTA"))),
        "EFG": float(efg_pct(tot("FGM"), tot("FG3M"), tot("FGA"))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARRANQUE: equipos, temporada y datos base
# ─────────────────────────────────────────────────────────────────────────────
TEAMS = {t["full_name"]: {"id": t["id"], "abbr": t["abbreviation"]} for t in static_teams.get_teams()}
TEAM_NAMES = sorted(TEAMS)
ID2ABBR = {v["id"]: v["abbr"] for v in TEAMS.values()}
ID2NAME = {v["id"]: k for k, v in TEAMS.items()}

st.sidebar.markdown("## 🏀 NBA Analytics Pro")
season_sel = st.sidebar.selectbox("Temporada", season_options(), index=0)
if st.sidebar.button("🔄 Refrescar datos"):
    st.cache_data.clear()
    st.rerun()


def resolve_season(selected: str):
    opts = season_options(8)
    start = opts.index(selected)
    last_err = None
    for s in opts[start:start + 3]:
        try:
            df = load_league_players(s)
            if not df.empty:
                return s, df, None
        except Exception as e:
            last_err = e
    return selected, pd.DataFrame(), last_err


with st.spinner("Conectando con la NBA y cargando jugadores..."):
    SEASON, LEAGUE, _err = resolve_season(season_sel)

if LEAGUE.empty:
    st.error(
        "No se pudieron cargar datos de la NBA. Puede ser un límite/bloqueo temporal de stats.nba.com "
        f"o un problema de red. Detalle: {_err}"
    )
    st.stop()

if SEASON != season_sel:
    st.sidebar.warning(f"Aún no hay datos de {season_sel}; usando {SEASON}.")

try:
    TEAM_TBL = load_team_table(SEASON)
    if not TEAM_TBL.empty and "TEAM_ID" in TEAM_TBL.columns:
        TEAM_TBL = TEAM_TBL.set_index("TEAM_ID", drop=False)
    else:
        TEAM_TBL = pd.DataFrame()
except Exception:
    TEAM_TBL = pd.DataFrame()
    st.sidebar.error("No se pudieron cargar las estadísticas de equipos.")

LG = league_avgs(LEAGUE)
NORM2NAME = {_norm(n): n for n in LEAGUE["PLAYER_NAME"]}
_qual = LEAGUE[(LEAGUE["GP"] >= 0.25 * LEAGUE["GP"].max()) & (LEAGUE["MIN"] >= 12)]
QUAL = _qual if len(_qual) >= 20 else LEAGUE

st.sidebar.markdown("---")
st.sidebar.caption(f"Jugadores: **{len(LEAGUE)}** · TS% liga: **{LG['TS']:.1f}%** · eFG% liga: **{LG['EFG']:.1f}%**")
with st.sidebar.expander("📘 Glosario de métricas"):
    st.markdown(
        """
- **TS%** = PTS / (2·(FGA + 0.44·FTA)). Eficiencia real de anotación (dobles, triples y libres).
- **eFG%** = (FGM + 0.5·3PM) / FGA. Ajusta el valor extra del triple.
- **USG%** = % de jugadas de su equipo que finaliza mientras está en cancha.
- **PIE** = impacto global del jugador (% de eventos del partido).
- **Pace** = posesiones por 48 min.
- **ORtg / DRtg / NetRtg** = puntos anotados / recibidos / diferencial por 100 posesiones.
- **FTr** = FTA/FGA · **3PAr** = 3PA/FGA · **AST/TO** = asistencias por pérdida.
        """
    )

st.markdown(
    f"""
<div class="hero">
  <h1>🏀 NBA ANALYTICS PRO</h1>
  <p><span class="badge">Temporada {SEASON}</span><span class="badge">Scouting & Analytics</span>
  Predictor (B2B + lesiones) · Comparador · Player Props · Shot Charts · Mapas de liga</p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SELECTOR DE JUGADOR REUTILIZABLE
# ─────────────────────────────────────────────────────────────────────────────
def player_selector(key: str, default_team_idx: int = 0):
    c1, c2 = st.columns(2)
    team_name = c1.selectbox("Equipo", TEAM_NAMES, index=default_team_idx, key=f"{key}_team")
    pool = LEAGUE[LEAGUE["TEAM_ID"] == TEAMS[team_name]["id"]].sort_values("MIN", ascending=False)
    if pool.empty:
        c2.warning("Sin jugadores con partidos en esta temporada.")
        return None
    names = pool["PLAYER_NAME"].tolist()
    pick = c2.selectbox("Jugador", names, key=f"{key}_player")
    return pool[pool["PLAYER_NAME"] == pick].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · MATCHUP PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
TEAM_RADAR = [
    ("OFF_RATING", "Ataque (ORtg)", False), ("DEF_RATING", "Defensa (DRtg)", True),
    ("PACE", "Ritmo (Pace)", False), ("TS_PCT", "TS%", False), ("EFG_PCT", "eFG%", False),
    ("TM_TOV_PCT", "Cuidado balón", True), ("OREB_PCT", "OREB%", False), ("AST_PCT", "AST%", False),
]
TEAM_ROWS = [
    ("OFF_RATING", "Offensive Rating", 1), ("DEF_RATING", "Defensive Rating", 1), ("NET_RATING", "Net Rating", 1),
    ("PACE", "Pace", 1), ("TS_PCT", "TS%", 1), ("EFG_PCT", "eFG%", 1), ("TM_TOV_PCT", "TOV%", 1),
    ("OREB_PCT", "OREB%", 1), ("AST_PCT", "AST%", 1), ("FTA_RATE", "FT Rate (FTA/FGA)", 1),
]


def injury_panel(team_id: int, k: float):
    inj_map, _ = load_injuries()
    pool = LEAGUE[LEAGUE["TEAM_ID"] == team_id]
    team_gp = float(TEAM_TBL.loc[team_id, "GP"]) if (not TEAM_TBL.empty and team_id in TEAM_TBL.index) else 0.0
    out_def, dbt_def, rep_rows = [], [], []
    for inj in inj_map.get(team_id, []):
        m = NORM2NAME.get(_norm(inj["name"]))
        grp = status_group(inj["status"])
        if m and grp == "out":
            out_def.append(m)
        elif m and grp == "duda":
            dbt_def.append(m)
        rep_rows.append({"Jugador": inj["name"], "Estado": inj["status"], "Lesión": inj["detail"],
                         "Regreso": inj["return"], "En base": "✔" if m else "—"})

    options = sorted(set(pool["PLAYER_NAME"]) | set(out_def) | set(dbt_def))
    out = st.multiselect("Bajas confirmadas (peso 100%)", options, default=out_def, key=f"inj_out_{team_id}")
    dbt = st.multiselect("Dudosos (peso 50%)", options, default=dbt_def, key=f"inj_dbt_{team_id}")
    dbt = [x for x in dbt if x not in out]

    rows, total = [], 0.0
    for name, w, label in [(n, 1.0, "Baja") for n in out] + [(n, 0.5, "Duda") for n in dbt]:
        r = LEAGUE[LEAGUE["PLAYER_NAME"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        imp = player_impact(r, team_gp, k) * w
        total += imp
        rows.append({"Jugador": name, "Estado": label, "MIN": round(r["MIN"], 1), "PIE": round(float(r["PIE"]), 1) if pd.notna(r["PIE"]) else None,
                     "Impacto (pts)": round(-imp, 2)})
    total = min(total, 10.0)
    if rows:
        show_df(pd.DataFrame(rows))
    kpi_row([dict(label="Impacto estimado por bajas", value=f"-{total:.1f} pts", sub="margen por partido", color=RED if total > 0.5 else GREEN,
                  tone="neg" if total > 0.5 else "pos")])
    if rep_rows:
        with st.expander("Ver reporte completo de lesiones"):
            show_df(pd.DataFrame(rep_rows))
    return total


def tab_matchup():
    if TEAM_TBL.empty:
        st.warning("No hay estadísticas de equipos disponibles para esta temporada.")
        return

    c1, c2, c3 = st.columns([1, 1, 0.8])
    local = c1.selectbox("🏠 Equipo local", TEAM_NAMES, index=0, key="m_local")
    visitante = c2.selectbox("✈️ Equipo visitante", [t for t in TEAM_NAMES if t != local], index=0, key="m_vis")
    match_date = c3.date_input("📅 Fecha del partido", value=date.today(), key="m_date")
    idl, idv = TEAMS[local]["id"], TEAMS[visitante]["id"]
    al, av = TEAMS[local]["abbr"], TEAMS[visitante]["abbr"]

    with st.expander("⚙️ Parámetros del modelo"):
        k1, k2, k3, k4 = st.columns(4)
        hca = k1.slider("Ventaja de local (pts)", 0.0, 5.0, 2.5, 0.5)
        sigma = k2.slider("Desvío del margen (σ)", 9.0, 15.0, 12.0, 0.5)
        form_w = k3.slider("Peso de la forma reciente", 0.0, 0.5, 0.25, 0.05)
        n_games = k4.select_slider("Partidos de forma", options=[5, 10, 15], value=10)
        k5, k6, k7, k8 = st.columns(4)
        b2b_pen = k5.slider("Penalización B2B (pts)", 0.0, 4.0, 2.0, 0.5)
        inj_k = k6.slider("Intensidad de lesiones", 0.0, 1.5, 0.7, 0.1)
        use_rest = k7.checkbox("Aplicar descanso / B2B", True)
        use_inj = k8.checkbox("Aplicar lesiones", True)

    if idl not in TEAM_TBL.index or idv not in TEAM_TBL.index:
        st.warning("Alguno de los equipos no tiene datos en esta temporada.")
        return

    try:
        with st.spinner("Consultando partidos, descanso y lesiones..."):
            gl, gv = load_team_games(idl, SEASON), load_team_games(idv, SEASON)
            h2h = load_h2h(idl, idv)
            inj_map, inj_err = load_injuries()
    except Exception as e:
        st.error(f"No se pudo consultar la API de la NBA: {e}")
        return

    tl, tv = TEAM_TBL.loc[idl], TEAM_TBL.loc[idv]

    poss = (tl["PACE"] + tv["PACE"]) / 2
    pts_l = poss * (tl["OFF_RATING"] + tv["DEF_RATING"]) / 200
    pts_v = poss * (tv["OFF_RATING"] + tl["DEF_RATING"]) / 200
    base_margin = pts_l - pts_v

    def form_delta(g: pd.DataFrame) -> float:
        return 0.0 if g.empty else float(g.head(n_games)["PLUS_MINUS"].mean() - g["PLUS_MINUS"].mean())

    adj_form = float(np.clip(form_w * (form_delta(gl) - form_delta(gv)), -3, 3))
    ril, riv = rest_info(gl, match_date), rest_info(gv, match_date)
    adj_rest = (rest_adjustment(ril, b2b_pen) - rest_adjustment(riv, b2b_pen)) if use_rest else 0.0

    top = st.container()

    st.markdown("### 🧬 Radiografía táctica")
    cr1, cr2 = st.columns([1.1, 1])
    cols_ok = [(c, lab, inv) for c, lab, inv in TEAM_RADAR if c in TEAM_TBL.columns]
    cats = [lab for _, lab, _ in cols_ok]
    vl = [pct_rank(TEAM_TBL[c], tl[c], inv) for c, _, inv in cols_ok]
    vv = [pct_rank(TEAM_TBL[c], tv[c], inv) for c, _, inv in cols_ok]
    with cr1:
        show_plot(radar([(al, vl, ORANGE), (av, vv, BLUE)], cats, 430, "Percentil vs. la liga (100 = mejor)"))
    with cr2:
        rows = [{"Métrica": lab, al: fmt(tl[c]), av: fmt(tv[c])} for c, lab, _ in TEAM_ROWS if c in TEAM_TBL.columns]
        rows.insert(0, {"Métrica": "Récord", al: f"{int(tl['W'])}-{int(tl['L'])}", av: f"{int(tv['W'])}-{int(tv['L'])}"})
        st.markdown("&nbsp;")
        show_df(pd.DataFrame(rows))

    st.markdown("### 😴 Descanso y back-to-back")
    if SEASON != current_season():
        st.caption("⚠️ Se usan datos de una temporada anterior: el descanso calculado puede no ser representativo.")
    d1, d2 = st.columns(2)
    for col, name, info, color in ((d1, local, ril, ORANGE), (d2, visitante, riv, BLUE)):
        with col:
            st.markdown(f"**{name}**")
            r = info["rest"]
            if r is None:
                estado, tone = "N/D", ""
            elif info["b2b"]:
                estado, tone = "Back-to-back", "neg"
            elif r >= 2:
                estado, tone = "Descansado", "pos"
            else:
                estado, tone = "Normal", ""
            kpi_row([
                dict(label="Días de descanso", value="N/D" if r is None else str(r), color=color,
                     sub="" if info["last"] is None else f"Último: {info['last'].strftime('%d/%m')}"),
                dict(label="Estado", value=estado, color=color, tone=tone, sub="3 partidos en 4 noches" if info["three_in_four"] else ""),
                dict(label="Juegos últimos 7 días", value=str(info["g7"]), color=color),
            ])

    rb = []
    for ab, g in ((al, gl), (av, gv)):
        cdf = add_rest(g)
        if cdf.empty:
            continue
        a = cdf.groupby("REST_BUCKET", observed=False).agg(
            PJ=("WL", "size"), MARGEN=("PLUS_MINUS", "mean"), W=("WL", lambda s: int((s == "W").sum()))
        ).reset_index()
        a["MARGEN"] = a["MARGEN"].fillna(0)
        a["EQ"] = ab
        rb.append(a)
    if rb:
        rbd = pd.concat(rb)
        figr = px.bar(rbd, x="REST_BUCKET", y="MARGEN", color="EQ", barmode="group",
                      color_discrete_map={al: ORANGE, av: BLUE},
                      text=rbd["W"].astype(str) + "-" + (rbd["PJ"] - rbd["W"]).astype(str),
                      labels={"REST_BUCKET": "Descanso previo", "MARGEN": "Margen medio", "EQ": ""})
        figr.update_layout(title="Margen medio según días de descanso (etiqueta = récord G-P)")
        show_plot(style_fig(figr, 330))

    st.markdown("### 🩹 Lesiones y disponibilidad")
    if inj_err and not inj_map:
        st.info("No se pudo obtener el reporte automático de lesiones (fuente: ESPN, no oficial). Podés cargar las bajas manualmente.")
    else:
        st.caption("Reporte automático vía ESPN (fuente no oficial, puede tener retraso). Ajustá las listas si hace falta.")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"**{local}**")
        loss_l = injury_panel(idl, inj_k)
    with i2:
        st.markdown(f"**{visitante}**")
        loss_v = injury_panel(idv, inj_k)
    if not use_inj:
        loss_l = loss_v = 0.0

    pl, pv = pts_l + hca / 2, pts_v - hca / 2
    pl, pv = pl + adj_form / 2, pv - adj_form / 2
    pl, pv = pl + adj_rest / 2, pv - adj_rest / 2
    pl, pv = pl - 0.6 * loss_l + 0.4 * loss_v, pv + 0.4 * loss_l - 0.6 * loss_v
    adj_inj = loss_v - loss_l
    spread, total = pl - pv, pl + pv
    win_l = norm_cdf(spread / sigma) * 100
    win_v = 100 - win_l

    with top:
        st.markdown(f"### 📊 {local} vs {visitante} · {pd.Timestamp(match_date).strftime('%d/%m/%Y')}")
        fig = go.Figure()
        fig.add_bar(y=[""], x=[win_l], orientation="h", name=al, marker_color=ORANGE,
                    text=f"{al} {win_l:.1f}%", textposition="inside", insidetextanchor="middle")
        fig.add_bar(y=[""], x=[win_v], orientation="h", name=av, marker_color=BLUE,
                    text=f"{av} {win_v:.1f}%", textposition="inside", insidetextanchor="middle")
        fig.update_layout(barmode="stack", xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
        show_plot(style_fig(fig, 90, margin=dict(l=0, r=0, t=0, b=0)))

        fav, fav_sp = (al, spread) if spread >= 0 else (av, -spread)
        kpi_row([
            dict(label=f"Prob. victoria {al}", value=f"{win_l:.1f}%", sub=f"Cuota justa {100 / win_l:.2f}", color=ORANGE),
            dict(label=f"Prob. victoria {av}", value=f"{win_v:.1f}%", sub=f"Cuota justa {100 / win_v:.2f}", color=BLUE),
            dict(label="Spread proyectado", value=f"{fav} -{fav_sp:.1f}", sub=f"Marcador {pl:.0f} - {pv:.0f}", color=GREEN),
            dict(label="Total proyectado (O/U)", value=f"{total:.1f}", sub=f"Pace esperado {poss:.1f}", color="#a78bfa"),
        ])

        steps = [("Base (ORtg/DRtg)", base_margin), ("Localía", hca), ("Forma", adj_form),
                 ("Descanso/B2B", adj_rest), ("Lesiones", adj_inj)]
        figw = go.Figure(go.Waterfall(
            x=[s for s, _ in steps] + ["Spread final"], measure=["absolute"] + ["relative"] * 4 + ["total"],
            y=[v for _, v in steps] + [spread], text=[f"{v:+.1f}" for _, v in steps] + [f"{spread:+.1f}"],
            textposition="outside", connector=dict(line=dict(color=GRID)),
            increasing=dict(marker=dict(color=ORANGE)), decreasing=dict(marker=dict(color=BLUE)),
            totals=dict(marker=dict(color=GREEN)),
        ))
        figw.update_layout(title=f"Construcción del margen proyectado (+ favorece a {al}, − favorece a {av})")
        show_plot(style_fig(figw, 320, showlegend=False))
        st.caption("Modelo: posesiones esperadas × cruce ORtg/DRtg + localía + forma reciente + descanso/B2B + bajas. "
                   "Probabilidad vía distribución normal del margen. Estimación estadística, no una garantía.")

    st.markdown("### 🔥 Forma reciente")
    f1, f2 = st.columns(2)
    for col, g, name, ab, color in [(f1, gl, local, al, ORANGE), (f2, gv, visitante, av, BLUE)]:
        with col:
            st.markdown(f"**{name}**")
            if g.empty:
                st.info("Sin partidos registrados.")
                continue
            last = g.head(n_games)
            wins = int((last["WL"] == "W").sum())
            st.markdown(chips(last["WL"].head(5).tolist()[::-1]), unsafe_allow_html=True)
            kpi_row([
                dict(label=f"Récord L{len(last)}", value=f"{wins}-{len(last) - wins}", color=color),
                dict(label="Margen medio", value=f"{last['PLUS_MINUS'].mean():+.1f}", color=color,
                     tone="pos" if last["PLUS_MINUS"].mean() >= 0 else "neg",
                     sub=f"Temp: {g['PLUS_MINUS'].mean():+.1f}"),
                dict(label="PPG / Recibe", value=f"{last['PTS'].mean():.0f} / {last['PTS_RIVAL'].mean():.0f}", color=color),
            ])
            ch = last.iloc[::-1]
            figm = go.Figure(go.Bar(
                x=ch["GAME_DATE"].dt.strftime("%d/%m") + " " + np.where(ch["LOCAL"], "vs ", "@ ") + ch["RIVAL"],
                y=ch["PLUS_MINUS"], marker_color=np.where(ch["PLUS_MINUS"] >= 0, GREEN, RED),
                text=ch["PLUS_MINUS"].astype(int).astype(str), textposition="outside",
            ))
            figm.update_layout(title="Margen por partido")
            show_plot(style_fig(figm, 300, showlegend=False))
            tbl = g.head(5).assign(
                Fecha=g.head(5)["GAME_DATE"].dt.strftime("%d/%m/%Y"),
                Marcador=g.head(5)["PTS"].astype(int).astype(str) + " - " + g.head(5)["PTS_RIVAL"].astype(int).astype(str),
            )[["Fecha", "MATCHUP", "WL", "Marcador", "PLUS_MINUS"]].rename(columns={"MATCHUP": "Partido", "WL": "Res", "PLUS_MINUS": "+/-"})
            show_df(tbl)

    st.markdown(f"### ⚔️ Historial directo: {local} vs {visitante}")
    if h2h.empty:
        st.info("No hay cruces registrados.")
        return
    last = h2h.head(10)
    w = int((last["WL"] == "W").sum())
    totals = last["PTS"] + last["PTS_RIVAL"]
    kpi_row([
        dict(label=f"Récord {al} (últ. {len(last)})", value=f"{w}-{len(last) - w}", color=ORANGE),
        dict(label="Margen medio", value=f"{last['PLUS_MINUS'].mean():+.1f}", sub=f"desde {al}", color=GREEN,
             tone="pos" if last["PLUS_MINUS"].mean() >= 0 else "neg"),
        dict(label="Total medio de puntos", value=f"{totals.mean():.1f}", color="#a78bfa"),
        dict(label="Cruces en base", value=str(len(h2h)), sub="todas las temporadas", color=BLUE),
    ])
    hc1, hc2 = st.columns([1.2, 1])
    with hc1:
        ch = last.iloc[::-1]
        figh = go.Figure(go.Bar(
            x=ch["GAME_DATE"].dt.strftime("%d/%m/%y"), y=ch["PLUS_MINUS"],
            marker_color=np.where(ch["PLUS_MINUS"] >= 0, ORANGE, BLUE), hovertext=ch["MATCHUP"],
        ))
        figh.update_layout(title=f"Margen H2H desde la óptica de {al}")
        show_plot(style_fig(figh, 320, showlegend=False))
    with hc2:
        t8 = last.head(8).assign(
            Fecha=last.head(8)["GAME_DATE"].dt.strftime("%d/%m/%Y"),
            Marcador=last.head(8)["PTS"].astype(int).astype(str) + " - " + last.head(8)["PTS_RIVAL"].astype(int).astype(str),
        )[["Fecha", "MATCHUP", "WL", "Marcador"]].rename(columns={"MATCHUP": "Partido", "WL": "Res"})
        show_df(t8)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · COMPARADOR DE JUGADORES
# ─────────────────────────────────────────────────────────────────────────────
PLAYER_RADAR = [
    ("PTS", "Puntos"), ("REB", "Rebotes"), ("AST", "Asistencias"), ("STL", "Robos"), ("BLK", "Tapones"),
    ("TS_CALC", "TS%"), ("EFG_CALC", "eFG%"), ("USG_PCT", "USG%"), ("PIE", "PIE"),
]
COMPARE_ROWS = [
    ("Partidos", "PJ", 0, True), ("Minutos", "MIN", 1, True), ("Puntos", "PTS", 1, True),
    ("Rebotes", "REB", 1, True), ("Asistencias", "AST", 1, True), ("Robos", "STL", 1, True),
    ("Tapones", "BLK", 1, True), ("Pérdidas", "TOV", 1, False), ("Triples/partido", "FG3M", 1, True),
    ("FG%", "FGP", 1, True), ("3P%", "TPP", 1, True), ("FT%", "FTP", 1, True),
    ("TS%", "TS", 1, True), ("eFG%", "EFG", 1, True), ("FT Rate", "FTr", 1, True),
    ("3PA Rate", "TPAr", 1, True), ("AST/TO", "AST_TO", 2, True), ("Puntos por 36'", "PTS36", 1, True),
    ("+/- por partido", "PM", 1, True), ("USG%", "USG", 1, True), ("PIE", "PIE", 1, True),
    ("Net Rating", "NET", 1, True),
]


def _player_header(p: pd.Series, s: dict, color: str, container) -> None:
    team = ID2ABBR.get(int(p["TEAM_ID"]), p.get("TEAM_ABBREVIATION", ""))
    container.markdown(
        f'<div class="pname" style="color:{color}">{p["PLAYER_NAME"]}</div>'
        f'<div class="psub">{team} · {int(p["AGE"])} años · {s["PJ"]} PJ · {s["MIN"]:.1f} min</div>',
        unsafe_allow_html=True,
    )


def _eff_cards(p: pd.Series, s: dict, color: str, container) -> None:
    def tone(d):
        return "pos" if d >= 0 else "neg"
    d_ts, d_efg = s["TS"] - LG["TS"], s["EFG"] - LG["EFG"]
    items = [
        dict(label="True Shooting %", value=fmt(s["TS"], 1, "%"), sub=f"{d_ts:+.1f} vs liga ({LG['TS']:.1f}%)", color=color, tone=tone(d_ts)),
        dict(label="Effective FG %", value=fmt(s["EFG"], 1, "%"), sub=f"{d_efg:+.1f} vs liga ({LG['EFG']:.1f}%)", color=color, tone=tone(d_efg)),
        dict(label="Usage Rate", value=fmt(p.get("USG_PCT"), 1, "%"), sub=f"Pct. liga {pct_rank(QUAL['USG_PCT'], p.get('USG_PCT')):.0f}", color=color),
        dict(label="PIE", value=fmt(p.get("PIE"), 1, "%"), sub=f"Net Rtg {fmt(p.get('NET_RATING'), 1)}", color=color),
    ]
    kpi_row(items, container)


def tab_compare():
    st.markdown("### ⚔️ Comparador cara a cara")
    ca, cb = st.columns(2)
    with ca:
        pa = player_selector("cmpA", 0)
    with cb:
        pb = player_selector("cmpB", 1)
    if pa is None or pb is None:
        return
    if pa["PLAYER_ID"] == pb["PLAYER_ID"]:
        st.info("Elige dos jugadores distintos para comparar.")
        return

    try:
        with st.spinner("Cargando registros de partidos..."):
            la, lb = load_player_log(int(pa["PLAYER_ID"]), SEASON), load_player_log(int(pb["PLAYER_ID"]), SEASON)
    except Exception as e:
        st.error(f"No se pudo consultar la API de la NBA: {e}")
        return
    if la.empty or lb.empty:
        st.warning("Alguno de los jugadores no tiene partidos registrados.")
        return

    sa, sb = summarize_log(la), summarize_log(lb)
    for s, p in ((sa, pa), (sb, pb)):
        s["USG"], s["PIE"], s["NET"] = p.get("USG_PCT"), p.get("PIE"), p.get("NET_RATING")
    na, nb = pa["PLAYER_NAME"], pb["PLAYER_NAME"]

    st.markdown("---")
    h1, h2 = st.columns(2)
    _player_header(pa, sa, ORANGE, h1)
    _player_header(pb, sb, BLUE, h2)
    k1, k2 = st.columns(2)
    _eff_cards(pa, sa, ORANGE, k1)
    _eff_cards(pb, sb, BLUE, k2)

    st.markdown("### 📊 Perfil y producción")
    g1, g2 = st.columns([1.1, 1])
    with g1:
        vals = {}
        for key, s, p in (("a", sa, pa), ("b", sb, pb)):
            ov = {"TS_CALC": s["TS"], "EFG_CALC": s["EFG"], "PTS": s["PTS"], "REB": s["REB"], "AST": s["AST"],
                  "STL": s["STL"], "BLK": s["BLK"], "USG_PCT": p.get("USG_PCT"), "PIE": p.get("PIE")}
            vals[key] = [pct_rank(QUAL[c], ov[c]) for c, _ in PLAYER_RADAR]
        show_plot(radar([(na, vals["a"], ORANGE), (nb, vals["b"], BLUE)], [l for _, l in PLAYER_RADAR], 440,
                        "Percentil vs. jugadores de rotación (100 = élite)"))
    with g2:
        keys = [("PTS", "PTS"), ("REB", "REB"), ("AST", "AST"), ("STL", "STL"), ("BLK", "BLK"), ("TOV", "TOV")]
        figb = go.Figure()
        figb.add_bar(name=na, x=[k for k, _ in keys], y=[sa[c] for _, c in keys], marker_color=ORANGE)
        figb.add_bar(name=nb, x=[k for k, _ in keys], y=[sb[c] for _, c in keys], marker_color=BLUE)
        figb.update_layout(barmode="group", title="Promedios por partido")
        show_plot(style_fig(figb, 440))

    e1, e2 = st.columns(2)
    with e1:
        cats = ["TS%", "eFG%", "FG%", "3P%", "FT%"]
        figs = go.Figure()
        figs.add_bar(name=na, x=cats, y=[sa["TS"], sa["EFG"], sa["FGP"], sa["TPP"], sa["FTP"]], marker_color=ORANGE,
                     text=[f"{v:.1f}" for v in [sa["TS"], sa["EFG"], sa["FGP"], sa["TPP"], sa["FTP"]]], textposition="outside")
        figs.add_bar(name=nb, x=cats, y=[sb["TS"], sb["EFG"], sb["FGP"], sb["TPP"], sb["FTP"]], marker_color=BLUE,
                     text=[f"{v:.1f}" for v in [sb["TS"], sb["EFG"], sb["FGP"], sb["TPP"], sb["FTP"]]], textposition="outside")
        figs.add_hline(y=LG["TS"], line_dash="dot", line_color=MUTED, annotation_text="TS% liga")
        figs.update_layout(barmode="group", title="Eficiencia de tiro (%)")
        show_plot(style_fig(figs, 360))
    with e2:
        figr = go.Figure()
        for name, log, color in ((na, la, ORANGE), (nb, lb, BLUE)):
            r = log[["PTS", "FGA", "FTA"]].rolling(5, min_periods=3).sum()
            figr.add_trace(go.Scatter(x=log["GAME_NO"], y=ts_pct(r["PTS"], r["FGA"], r["FTA"]), mode="lines+markers",
                                      name=name, line=dict(color=color, width=3)))
        figr.add_hline(y=LG["TS"], line_dash="dot", line_color=MUTED, annotation_text="TS% liga")
        figr.update_layout(title="Tendencia TS% (media móvil 5 partidos)", xaxis_title="N° de partido")
        show_plot(style_fig(figr, 360))

    p1, p2 = st.columns(2)
    for col, title, key, ylab in ((p1, "Puntos por partido", "PTS", "PTS"), (p2, "Minutos por partido", "MIN", "MIN")):
        with col:
            figp = go.Figure()
            for name, log, color in ((na, la, ORANGE), (nb, lb, BLUE)):
                figp.add_trace(go.Scatter(x=log["GAME_NO"], y=log[key], mode="lines+markers", name=name,
                                          line=dict(color=color, width=2), marker=dict(size=5), opacity=0.9))
            figp.update_layout(title=title, xaxis_title="N° de partido", yaxis_title=ylab)
            show_plot(style_fig(figp, 320))

    st.markdown("### 📋 Comparativa completa")
    rows = []
    for label, key, d, higher in COMPARE_ROWS:
        a, b = sa.get(key), sb.get(key)
        if a is None or b is None or pd.isna(a) or pd.isna(b) or a == b:
            adv = "="
        else:
            adv = na if ((a > b) == higher) else nb
        rows.append({"Métrica": label, na: fmt(a, d), nb: fmt(b, d), "Ventaja": adv})
    show_df(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · PLAYER PROPS STUDIO
# ─────────────────────────────────────────────────────────────────────────────
MARKETS = {
    "Puntos (PTS)": ["PTS"], "Rebotes (REB)": ["REB"], "Asistencias (AST)": ["AST"],
    "Triples (3PM)": ["FG3M"], "Robos (STL)": ["STL"], "Bloqueos (BLK)": ["BLK"],
    "Pts + Reb + Ast": ["PTS", "REB", "AST"], "Pts + Reb": ["PTS", "REB"],
    "Pts + Ast": ["PTS", "AST"], "Reb + Ast": ["REB", "AST"], "Minutos (MIN)": ["MIN"],
}


def tab_props():
    st.markdown("### ⭐ Player Props Studio")
    p = player_selector("props", 0)
    if p is None:
        return
    pid = int(p["PLAYER_ID"])
    try:
        with st.spinner(f"Consultando partidos de {p['PLAYER_NAME']}..."):
            log = load_player_log(pid, SEASON)
    except Exception as e:
        st.error(f"No se pudo consultar la API de la NBA: {e}")
        return
    if log.empty:
        st.warning("Sin partidos registrados para este jugador.")
        return

    log = attach_rest(log, int(p["TEAM_ID"]))
    status_banner(p)
    s = summarize_log(log)
    st.markdown("#### Perfil de eficiencia")
    _eff_cards(p, s, ORANGE, st)

    st.markdown("#### Configuración de la línea")
    c1, c2 = st.columns([2, 1])
    market = c1.selectbox("Mercado", list(MARKETS))
    cols = MARKETS[market]
    default = float(math.floor(log[cols].sum(axis=1).mean()) + 0.5)
    line = c2.number_input("Línea (Over/Under)", min_value=0.0, value=default, step=0.5, key=f"line_{pid}_{market}")
    scope = st.radio("Filtro de partidos", ["Todos", "Local", "Visitante", "Back-to-back", "Con descanso"], horizontal=True)

    if scope == "Todos":
        sub = log
    elif scope == "Local":
        sub = log[log["LOCAL"]]
    elif scope == "Visitante":
        sub = log[~log["LOCAL"]]
    elif scope == "Back-to-back":
        sub = log[log["B2B"]]
    else:
        sub = log[(~log["B2B"]) & (log["REST"] >= 1)]
    if sub.empty:
        st.info("No hay partidos con ese filtro.")
        return
    vals = sub[cols].sum(axis=1)
    n = len(vals)
    l5, l10, avg = vals.tail(5).mean(), vals.tail(10).mean(), vals.mean()
    over = float((vals > line).mean() * 100)
    under = float((vals < line).mean() * 100)
    over5, over10 = float((vals.tail(5) > line).mean() * 100), float((vals.tail(10) > line).mean() * 100)

    proj = 0.4 * l5 + 0.3 * l10 + 0.3 * avg
    sd = max(float(vals.std(ddof=1)) if n > 1 else 1.0, 0.5)
    p_over = (1 - norm_cdf((line - proj) / sd)) * 100
    p_under = 100 - p_over

    kpi_row([
        dict(label=f"Promedio ({n} PJ)", value=f"{avg:.1f}", sub=f"Mediana {vals.median():.1f}", color=ORANGE),
        dict(label="Últimos 5", value=f"{l5:.1f}", sub=f"{l5 - line:+.1f} vs línea", color=BLUE, tone="pos" if l5 >= line else "neg"),
        dict(label="Últimos 10", value=f"{l10:.1f}", sub=f"{l10 - line:+.1f} vs línea", color=BLUE, tone="pos" if l10 >= line else "neg"),
        dict(label="Hit Rate OVER", value=f"{over:.0f}%", sub=f"L10: {over10:.0f}% · L5: {over5:.0f}%", color=GREEN),
        dict(label="Hit Rate UNDER", value=f"{under:.0f}%", sub=f"Push: {100 - over - under:.0f}%", color=RED),
    ])
    st.write("")
    kpi_row([
        dict(label="Proyección ponderada", value=f"{proj:.1f}", sub="40% L5 · 30% L10 · 30% temp", color="#a78bfa"),
        dict(label="Prob. Over (modelo)", value=f"{p_over:.0f}%", sub=f"Cuota justa {100 / max(p_over, 1):.2f}", color=GREEN),
        dict(label="Prob. Under (modelo)", value=f"{p_under:.0f}%", sub=f"Cuota justa {100 / max(p_under, 1):.2f}", color=RED),
        dict(label="Volatilidad (σ)", value=f"{sd:.1f}", sub=f"Mín {vals.min():.0f} · Máx {vals.max():.0f}", color=MUTED),
    ])
    if n < 8:
        st.warning("Muestra pequeña (< 8 partidos): interpreta hit rate y proyección con cautela.")

    ch = sub.assign(VAL=vals)
    figb = go.Figure()
    figb.add_bar(x=ch["LABEL"], y=ch["VAL"], marker_color=np.where(ch["VAL"] > line, GREEN, np.where(ch["VAL"] < line, RED, MUTED)),
                 text=ch["VAL"].round(0).astype(int).astype(str), textposition="outside", name=market,
                 hovertext=ch["MATCHUP"])
    figb.add_trace(go.Scatter(x=ch["LABEL"], y=ch["VAL"].rolling(5, min_periods=1).mean(), mode="lines",
                              name="Media móvil (5)", line=dict(color=BLUE, width=3)))
    figb.add_hline(y=line, line_dash="dash", line_color=ORANGE, annotation_text=f"Línea {line}")
    figb.update_layout(title=f"{p['PLAYER_NAME']} · {market} por partido (verde = Over)")
    figb.update_xaxes(tickangle=-60)
    show_plot(style_fig(figb, 420))

    d1, d2 = st.columns(2)
    with d1:
        figh = go.Figure(go.Histogram(x=vals, nbinsx=min(max(int(vals.max() - vals.min()) + 1, 6), 20),
                                      marker_color=ORANGE, opacity=0.85))
        figh.add_vline(x=line, line_dash="dash", line_color=GREEN, annotation_text="Línea")
        figh.add_vline(x=proj, line_dash="dot", line_color=BLUE, annotation_text="Proyección")
        figh.update_layout(title="Distribución de resultados", bargap=0.05)
        show_plot(style_fig(figh, 340, showlegend=False))
    with d2:
        rows = []
        for name, d in (("Todos", log), ("Local", log[log["LOCAL"]]), ("Visitante", log[~log["LOCAL"]]),
                        ("Back-to-back", log[log["B2B"]]), ("Con descanso", log[(~log["B2B"]) & (log["REST"] >= 1)])):
            if d.empty:
                continue
            v = d[cols].sum(axis=1)
            rows.append({"Sede": name, "PJ": len(d), "Promedio": round(v.mean(), 1), "Mediana": round(v.median(), 1),
                         "Over %": round((v > line).mean() * 100), "TS%": round(float(ts_pct(d.PTS.sum(), d.FGA.sum(), d.FTA.sum())), 1),
                         "MIN": round(d["MIN"].mean(), 1)})
        st.markdown("**Splits: sede y descanso (B2B)**")
        show_df(pd.DataFrame(rows))
        opp = (log.assign(VAL=log[cols].sum(axis=1)).groupby("RIVAL")["VAL"].agg(["count", "mean"])
               .rename(columns={"count": "PJ", "mean": "Prom"}).reset_index().sort_values("Prom", ascending=False))
        opp["Prom"] = opp["Prom"].round(1)
        st.markdown("**Rendimiento por rival (esta temporada)**")
        show_df(opp.rename(columns={"RIVAL": "Rival"}).head(8))

    st.markdown("#### 📅 Últimos 10 partidos")
    last = log.tail(10).iloc[::-1].copy()
    tbl = pd.DataFrame({
        "Fecha": last["GAME_DATE"].dt.strftime("%d/%m/%Y"), "Partido": last["MATCHUP"], "Res": last["WL"],
        "MIN": last["MIN"].round(0), "PTS": last["PTS"], "REB": last["REB"], "AST": last["AST"],
        "3PM": last["FG3M"], "STL": last["STL"], "BLK": last["BLK"],
        "TS%": pd.Series(last["TS_G"]).round(1).values, "eFG%": pd.Series(last["EFG_G"]).round(1).values,
        "+/-": last["PLUS_MINUS"],
    })
    show_df(tbl)
    st.caption("⚠️ Herramienta de análisis estadístico con fines informativos. No constituye asesoramiento de apuestas.")


def status_banner(p: pd.Series) -> None:
    data, err = load_injuries()
    if err and not data:
        st.caption("Reporte de lesiones no disponible en este momento.")
    else:
        inj = injury_for_player(p["PLAYER_NAME"])
        if inj:
            extra = f" · {inj['detail']}" if inj["detail"] else ""
            ret = f" · regreso estimado: {inj['return']}" if inj["return"] else ""
            (st.error if status_group(inj["status"]) == "out" else st.warning)(
                f"🩹 **{p['PLAYER_NAME']}**: {inj['status']}{extra}{ret}")
        else:
            st.success(f"✅ {p['PLAYER_NAME']} no figura en el reporte de lesiones.")
    try:
        ri = rest_info(load_team_games(int(p["TEAM_ID"]), SEASON), date.today())
        if ri["rest"] is not None:
            tag = " · ⚠️ BACK-TO-BACK" if ri["b2b"] else ""
            st.caption(f"Descanso de su equipo para hoy: {ri['rest']} día(s) · {ri['g7']} juegos en los últimos 7 días{tag}")
    except Exception:
        pass


def court_traces() -> list:
    ln = dict(color="#64748b", width=2)

    def sc(x, y, dash=None):
        return go.Scatter(x=x, y=y, mode="lines", line=dict(**ln, **({"dash": dash} if dash else {})),
                          hoverinfo="skip", showlegend=False)

    th = np.linspace(0, np.pi, 60)
    tc = np.linspace(0, 2 * np.pi, 40)
    ang = np.arccos(220 / 237.5)
    ta = np.linspace(np.pi - ang, ang, 100)
    arc_x, arc_y = 237.5 * np.cos(ta), 237.5 * np.sin(ta)
    return [
        sc([-250, 250, 250, -250, -250], [-47.5, -47.5, 422.5, 422.5, -47.5]),
        sc([-80, -80, 80, 80], [-47.5, 142.5, 142.5, -47.5]),
        sc([-60, -60, 60, 60], [-47.5, 142.5, 142.5, -47.5]),
        sc(60 * np.cos(th), 142.5 + 60 * np.sin(th)),
        sc(60 * np.cos(th), 142.5 - 60 * np.sin(th), dash="dot"),
        sc(40 * np.cos(th), 40 * np.sin(th)),
        sc(7.5 * np.cos(tc), 7.5 * np.sin(tc)),
        sc([-30, 30], [-7.5, -7.5]),
        sc(np.r_[-220, arc_x, 220], np.r_[-47.5, arc_y, -47.5]),
        sc(60 * np.cos(th), 422.5 - 60 * np.sin(th)),
    ]


def tab_shots():
    st.markdown("### 🎯 Shot Charts")
    p = player_selector("shots", 0)
    if p is None:
        return
    pid = int(p["PLAYER_ID"])
    status_banner(p)
    try:
        with st.spinner(f"Descargando tiros de {p['PLAYER_NAME']}..."):
            shots, lg_zones = load_shots(pid, SEASON)
    except Exception as e:
        st.error(f"No se pudo consultar la API de la NBA: {e}")
        return
    if shots.empty:
        st.warning("Sin tiros registrados para este jugador en la temporada.")
        return

    shots = shots.copy()
    shots["PTS_VAL"] = np.where(shots["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2)
    shots["PTS_MADE"] = shots["SHOT_MADE_FLAG"] * shots["PTS_VAL"]

    order = shots.drop_duplicates("GAME_ID").sort_values("GAME_DATE")["GAME_ID"].tolist()
    f1, f2, f3, f4 = st.columns([1, 1.3, 1.3, 1.4])
    tipo = f1.selectbox("Tipo de tiro", ["Todos", "2 pts", "3 pts"])
    per_sel = f2.multiselect("Cuartos", ["Q1", "Q2", "Q3", "Q4", "OT"], default=["Q1", "Q2", "Q3", "Q4", "OT"])
    n_last = f3.slider("Últimos N partidos", 1, max(len(order), 2), len(order))
    vista = f4.selectbox("Vista", ["Aciertos y fallos", "Mapa de calor (volumen)", "Eficiencia por área (hexbin)"])

    d = shots[shots["GAME_ID"].isin(order[-n_last:])]
    if tipo != "Todos":
        d = d[d["PTS_VAL"] == (2 if tipo == "2 pts" else 3)]
    qs = [int(q[1]) for q in per_sel if q.startswith("Q")]
    d = d[d["PERIOD"].isin(qs) | ((d["PERIOD"] >= 5) & ("OT" in per_sel))]
    if d.empty:
        st.info("Ningún tiro cumple los filtros.")
        return

    fga, fgm = len(d), int(d["SHOT_MADE_FLAG"].sum())
    fg3m = int(d.loc[d["PTS_VAL"] == 3, "SHOT_MADE_FLAG"].sum())
    efg = (fgm + 0.5 * fg3m) / fga * 100
    pps = d["PTS_MADE"].sum() / fga
    lz = pd.DataFrame()
    lg_pps = float("nan")
    if not lg_zones.empty:
        lz = lg_zones.groupby("SHOT_ZONE_BASIC")[["FGA", "FGM"]].sum()
        lz["PV"] = [3 if "3" in z_ else 2 for z_ in lz.index]
        lz["LG_FG"] = lz["FGM"] / lz["FGA"] * 100
        lz["LG_PPS"] = lz["FGM"] * lz["PV"] / lz["FGA"]
        lg_pps = float((lz["FGM"] * lz["PV"]).sum() / lz["FGA"].sum())
    kpi_row([
        dict(label="Tiros (FGA)", value=str(fga), sub=f"{n_last} partido(s) · dist. media {d['SHOT_DISTANCE'].mean():.1f} ft", color=ORANGE),
        dict(label="FG%", value=f"{fgm / fga * 100:.1f}%", sub=f"{fgm}/{fga}", color=ORANGE),
        dict(label="eFG%", value=f"{efg:.1f}%", sub="ajustado por triples", color=ORANGE),
        dict(label="Puntos por tiro", value=f"{pps:.2f}",
             sub="" if np.isnan(lg_pps) else f"{pps - lg_pps:+.2f} vs liga ({lg_pps:.2f})",
             tone="" if np.isnan(lg_pps) else ("pos" if pps >= lg_pps else "neg"), color=GREEN),
        dict(label="% Triples", value=f"{(d['PTS_VAL'] == 3).mean() * 100:.0f}%", color=BLUE),
        dict(label="% Zona restringida", value=f"{(d['SHOT_ZONE_BASIC'] == 'Restricted Area').mean() * 100:.0f}%",
             sub=f"Mid-range {(d['SHOT_ZONE_BASIC'] == 'Mid-Range').mean() * 100:.0f}%", color=BLUE),
    ])
    st.write("")

    z = d.groupby("SHOT_ZONE_BASIC").agg(FGA=("SHOT_ATTEMPTED_FLAG", "sum"), FGM=("SHOT_MADE_FLAG", "sum"),
                                         PTS=("PTS_MADE", "sum"))
    z["FG%"] = z["FGM"] / z["FGA"] * 100
    z["Freq%"] = z["FGA"] / z["FGA"].sum() * 100
    z["PPS"] = z["PTS"] / z["FGA"]
    if not lz.empty:
        z = z.join(lz[["LG_FG", "LG_PPS"]])
    else:
        z["LG_FG"], z["LG_PPS"] = np.nan, np.nan
    z["ΔFG%"] = z["FG%"] - z["LG_FG"]
    z["ΔPPS"] = z["PPS"] - z["LG_PPS"]
    z = z.sort_values("FGA", ascending=False).reset_index()

    cl, cr = st.columns([1.3, 1])
    with cl:
        fig = go.Figure()
        if vista.startswith("Aciertos"):
            hov = d["ACTION_TYPE"].astype(str) + " · " + d["SHOT_DISTANCE"].astype(str) + " ft · " + d["GAME_DATE"].astype(str)
            for flag, name, color, sym, size in ((0, "Fallado", RED, "x", 6), (1, "Anotado", GREEN, "circle", 7)):
                m = d["SHOT_MADE_FLAG"] == flag
                fig.add_trace(go.Scatter(x=d.loc[m, "LOC_X"], y=d.loc[m, "LOC_Y"], mode="markers", name=name,
                                         marker=dict(color=color, symbol=sym, size=size, opacity=0.7),
                                         text=hov[m], hovertemplate="%{text}<extra></extra>"))
        elif vista.startswith("Mapa"):
            fig.add_trace(go.Histogram2d(
                x=d["LOC_X"], y=d["LOC_Y"], xbins=dict(start=-250, end=250, size=20), ybins=dict(start=-50, end=420, size=20),
                colorscale=[[0, "rgba(11,14,20,0)"], [0.15, "#4c1d95"], [0.5, "#ea580c"], [1, "#fde047"]],
                zsmooth="best", showscale=False, hovertemplate="Tiros: %{z}<extra></extra>"))
        else:
            b = d.assign(BX=np.floor((d["LOC_X"] + 250) / 30), BY=np.floor((d["LOC_Y"] + 50) / 30))
            hb = b.groupby(["BX", "BY"]).agg(N=("SHOT_MADE_FLAG", "size"), FG=("SHOT_MADE_FLAG", "mean")).reset_index()
            min_att = max(2, int(len(d) * 0.004))
            hb = hb[hb["N"] >= min_att]
            hb["X"], hb["Y"] = hb["BX"] * 30 - 250 + 15, hb["BY"] * 30 - 50 + 15
            fig.add_trace(go.Scatter(
                x=hb["X"], y=hb["Y"], mode="markers",
                marker=dict(symbol="hexagon", size=10 + 26 * np.sqrt(hb["N"] / hb["N"].max()), color=hb["FG"] * 100,
                            colorscale="RdYlGn", cmin=25, cmax=65, line=dict(width=0.5, color="#0b0e14"),
                            colorbar=dict(title="FG%", thickness=10, len=0.6)),
                text="Intentos: " + hb["N"].astype(str) + " · FG%: " + (hb["FG"] * 100).round(1).astype(str),
                hovertemplate="%{text}<extra></extra>", showlegend=False))
        for tr in court_traces():
            fig.add_trace(tr)
        fig.update_layout(title=f"{p['PLAYER_NAME']} · {vista}")
        style_fig(fig, 640)
        fig.update_xaxes(range=[-255, 255], visible=False, showgrid=False)
        fig.update_yaxes(range=[-55, 425], visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
        show_plot(fig)

    with cr:
        figz = go.Figure()
        figz.add_bar(name=p["PLAYER_NAME"], x=z["SHOT_ZONE_BASIC"], y=z["FG%"], marker_color=ORANGE,
                     text=z["FG%"].round(1).astype(str), textposition="outside")
        figz.add_bar(name="Liga", x=z["SHOT_ZONE_BASIC"], y=z["LG_FG"], marker_color=BLUE, opacity=0.7)
        figz.update_layout(barmode="group", title="FG% por zona vs. liga")
        figz.update_xaxes(tickangle=-35)
        show_plot(style_fig(figz, 320))

        bins = [0, 5, 10, 15, 20, 25, 30, 100]
        labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
        dd = d.assign(DIST=pd.cut(d["SHOT_DISTANCE"], bins=bins, labels=labels, right=False))
        dist = dd.groupby("DIST", observed=False).agg(FGA=("SHOT_MADE_FLAG", "size"), FG=("SHOT_MADE_FLAG", "mean")).reset_index()
        figd = go.Figure(go.Bar(x=dist["DIST"], y=dist["FG"] * 100, marker_color=GREEN,
                                text="n=" + dist["FGA"].astype(str), textposition="outside"))
        figd.update_layout(title="FG% por distancia (ft)")
        show_plot(style_fig(figd, 300, showlegend=False))

    st.markdown("#### 🧭 Detalle por zona")
    tbl = z.rename(columns={"SHOT_ZONE_BASIC": "Zona", "FG%": "FG%", "Freq%": "Frecuencia %", "LG_FG": "FG% liga"})
    tbl = tbl[["Zona", "FGA", "FGM", "FG%", "FG% liga", "ΔFG%", "Frecuencia %", "PPS", "ΔPPS"]].round(2)
    show_df(tbl)
    st.caption("PPS = puntos por tiro de campo. ΔFG% / ΔPPS comparan al jugador contra el promedio de la liga en cada zona. "
               "Los shot charts incluyen solo tiros de campo (sin libres).")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · LIGA
# ─────────────────────────────────────────────────────────────────────────────
def tab_league():
    view = st.radio("Vista", ["Jugadores: Eficiencia vs Uso", "Equipos: Ataque vs Defensa", "Líderes"], horizontal=True)

    if view.startswith("Jugadores"):
        f1, f2 = st.columns(2)
        max_gp = int(LEAGUE["GP"].max()) if not LEAGUE.empty else 1
        min_gp = f1.slider("Mínimo de partidos", 1, max(max_gp, 2), max(1, int(max_gp * 0.25)))
        min_min = f2.slider("Mínimo de minutos por partido", 0, 36, 15)
        df = LEAGUE[(LEAGUE["GP"] >= min_gp) & (LEAGUE["MIN"] >= min_min)].dropna(subset=["USG_PCT", "TS_CALC"]).copy()
        if df.empty:
            st.info("Ningún jugador cumple los filtros.")
            return
        df["PTS_SIZE"] = df["PTS"].clip(lower=0.5)
        fig = px.scatter(
            df, x="USG_PCT", y="TS_CALC", size="PTS_SIZE", color="PIE", hover_name="PLAYER_NAME",
            hover_data={"TEAM_ABBREVIATION": True, "PTS": ":.1f", "TS_CALC": ":.1f", "USG_PCT": ":.1f",
                        "PIE": ":.1f", "GP": True, "PTS_SIZE": False},
            color_continuous_scale="YlOrRd", size_max=26,
            labels={"USG_PCT": "Usage Rate (%)", "TS_CALC": "True Shooting (%)"},
        )
        fig.add_hline(y=LG["TS"], line_dash="dot", line_color=MUTED, annotation_text="TS% liga")
        fig.add_vline(x=df["USG_PCT"].mean(), line_dash="dot", line_color=MUTED, annotation_text="USG% medio")
        fig.update_layout(title="Eficiencia (TS%) vs Volumen (USG%) · tamaño = puntos · color = PIE")
        show_plot(style_fig(fig, 560))
        st.caption("Arriba a la derecha: anotadores de alto volumen y alta eficiencia (los más valiosos ofensivamente).")

    elif view.startswith("Equipos"):
        if TEAM_TBL.empty:
            st.warning("Sin estadísticas de equipos.")
            return
        t = TEAM_TBL.reset_index(drop=True).copy()
        t["ABBR"] = t["TEAM_ID"].map(ID2ABBR).fillna(t["TEAM_NAME"])
        fig = px.scatter(t, x="OFF_RATING", y="DEF_RATING", text="ABBR", color="NET_RATING", hover_name="TEAM_NAME",
                         color_continuous_scale="RdYlGn", labels={"OFF_RATING": "Offensive Rating", "DEF_RATING": "Defensive Rating"})
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_yaxes(autorange="reversed")
        fig.add_hline(y=t["DEF_RATING"].mean(), line_dash="dot", line_color=MUTED)
        fig.add_vline(x=t["OFF_RATING"].mean(), line_dash="dot", line_color=MUTED)
        fig.update_layout(title="Ataque vs Defensa (eje Y invertido: arriba = mejor defensa)")
        show_plot(style_fig(fig, 580))

        fig2 = px.scatter(t, x="PACE", y="OFF_RATING", text="ABBR", color="TS_PCT", hover_name="TEAM_NAME",
                          color_continuous_scale="YlOrRd", labels={"PACE": "Pace", "OFF_RATING": "Offensive Rating", "TS_PCT": "TS%"})
        fig2.update_traces(textposition="top center", marker=dict(size=12))
        fig2.update_layout(title="Ritmo vs Eficiencia ofensiva · color = TS%")
        show_plot(style_fig(fig2, 480))

    else:
        options = {
            "Puntos": "PTS", "Rebotes": "REB", "Asistencias": "AST", "Robos": "STL", "Tapones": "BLK",
            "True Shooting % (TS%)": "TS_CALC", "Effective FG % (eFG%)": "EFG_CALC",
            "Usage Rate (USG%)": "USG_PCT", "PIE": "PIE", "Net Rating": "NET_RATING",
        }
        f1, f2, f3 = st.columns(3)
        metric = f1.selectbox("Métrica", list(options))
        max_gp = int(LEAGUE["GP"].max()) if not LEAGUE.empty else 1
        min_gp = f2.slider("Mínimo de partidos", 1, max(max_gp, 2), max(1, int(max_gp * 0.25)), key="ld_gp")
        min_min = f3.slider("Mínimo de minutos", 0, 36, 20, key="ld_min")
        col = options[metric]
        df = LEAGUE[(LEAGUE["GP"] >= min_gp) & (LEAGUE["MIN"] >= min_min)].dropna(subset=[col])
        if df.empty:
            st.info("Ningún jugador cumple los filtros.")
            return
        top = df.nlargest(15, col).iloc[::-1]
        fig = px.bar(top, x=col, y="PLAYER_NAME", orientation="h", color=col, color_continuous_scale="YlOrRd",
                     text=top[col].round(1), hover_data=["TEAM_ABBREVIATION", "GP"],
                     labels={col: metric, "PLAYER_NAME": ""})
        fig.update_layout(title=f"Top 15 · {metric}", coloraxis_showscale=False)
        show_plot(style_fig(fig, 560))


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔥 Predictor & H2H", "⚔️ Comparador", "⭐ Player Props", "🎯 Shot Charts", "🌐 Liga"])
with tabs[0]:
    tab_matchup()
with tabs[1]:
    tab_compare()
with tabs[2]:
    tab_props()
with tabs[3]:
    tab_shots()
with tabs[4]:
    tab_league()

st.write("")
st.caption("Datos: stats.nba.com vía nba_api · Las métricas TS% y eFG% se calculan a partir de los registros de partidos de cada jugador.")