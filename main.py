#!/usr/bin/env python3
"""
Robinhood Buy-the-Dip Trader
Cloud-hosted autonomous trading agent powered by Claude Opus 4.6.
Runs daily at market open, researches top stocks, buys dips aggressively.
"""

import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path

import anthropic
import yfinance as yf
import robin_stocks.robinhood as rh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Persistent state on Railway volume (falls back to local)
STATE_FILE = Path(os.getenv("STATE_FILE_PATH", "/data/state.json"))

CORE_WATCHLIST = ["NVDA", "PLTR", "GOOGL", "NVO", "ABBV", "RARE"]

# Spend tiers keyed by minimum dip %
TIERS = [
    (20, 500,  "extreme"),
    (15, 350,  "major"),
    (10, 200,  "large"),
    (5,  100,  "medium"),
    (3,   50,  "small"),
]

MA_WINDOW = 200
TRAILING_STOP_PCT = 0.25


# ── State helpers ────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"peaks": {}, "total_deployed": 0.0, "daily_log": [], "last_watchlist": CORE_WATCHLIST}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Market data ──────────────────────────────────────────────────────────────

def get_20day_high(symbol: str) -> float | None:
    """Highest closing price over the last 20 trading days."""
    try:
        hist = yf.Ticker(symbol).history(period="1mo")
        return float(hist["Close"].max()) if not hist.empty else None
    except Exception as e:
        log.warning(f"{symbol}: 20d-high lookup failed — {e}")
        return None


def get_200day_ma(symbol: str) -> float | None:
    try:
        hist = yf.Ticker(symbol).history(period="1y")
        if hist.empty:
            return None
        closes = hist["Close"].tail(MA_WINDOW)
        return float(closes.mean()) if len(closes) else None
    except Exception as e:
        log.warning(f"{symbol}: 200d-MA lookup failed — {e}")
        return None


def get_current_price(symbol: str) -> float | None:
    try:
        info = yf.Ticker(symbol).fast_info
        price = getattr(info, "last_price", None)
        if price and price > 0:
            return float(price)
        hist = yf.Ticker(symbol).history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception as e:
        log.warning(f"{symbol}: price lookup failed — {e}")
        return None


def get_market_context() -> str:
    try:
        hist = yf.Ticker("SPY").history(period="2d")
        if len(hist) >= 2:
            chg = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
            if chg < -1:
                return f"BROAD SELLOFF: S&P 500 down {chg:.1f}% today — fear-driven, lean in hard"
            elif chg > 1:
                return f"S&P 500 up {chg:.1f}% today — rally day"
            return f"S&P 500 roughly flat {chg:.1f}% today"
    except Exception:
        pass
    return "Market context unavailable"


# ── Claude market research ───────────────────────────────────────────────────

def research_stocks(client: anthropic.Anthropic) -> list[dict]:
    """Five-voice investment committee: Marco, Fabian, Cara, Theo, Rico."""
    log.info("Running five-voice investment committee research...")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        system="""You are a five-voice investment committee running a systematic dip-buying
strategy. You are not one analyst — you are five, and they disagree with
each other. Every stock recommendation must survive a structured debate
between all five voices before it earns a buy signal. Weak ideas get
killed by the committee. Only high-conviction, multi-factor-supported
setups make it through.

Your five internal voices are:

MARCO (Macro & Regime Analyst) — thinks top-down. Asks: what is the market
regime, where is the Fed, what is the yield curve doing, is this a
risk-on or risk-off environment? He vetos everything in a confirmed
downtrend.

FABIAN (Factor Model Analyst) — thinks like AQR and Fama-French. Asks:
does this stock score well on the six proven factors — market beta,
size, value, profitability, investment quality, and momentum? He only
endorses names where at least four of six factors are favorable.

CARA (Catalyst & Event Analyst) — thinks like a biotech desk and an
event-driven hedge fund. Asks: WHY did this stock dip? Is it a macro
pullback (opportunity) or a company-specific failure (avoid)? She
hard-vetoes any dip caused by earnings misses, FDA failures, fraud, or
guidance cuts. She also flags upcoming binary events as risk.

THEO (Technical & Behavioral Analyst) — thinks like a quant with a
psychology PhD. Asks: is the price structure showing capitulation and
reversal, or distribution and continuation? He reads RSI, volume
profile, distance from key MAs, and whether institutional money is
accumulating or distributing. He flags knife-catches.

RICO (Risk & Portfolio Construction Analyst) — thinks like a risk
manager at a multi-strategy fund. Asks: what does adding this position
do to portfolio concentration, factor correlation, and maximum
drawdown? He enforces position limits and kills ideas that create
dangerous overlap. He applies Kelly Criterion logic to size positions.

---

## STEP 1 — MARCO: REGIME ASSESSMENT (run this before anything else)

Search for current data on:
- SPY price vs its 50-day and 200-day moving averages
- VIX current level and 20-day trend
- US 2Y/10Y yield curve shape (inverted/flat/normal)
- Fed funds futures: are cuts or hikes priced in?
- Credit spreads: HYG vs LQD spread (risk appetite indicator)

Classify the current regime as one of:
- GOLDILOCKS: SPY above 200MA, VIX < 18, curve normal, spreads tight
  -> full sizing, lean in on dips
- CAUTIOUS: SPY above 200MA but VIX 18-25, some spread widening
  -> 75% normal sizing, be selective
- DEFENSIVE: SPY below 200MA OR VIX > 25
  -> 50% sizing maximum, only highest conviction names
- CRISIS: VIX > 35 OR credit spreads blowing out
  -> return empty array, cash is the position, do not buy anything

State the regime classification and a one-sentence rationale at the top
of your response. This overrides everything else.

If regime is CRISIS: stop here. Return:
{"regime": "CRISIS", "explanation": "[reason]", "picks": []}

---

## STEP 2 — FABIAN: SIX-FACTOR SCREEN

For each candidate stock, score it on the six proven Fama-French +
momentum factors. Use web search to find recent data.

Factor 1 — MARKET (Beta): beta < 1.5 preferred for dip-buying (high
beta names move too fast to catch cleanly). Score: LOW/MED/HIGH beta.

Factor 2 — SIZE: Small/mid caps (<$10B market cap) have higher expected
returns but higher volatility. For a small account: flag if market cap
< $500M as "micro-cap risk — spread and liquidity concern."

Factor 3 — VALUE (P/E, P/B, EV/EBITDA vs sector): below-sector-median
multiples = value tailwind. Above 3x sector median = valuation headwind.
Score: CHEAP / FAIR / EXPENSIVE / EXTREME.

Factor 4 — PROFITABILITY (Gross profit margin, ROE, operating cash flow):
is the company actually generating cash? Negative operating cash flow +
high debt = structural short, not a dip to buy. Score: STRONG / OK /
WEAK / NEGATIVE.

Factor 5 — INVESTMENT QUALITY (D/E ratio, capex discipline, buybacks vs
dilution): is management allocating capital well? Recent share dilution
is a red flag. Score: DISCIPLINED / NEUTRAL / CONCERNING / DILUTIVE.

Factor 6 — MOMENTUM (12-1 month price momentum, excluding last month):
is the stock in an intermediate uptrend despite the recent dip?
Score: POSITIVE / NEUTRAL / NEGATIVE.

Factor score summary: count favorable factors.
5-6 favorable = STRONG -> size_modifier 1.0-1.5
3-4 favorable = MODERATE -> size_modifier 0.75
1-2 favorable = WEAK -> size_modifier 0.25-0.5
0 favorable = AVOID -> size_modifier 0.0

---

## STEP 3 — CARA: CATALYST CLASSIFICATION

For each candidate, web search last 72 hours of news. Classify the dip:

MACRO_DIP: broad market sold off, fundamentals unchanged -> GREEN LIGHT.
SECTOR_ROTATION: money rotating out of sector -> YELLOW, reduce size_modifier by 0.25.
VALUATION_RESET: re-rated lower without bad news -> YELLOW, check valuation.
EARNINGS_MISS: reported below expectations, cut guidance -> RED. size_modifier = 0.
GUIDANCE_CUT: management lowered forward expectations -> RED. size_modifier = 0.
FDA_FAILURE / REGULATORY_ADVERSE: binary event went wrong -> RED. size_modifier = 0.
FRAUD_ACCOUNTING: SEC inquiry, restatement, auditor resignation -> BLACK FLAG. size_modifier = 0.
BINARY_EVENT_UPCOMING: earnings within 14 days or FDA within 30 days -> ORANGE. Cut size_modifier by 0.5.
UNKNOWN: cannot determine why stock dipped -> size_modifier 0.25 maximum.

---

## STEP 4 — THEO: TECHNICAL STRUCTURE ASSESSMENT

RSI (14-day): below 30 = oversold / 30-50 = neutral / above 50 = not a real dip.
Distance from 200-day MA: buying more than 20% below the 200MA flags as EXTENDED_BREAKDOWN.
Ideal dip: 3-15% below a rising 20-day MA, while above the 200-day MA.
Volume: high-volume dip on no news = distribution (bad). Low-volume dip on macro fear = opportunity.
Support: buying at known support level > buying in midair.

Theo's verdict: ACCUMULATE / WAIT / AVOID_TECHNICAL.
ACCUMULATE: no change / WAIT: -0.25 / AVOID_TECHNICAL: size_modifier = 0.

---

## STEP 5 — RICO: PORTFOLIO CONSTRUCTION AND KELLY SIZING

Concentration: no single name > 25% of portfolio, no single sector > 40%.
No more than 2 names in the same sector per session.
Correlation: if two names have >0.7 correlation, keep only highest conviction.
Kelly: use half-Kelly for new positions. Scale toward full Kelly only for names
with proven positive expectancy in account history.
Session cap: never deploy more than 40% of available cash in one session.

Rico's final size_modifier: minimum of all modifiers from Fabian, Cara, Theo,
then applies concentration/correlation/Kelly adjustments.

---

## STEP 6 — COMMITTEE VOTE AND OUTPUT

conviction_score = Marco 20% + Fabian 25% + Cara 25% + Theo 15% + Rico 15%
Only include stocks with conviction_score >= 0.55 in picks array.

Return a JSON object with this exact structure — no other text, no markdown:

{
  "regime": "GOLDILOCKS|CAUTIOUS|DEFENSIVE|CRISIS",
  "regime_rationale": "one sentence",
  "spy_vs_200ma": "above|below",
  "vix_level": 18.5,
  "session_max_deploy_pct": 0.40,
  "picks": [
    {
      "symbol": "AAPL",
      "conviction_score": 0.78,
      "size_modifier": 0.90,
      "dip_classification": "MACRO_DIP",
      "factor_score": "strong",
      "factors_favorable": 5,
      "bull_case": "one sentence — the specific reason to buy today",
      "bear_case": "one sentence — the strongest argument against",
      "invalidation": "one sentence — what price/event proves the thesis wrong",
      "binary_event_risk": "none|earnings_14d|fda_30d|other",
      "theo_verdict": "ACCUMULATE|WAIT|AVOID_TECHNICAL",
      "rico_flags": "none|concentration|correlation|kelly_reduced",
      "committee_votes": {
        "marco": 0.8, "fabian": 0.7, "cara": 0.9, "theo": 0.7, "rico": 0.8
      }
    }
  ],
  "committee_rejected": [
    {"symbol": "XYZ", "rejection_reason": "Cara: EARNINGS_MISS", "size_modifier": 0.0}
  ],
  "session_notes": "one paragraph of market color"
}

---

## HARD RULES — NEVER VIOLATE

1. CRISIS regime = empty picks array. No exceptions.
2. EARNINGS_MISS, GUIDANCE_CUT, FDA_FAILURE, FRAUD_ACCOUNTING = size_modifier 0.0.
3. Never deploy more than session_max_deploy_pct of buying power in one session.
4. bear_case and invalidation fields are MANDATORY for every pick.
5. When in doubt, the committee does not buy. Cash is a position.

---

## CORE WATCHLIST (always research, but apply all filters — can score 0)

NVDA, PLTR, GOOGL, NVO, ABBV, RARE

Expand universe with web search. Prioritize names with upcoming positive
catalysts, high factor scores pulling back on macro fear, spinoffs, activist
involvement, M&A targets, and sector leaders showing relative strength.

Sectors to monitor: defense tech, energy transition infrastructure,
pharma with near-term catalysts, regional banks, industrial automation.""",
        messages=[{
            "role": "user",
            "content": f"""Today is {datetime.now().strftime('%A %B %d, %Y')}.

Run the full five-voice committee analysis. Use web search to gather current
macro data (SPY, VIX, yield curve), and research each stock candidate.

Return ONLY the JSON object as specified. No markdown, no extra text."""
        }]
    )

    for block in response.content:
        if block.type == "text":
            match = re.search(r'\{[\s\S]*\}', block.text)
            if match:
                try:
                    data = json.loads(match.group())
                    regime = data.get("regime", "GOLDILOCKS")
                    log.info(f"Regime: {regime} — {data.get('regime_rationale', '')}")
                    log.info(f"VIX: {data.get('vix_level', 'N/A')}  SPY vs 200MA: {data.get('spy_vs_200ma', 'N/A')}")

                    if regime == "CRISIS":
                        log.warning("CRISIS regime — committee says cash is the position. No buys today.")
                        return []

                    picks = data.get("picks", [])
                    rejected = data.get("committee_rejected", [])
                    session_notes = data.get("session_notes", "")
                    if session_notes:
                        log.info(f"Committee notes: {session_notes}")
                    log.info(f"Committee: {len(picks)} approved, {len(rejected)} rejected")
                    for r in rejected:
                        log.info(f"  REJECTED {r.get('symbol','?')}: {r.get('rejection_reason','')}")
                    return picks
                except json.JSONDecodeError as e:
                    log.warning(f"JSON parse error: {e}")

    log.warning("Research parsing failed — falling back to core watchlist")
    return [{"symbol": s, "size_modifier": 1.0, "conviction_score": 0.6,
             "dip_classification": "UNKNOWN", "factor_score": "moderate",
             "bull_case": "Core watchlist fallback", "bear_case": "N/A",
             "invalidation": "N/A"} for s in CORE_WATCHLIST]


# ── Robinhood auth ───────────────────────────────────────────────────────────

def login_robinhood():
    username = os.environ["RH_USERNAME"]
    password = os.environ["RH_PASSWORD"]

    # robin_stocks resolves the session file to:
    #   <pickle_path>/("robinhood" + pickle_name + ".pickle")
    # so pickle_path="/data" + pickle_name="rh_session" -> /data/robinhoodrh_session.pickle
    PICKLE_DIR = "/data"
    PICKLE_NAME = "rh_session"
    pickle_file = Path(PICKLE_DIR) / f"robinhood{PICKLE_NAME}.pickle"

    # Seed pickle from env var on first run (before the file exists on the volume)
    session_b64 = os.environ.get("RH_SESSION_B64")
    if session_b64 and not pickle_file.exists():
        import base64
        pickle_file.parent.mkdir(parents=True, exist_ok=True)
        pickle_file.write_bytes(base64.b64decode(session_b64))
        log.info("Seeded session from RH_SESSION_B64")

    kwargs = {
        "username": username,
        "password": password,
        "store_session": True,
        "pickle_path": PICKLE_DIR,
        "pickle_name": PICKLE_NAME,
    }

    # TOTP fallback — only used if RH_TOTP_SECRET is still set
    totp_secret = os.environ.get("RH_TOTP_SECRET")
    if totp_secret:
        import pyotp
        kwargs["mfa_code"] = pyotp.TOTP(totp_secret).now()
        log.info("Using TOTP MFA")

    rh.login(**kwargs)
    log.info("Robinhood login successful")


def get_buying_power() -> float:
    """Returns cash-only balance — never margin."""
    try:
        profile = rh.load_account_profile()
        # Use 'cash' not 'buying_power' — buying_power includes margin
        cash = float(profile.get("cash", 0) or 0)
        return cash
    except Exception as e:
        log.warning(f"Could not fetch cash balance: {e}")
        return 0.0


def place_order(symbol: str, amount_usd: float) -> bool:
    try:
        result = rh.order_buy_fractional_by_price(
            symbol,
            amount_usd,
            timeInForce="gfd",
            extendedHours=False,
        )
        if result and result.get("id"):
            log.info(f"  ✓ Order placed: {symbol} ${amount_usd:.2f} (id={result['id'][:8]}...)")
            return True
        log.warning(f"  ✗ Order rejected for {symbol}: {result}")
        return False
    except Exception as e:
        log.error(f"  ✗ Order error for {symbol}: {e}")
        return False


def get_holdings() -> dict:
    try:
        return rh.build_holdings() or {}
    except Exception as e:
        log.warning(f"Could not fetch holdings: {e}")
        return {}


def place_sell_all(symbol: str, quantity: float) -> bool:
    try:
        result = rh.order_sell_fractional_by_quantity(
            symbol, round(quantity, 6), timeInForce="gfd", extendedHours=False)
        if result and result.get("id"):
            log.info(f"  ✓ SOLD {symbol} {quantity:.6f} sh (id={result['id'][:8]}...)")
            return True
        log.warning(f"  ✗ Sell rejected for {symbol}: {result}")
        return False
    except Exception as e:
        log.error(f"  ✗ Sell error for {symbol}: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 70)
    log.info(f"  BUY-THE-DIP TRADER  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    login_robinhood()

    state = load_state()
    watchlist = research_stocks(claude)
    state["last_watchlist"] = [p["symbol"] if isinstance(p, dict) else p for p in watchlist]

    market_ctx = get_market_context()
    log.info(f"Market context: {market_ctx}")

    buying_power = get_buying_power()
    log.info(f"Cash available (no margin): ${buying_power:.2f}\n")

    sells_made = []
    holdings = get_holdings()
    for symbol, h in holdings.items():
        try:
            price = float(h.get("price", 0) or 0)
            qty = float(h.get("quantity", 0) or 0)
            if price <= 0 or qty <= 0:
                continue
            peak = max(state["peaks"].get(symbol, 0.0), price)
            state["peaks"][symbol] = peak
            drawdown = (peak - price) / peak if peak > 0 else 0.0
            if drawdown >= TRAILING_STOP_PCT:
                log.info(f"{symbol:6s} down {drawdown*100:.1f}% from peak -> TRAILING STOP SELL ALL")
                if place_sell_all(symbol, qty):
                    sells_made.append({"symbol": symbol, "price": round(price, 4),
                                       "peak": round(peak, 4), "drawdown_pct": round(drawdown * 100, 2)})
                    state["peaks"].pop(symbol, None)
        except Exception as e:
            log.error(f"{symbol}: sell-check error — {e}")

    today_spent = 0.0
    orders_placed = []
    skipped = []

    session_cap = buying_power * 0.40

    for pick in watchlist:
        symbol = pick["symbol"] if isinstance(pick, dict) else str(pick)
        size_modifier = float(pick.get("size_modifier", 1.0)) if isinstance(pick, dict) else 1.0
        conviction = float(pick.get("conviction_score", 0.6)) if isinstance(pick, dict) else 0.6
        bull_case = pick.get("bull_case", "") if isinstance(pick, dict) else ""
        bear_case = pick.get("bear_case", "") if isinstance(pick, dict) else ""
        invalidation = pick.get("invalidation", "") if isinstance(pick, dict) else ""
        dip_class = pick.get("dip_classification", "") if isinstance(pick, dict) else ""

        if size_modifier == 0:
            skipped.append((symbol, "analyst committee voted AVOID"))
            continue

        if today_spent >= session_cap:
            skipped.append((symbol, f"session cap ${session_cap:.0f} (40% of cash) reached"))
            continue

        try:
            current_price = get_current_price(symbol)
            high_20d = get_20day_high(symbol)

            if not current_price or not high_20d or high_20d <= 0:
                skipped.append((symbol, "price data unavailable"))
                continue

            # Update high water mark
            if symbol not in state["peaks"] or current_price > state["peaks"][symbol]:
                state["peaks"][symbol] = current_price

            dip_pct = ((high_20d - current_price) / high_20d) * 100

            if dip_pct < 3:
                skipped.append((symbol, f"dip {dip_pct:.1f}% < 3% threshold"))
                continue

            ma200 = get_200day_ma(symbol)
            if ma200 and current_price < ma200:
                skipped.append((symbol, f"below 200-day MA (${current_price:.2f} < ${ma200:.2f}) — downtrend"))
                continue

            # Assign tier
            spend, tier = 0.0, "skip"
            for min_dip, amount, name in TIERS:
                if dip_pct >= min_dip:
                    spend, tier = float(amount), name
                    break

            # Boost +50% on broad market selloff
            boosted = False
            if "SELLOFF" in market_ctx.upper() or "down" in market_ctx.lower():
                spend *= 1.5
                boosted = True

            # Apply committee size_modifier
            spend = round(spend * size_modifier, 2)
            if spend < 1.0:
                skipped.append((symbol, f"size_modifier {size_modifier:.2f} reduced spend below $1"))
                continue

            if spend > buying_power - today_spent:
                skipped.append((symbol, f"insufficient buying power (need ${spend:.0f})"))
                continue

            log.info(
                f"{symbol:6s}  price=${current_price:8.2f}  20d-high=${high_20d:8.2f}"
                f"  dip={dip_pct:5.1f}%  conviction={conviction:.2f}  -> BUY ${spend:.0f}"
                f" [{tier}{'★' if boosted else ''} x{size_modifier:.2f}]"
            )
            if bull_case:
                log.info(f"         BULL: {bull_case}")

            if place_order(symbol, spend):
                today_spent += spend
                shares = spend / current_price
                record = {
                    "date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "high_20d": round(high_20d, 4),
                    "current_price": round(current_price, 4),
                    "dip_pct": round(dip_pct, 2),
                    "tier": tier + ("_boosted" if boosted else ""),
                    "conviction_score": round(conviction, 3),
                    "size_modifier": round(size_modifier, 3),
                    "dip_classification": dip_class,
                    "amount_spent": round(spend, 2),
                    "shares_bought": round(shares, 6),
                    "bull_case": bull_case,
                    "bear_case": bear_case,
                    "invalidation": invalidation,
                }
                orders_placed.append(record)
                state["daily_log"].append(record)
                state["total_deployed"] = state.get("total_deployed", 0) + spend

        except Exception as e:
            log.error(f"{symbol}: unexpected error — {e}")
            skipped.append((symbol, f"error: {e}"))

    save_state(state)

    # ── Report ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  DAILY ALPHA REPORT")
    log.info("=" * 70)
    log.info(f"  {market_ctx}")
    log.info(f"  Stocks scanned: {len(watchlist)}")
    log.info(f"\n  TRAILING-STOP SELLS ({len(sells_made)}):")
    for s in sells_made:
        log.info(f"    {s['symbol']:6s}  down {s['drawdown_pct']:.1f}% from peak ${s['peak']:.2f} -> sold @ ${s['price']:.2f}")
    log.info(f"\n  ORDERS PLACED ({len(orders_placed)}):")
    for o in orders_placed:
        log.info(
            f"    {o['symbol']:6s}  dip={o['dip_pct']:5.1f}%  tier={o['tier']:20s}"
            f"  conviction={o.get('conviction_score', 0):.2f}  x{o.get('size_modifier', 1):.2f}"
            f"  ${o['amount_spent']:6.0f}  {o['shares_bought']:.5f} sh @ ${o['current_price']:.2f}"
        )
        if o.get("bull_case"):
            log.info(f"           BULL: {o['bull_case']}")
        if o.get("bear_case"):
            log.info(f"           BEAR: {o['bear_case']}")
        if o.get("invalidation"):
            log.info(f"           KILL: {o['invalidation']}")
    log.info(f"\n  SKIPPED ({len(skipped)}):")
    for sym, reason in skipped:
        log.info(f"    {sym:6s}  {reason}")
    log.info(f"\n  Spent today:            ${today_spent:.2f}")
    log.info(f"  Total deployed ever:    ${state['total_deployed']:.2f}")
    log.info(f"  Remaining cash:         ${buying_power - today_spent:.2f}")
    log.info("=" * 70)


if __name__ == "__main__":
    run()
