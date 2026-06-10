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

def research_stocks(client: anthropic.Anthropic) -> list[str]:
    """Use Claude Opus with web search to identify today's top stock picks."""
    log.info("Running Claude market research...")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        system="""You are an aggressive, alpha-seeking stock analyst. Your goal is to
maximise returns and beat the market by the widest margin possible.
Identify the highest-conviction dip-buying opportunities across all sectors today.
Focus on: AI/tech, pharma/biotech, semiconductors, defense tech, fintech, energy tech.
Quality filter: strong earnings growth, positive analyst sentiment, no fraud/existential risk.
Always include the core list: NVDA, PLTR, GOOGL, NVO, ABBV, RARE.""",
        messages=[{
            "role": "user",
            "content": f"""Today is {datetime.now().strftime('%A %B %d, %Y')}.

Search for the best current stock opportunities. Run these searches:
1. "top AI stocks analyst buy rating 2026 outperform"
2. "best pharma biotech stocks to buy on dip 2026"
3. "analyst upgrades stocks this week strong buy"
4. "best semiconductor stocks buy 2026 price target"
5. "high growth stocks oversold buying opportunity 2026"

After researching, return ONLY a JSON array of 12-18 US stock tickers.
Rules:
- Always include: NVDA, PLTR, GOOGL, NVO, ABBV, RARE
- Only stocks on US exchanges tradable on Robinhood
- Exclude stocks with pending fraud investigations, delisting risk, or recent guidance cuts
- Prioritise stocks with strong upcoming catalysts (earnings, FDA decisions, product launches)

Return ONLY the JSON array. Example: ["NVDA", "PLTR", "GOOGL", "AMD", "LLY"]"""
        }]
    )

    for block in response.content:
        if block.type == "text":
            match = re.search(r'\[[\s\S]*?\]', block.text)
            if match:
                try:
                    tickers = json.loads(match.group())
                    valid = [str(t).upper().strip() for t in tickers if isinstance(t, str) and 1 <= len(t) <= 6]
                    for t in CORE_WATCHLIST:
                        if t not in valid:
                            valid.append(t)
                    log.info(f"Research complete — {len(valid)} stocks: {valid}")
                    return valid
                except json.JSONDecodeError:
                    pass

    log.warning("Research parsing failed — using core watchlist")
    return CORE_WATCHLIST


# ── Robinhood auth ───────────────────────────────────────────────────────────

def login_robinhood():
    username = os.environ["RH_USERNAME"]
    password = os.environ["RH_PASSWORD"]

    pickle_path = Path("/data/rh_session")
    pickle_file = Path("/data/rh_session.pickle")

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
        "pickle_name": "rh_session",
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
    state["last_watchlist"] = watchlist

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

    for symbol in watchlist:
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

            if spend > buying_power - today_spent:
                skipped.append((symbol, f"insufficient buying power (need ${spend:.0f})"))
                continue

            log.info(
                f"{symbol:6s}  price=${current_price:8.2f}  20d-high=${high_20d:8.2f}"
                f"  dip={dip_pct:5.1f}%  -> BUY ${spend:.0f} [{tier}{'★' if boosted else ''}]"
            )

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
                    "amount_spent": round(spend, 2),
                    "shares_bought": round(shares, 6),
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
            f"    {o['symbol']:6s}  dip={o['dip_pct']:5.1f}%  tier={o['tier']:22s}"
            f"  ${o['amount_spent']:6.0f}  {o['shares_bought']:.5f} shares @ ${o['current_price']:.2f}"
        )
    log.info(f"\n  SKIPPED ({len(skipped)}):")
    for sym, reason in skipped:
        log.info(f"    {sym:6s}  {reason}")
    log.info(f"\n  Spent today:            ${today_spent:.2f}")
    log.info(f"  Total deployed ever:    ${state['total_deployed']:.2f}")
    log.info(f"  Remaining cash:         ${buying_power - today_spent:.2f}")
    log.info("=" * 70)


if __name__ == "__main__":
    run()
