#!/usr/bin/env python3
"""
One-time interactive Robinhood login setup.
Run this ONCE on Railway (or locally) to cache your session token.
After this, main.py runs silently without needing interactive MFA.

Usage:
    python setup_login.py
"""

import os
from pathlib import Path
import robin_stocks.robinhood as rh

PICKLE_NAME = "rh_session"
# robin_stocks always saves to ~/.tokens/robinhooddata/<name>.pickle
PICKLE_FILE = Path.home() / ".tokens" / "robinhooddata" / f"{PICKLE_NAME}.pickle"

print("=" * 50)
print("  Robinhood One-Time Login Setup")
print("=" * 50)

username = os.environ.get("RH_USERNAME") or input("Robinhood email: ").strip()
password = os.environ.get("RH_PASSWORD") or input("Robinhood password: ").strip()

print("\nIf you use MFA, enter your code now (or press Enter to skip):")
mfa_code = input("MFA code (leave blank if none): ").strip() or None

kwargs = {
    "username": username,
    "password": password,
    "store_session": True,
    "pickle_name": PICKLE_NAME,
}
if mfa_code:
    kwargs["mfa_code"] = mfa_code

try:
    rh.login(**kwargs)
    print(f"\n✓ Login successful!")

    profile = rh.load_account_profile()
    bp = profile.get("buying_power", "unknown")
    print(f"  Buying power: ${bp}")

    import base64
    if PICKLE_FILE.exists():
        b64 = base64.b64encode(PICKLE_FILE.read_bytes()).decode()
        print("\n" + "=" * 50)
        print("  COPY THIS TO RAILWAY as RH_SESSION_B64:")
        print("=" * 50)
        print(b64)
        print("=" * 50)
        print("  Railway seeds from this on first run, then auto-refreshes daily.")
    else:
        print(f"  Warning: pickle not found at {PICKLE_FILE}")
except Exception as e:
    print(f"\n✗ Login failed: {e}")
    raise
