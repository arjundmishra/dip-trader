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

DATA_DIR = Path(os.getenv("STATE_FILE_PATH", "/data/state.json")).parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("  Robinhood One-Time Login Setup")
print("=" * 50)

username = os.environ.get("RH_USERNAME") or input("Robinhood email: ").strip()
password = os.environ.get("RH_PASSWORD") or input("Robinhood password: ").strip()

print("\nIf you use MFA, enter your code now (or press Enter to skip):")
mfa_code = input("MFA code (leave blank if none): ").strip() or None

pickle_path = str(DATA_DIR / "rh_session")

kwargs = {
    "username": username,
    "password": password,
    "store_session": True,
    "pickle_name": pickle_path,
}
if mfa_code:
    kwargs["mfa_code"] = mfa_code

try:
    rh.login(**kwargs)
    print(f"\n✓ Login successful! Session cached at: {pickle_path}.pickle")
    print("  You can now run main.py without interactive prompts.")

    profile = rh.load_account_profile()
    bp = profile.get("buying_power", "unknown")
    print(f"  Buying power: ${bp}")
except Exception as e:
    print(f"\n✗ Login failed: {e}")
    raise
