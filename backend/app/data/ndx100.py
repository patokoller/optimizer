"""
app/data/ndx100.py
──────────────────────────────────────────────────────────────────────────────
NASDAQ-100 constituent list (NDX).

Hardcoded for reliability — AV INDEX_CONSTITUENTS requires premium tier.
Update quarterly or when index rebalancing occurs (typically March/June/Sep/Dec).
Last verified: May 2026.

Excludes: dual-class shares listed separately (GOOG vs GOOGL — keep GOOGL only),
          non-equity instruments, and tickers known to break Alpaca's IEX feed.
"""

NDX100_TICKERS: list[str] = [
    "AAPL",  "MSFT",  "NVDA",  "AMZN",  "META",  "GOOGL", "TSLA",  "AVGO",
    "COST",  "NFLX",  "ASML",  "AMD",   "TMUS",  "LIN",   "CSCO",  "PEP",
    "ADBE",  "TXN",   "QCOM",  "AMGN",  "INTU",  "ISRG",  "CMCSA", "AMAT",
    "MU",    "HON",   "BKNG",  "VRTX",  "REGN",  "ADI",   "LRCX",  "PANW",
    "SBUX",  "MELI",  "KLAC",  "SNPS",  "CDNS",  "MDLZ",  "ORLY",  "ABNB",
    "PDD",   "CEG",   "CSX",   "CTAS",  "PYPL",  "FTNT",  "MRVL",  "WDAY",
    "ROP",   "KDP",   "CPRT",  "ROST",  "ODFL",  "PAYX",  "MNST",  "AEP",
    "FAST",  "NXPI",  "DXCM",  "IDXX",  "VRSK",  "EXC",   "LULU",  "GEHC",
    "MCHP",  "CTSH",  "TTD",   "TEAM",  "BIIB",  "ON",    "PCAR",  "CDW",
    "ANSS",  "DDOG",  "ZS",    "CRWD",  "MRNA",  "ILMN",  "ARM",   "DASH",
    "FANG",  "CSGP",  "CCEP",  "GFS",   "APP",   "PLTR",  "AZN",   "CHTR",
    "SMCI",  "DLTR",  "WBD",   "SIRI",  "FSLR",  "ENPH",  "ZM",    "ALGN",
    "NOW",   "FICO",
]

# Sectors for each ticker (used for concentration analysis in Discovery UI)
# Source: GICS classification, approximate
NDX100_SECTORS: dict[str, str] = {
    "AAPL": "Technology",    "MSFT": "Technology",    "NVDA": "Technology",
    "AMZN": "Consumer Disc", "META": "Communication", "GOOGL": "Communication",
    "TSLA": "Consumer Disc", "AVGO": "Technology",    "COST": "Consumer Staples",
    "NFLX": "Communication", "ASML": "Technology",    "AMD":  "Technology",
    "TMUS": "Communication", "LIN":  "Materials",     "CSCO": "Technology",
    "PEP":  "Consumer Staples", "ADBE": "Technology", "TXN":  "Technology",
    "QCOM": "Technology",    "AMGN": "Healthcare",    "INTU": "Technology",
    "ISRG": "Healthcare",    "CMCSA": "Communication","AMAT": "Technology",
    "MU":   "Technology",    "HON":  "Industrials",   "BKNG": "Consumer Disc",
    "VRTX": "Healthcare",    "REGN": "Healthcare",    "ADI":  "Technology",
    "LRCX": "Technology",    "PANW": "Technology",    "SBUX": "Consumer Disc",
    "MELI": "Consumer Disc", "KLAC": "Technology",    "SNPS": "Technology",
    "CDNS": "Technology",    "MDLZ": "Consumer Staples","ORLY":"Consumer Disc",
    "ABNB": "Consumer Disc", "PDD":  "Consumer Disc", "CEG":  "Utilities",
    "CSX":  "Industrials",   "CTAS": "Industrials",   "PYPL": "Financials",
    "FTNT": "Technology",    "MRVL": "Technology",    "WDAY": "Technology",
    "ROP":  "Technology",    "KDP":  "Consumer Staples","CPRT":"Industrials",
    "ROST": "Consumer Disc", "ODFL": "Industrials",   "PAYX": "Industrials",
    "MNST": "Consumer Staples","AEP": "Utilities",    "FAST": "Industrials",
    "NXPI": "Technology",    "DXCM": "Healthcare",    "IDXX": "Healthcare",
    "VRSK": "Industrials",   "EXC":  "Utilities",     "LULU": "Consumer Disc",
    "GEHC": "Healthcare",    "MCHP": "Technology",    "CTSH": "Technology",
    "TTD":  "Technology",    "TEAM": "Technology",    "BIIB": "Healthcare",
    "ON":   "Technology",    "PCAR": "Industrials",   "CDW":  "Technology",
    "ANSS": "Technology",    "DDOG": "Technology",    "ZS":   "Technology",
    "CRWD": "Technology",    "MRNA": "Healthcare",    "ILMN": "Healthcare",
    "ARM":  "Technology",    "DASH": "Consumer Disc", "FANG": "Energy",
    "CSGP": "Real Estate",   "CCEP": "Consumer Staples","GFS": "Technology",
    "APP":  "Technology",    "PLTR": "Technology",    "AZN":  "Healthcare",
    "CHTR": "Communication", "SMCI": "Technology",    "DLTR": "Consumer Disc",
    "WBD":  "Communication", "SIRI": "Communication", "FSLR": "Energy",
    "ENPH": "Energy",        "ZM":   "Technology",    "ALGN": "Healthcare",
    "NOW":  "Technology",    "FICO": "Technology",
}


def get_ndx100_tickers() -> list[str]:
    """Return the current NDX-100 ticker universe."""
    return NDX100_TICKERS.copy()


def get_sector(ticker: str) -> str:
    """Return the GICS sector for a ticker, or 'Unknown'."""
    return NDX100_SECTORS.get(ticker, "Unknown")
