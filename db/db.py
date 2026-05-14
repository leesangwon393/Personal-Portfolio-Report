from __future__ import annotations

import os
import re
import time
import math
import hashlib
import sqlite3
import argparse
import requests
import sys
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from dateutil import parser  # pubdate 정규화용

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from secrets_config import get_openai_api_key

# ===== (요약용) OpenAI =====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===== 전역 경로/파일 (런타임에 reset_base로 변경 가능) =====
BASE = os.environ.get("NEWS_DB_BASE", os.path.abspath("./news_db"))
DATA_DIR = os.path.join(BASE, "data")
DB_DIR   = os.path.join(BASE, "db")
DB_PATH  = os.path.join(DB_DIR, "news.db")

CSV_PATH           = os.path.join(DATA_DIR, "yahoo_links.csv")
CSV_PREMIUM        = os.path.join(DATA_DIR, "yahoo_premium.csv")
CSV_CRAWLED_LATEST = os.path.join(DATA_DIR, "crawled_latest.csv")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DB_DIR,   exist_ok=True)

_ensure_dirs()

def reset_base(new_base: str):
    """런타임에 베이스 경로를 샌드박스로 변경"""
    global BASE, DATA_DIR, DB_DIR, DB_PATH, CSV_PATH, CSV_PREMIUM, CSV_CRAWLED_LATEST
    if not new_base:
        return
    BASE = os.path.abspath(new_base)
    DATA_DIR = os.path.join(BASE, "data")
    DB_DIR   = os.path.join(BASE, "db")
    DB_PATH  = os.path.join(DB_DIR, "news.db")
    CSV_PATH           = os.path.join(DATA_DIR, "yahoo_links.csv")
    CSV_PREMIUM        = os.path.join(DATA_DIR, "yahoo_premium.csv")
    CSV_CRAWLED_LATEST = os.path.join(DATA_DIR, "crawled_latest.csv")
    _ensure_dirs()
    print(f"[reset_base] BASE={BASE}")

# ──────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────
def snake(s: str) -> str:
    s = re.sub(r"[\s\-]+", "_", str(s).strip())
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()

def choose_primary(row, priority, url_cols):
    for c in priority:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    for c in url_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_like = [c for c in df.columns if "date" in c or "time" in c or c.endswith("_at")]
    for c in date_like:
        try:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        except Exception:
            pass
    return df

def pick_pub_col(df: pd.DataFrame):
    cand = [c for c in ["pubdate","pub_date","date","time","published_time","created_at"] if c in df.columns]
    return cand[0] if cand else None

def _id_parts(row, pub_col: str|None):
    # primary_url 있으면 우선, 없으면 link/url/href, 모두 없으면 headline으로 식별성 보완
    key = (
        row.get("primary_url")
        or row.get("link")
        or row.get("url")
        or row.get("href")
        or row.get("headline")
        or ""
    )
    parts = [
        str(row.get("ticker") or ""),
        str(key),
        (str(row.get(pub_col)) if pub_col else "")
    ]
    return "|".join(parts)

def make_id(df: pd.DataFrame, pub_col: str|None):
    def _id(row):
        base = _id_parts(row, pub_col)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()
    df["id"] = df.apply(_id, axis=1)
    return df

def host(u: str|None):
    try:
        return urlparse(u).netloc
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# 날짜 정규화
# ──────────────────────────────────────────────────────────────
patterns = {
    'pattern1': r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}',
    'pattern2': r'\d{1,2}-[A-Za-z]{3}-\d{2,4}'
}

def normalize_pubdate(date_str):
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        clean_date = None
        for _, pattern in patterns.items():
            m = re.search(pattern, date_str)
            if m:
                clean_date = m.group(0)
                break
        if not clean_date:
            dt = parser.parse(date_str)
            return dt.strftime("%Y-%m-%d")
        dt = parser.parse(clean_date)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# DB 초기화
#   - articles: 사람이 읽는 용(메타+summary)
#   - integrated_index: 검색용 벡터(id, ivect only)
# ──────────────────────────────────────────────────────────────
def init_db(db_path: str | None = None):
    db_path = db_path or DB_PATH
    _ensure_dirs()
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # (1) 기사 메타+요약 저장
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        headline TEXT,
        ticker TEXT,
        pubdate TIMESTAMP,
        summary TEXT
    );
    """)
    cur.execute('CREATE INDEX IF NOT EXISTS ix_articles_pubdate ON articles(pubdate);')
    cur.execute('CREATE INDEX IF NOT EXISTS ix_articles_ticker  ON articles(ticker);')

    # (2) 검색용 벡터 인덱스 (id, ivect만)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS integrated_index (
        id    TEXT PRIMARY KEY,
        ivect BLOB
    );
    """)
    conn.commit()
    conn.close()
    print("테이블 준비 완료:", db_path)

# ──────────────────────────────────────────────────────────────
# (선택) NASDAQ-100 동적 로딩
# ──────────────────────────────────────────────────────────────
nasdaq100_tickers = ['NVDA','MSFT','AAPL','AMZN','META','AVGO','GOOGL','GOOG','TSLA','NFLX','COST','PLTR','ASML','TMUS','CSCO','AMD','AZN','LIN',
 'APP','PEP','SHOP','INTU','PDD','BKNG','MU','QCOM','TXN','ISRG','ARM','AMGN','ADBE','LRCX','GILD','HON','AMAT','PANW','KLAC','CMCSA','ADI','ADP',
 'MELI','INTC','DASH','CRWD','VRTX','CEG','MSTR','CDNS','SBUX','ORLY','CTAS','MDLZ','SNPS','TRI','ABNB','MAR','ADSK','PYPL','MNST','FTNT','CSX',
 'WDAY','AXON','REGN','AEP','MRVL','NXPI','ROP','FAST','PCAR','IDXX','PAYX','ROST','DDOG','CPRT','WBD','TEAM','BKR','TTWO','ZS','EXC','XEL','EA',
 'CCEP','FANG','KDP','CSGP','VRSK','CHTR','MCHP','GEHC','CTSH','KHC','ODFL','DXCM','TTD','CDW','BIIB','ON','LULU','GFS']

def _tickers_cache_path():
    return os.path.join(DATA_DIR, "nasdaq100_tickers.json")

def _load_cached_tickers(max_age_days=85):
    path = _tickers_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ts = datetime.fromisoformat(obj.get("fetched_at"))
        if datetime.utcnow() - ts > timedelta(days=max_age_days):
            return None
        tickers = obj.get("tickers") or []
        if isinstance(tickers, list) and tickers:
            return [t.strip().upper() for t in tickers]
    except Exception:
        return None
    return None

def _save_cached_tickers(tickers):
    os.makedirs(DATA_DIR, exist_ok=True)
    obj = {"fetched_at": datetime.utcnow().isoformat(), "tickers": tickers}
    with open(_tickers_cache_path(), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_nasdaq100_tickers(force_refresh=False, max_age_days=85):
    if not force_refresh:
        cached = _load_cached_tickers(max_age_days=max_age_days)
        if cached:
            return cached
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        for df in tables:
            cols = {str(c).strip().lower(): c for c in df.columns}
            if "ticker" in cols or "ticker symbol" in cols:
                colname = cols.get("ticker") or cols.get("ticker symbol")
                tickers = (
                    df[colname].dropna().astype(str)
                    .str.replace(r"[^A-Za-z\.]", "", regex=True)
                    .str.upper().tolist()
                )
                tickers = [t for t in tickers if t]
                tickers = sorted(list(dict.fromkeys(tickers)))
                if tickers:
                    _save_cached_tickers(tickers)
                    return tickers
    except Exception:
        pass
    return nasdaq100_tickers

# ──────────────────────────────────────────────────────────────
# 크롤링 (교체된 버전)
# ──────────────────────────────────────────────────────────────
import chromedriver_autoinstaller

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def get_href(query, count=20):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&newsCount={count}&start=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/127.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
        "Connection": "keep-alive",
    }
    response = requests.get(url, headers=headers, timeout=20)
    print(f"[{query}] Status Code: {response.status_code}")

    results = []
    if response.status_code == 200:
        data = response.json()
        for item in data.get("news", []):
            link = item.get("link")
            if link:
                results.append(link)
    return results[:count]

def update_csv(query, csv_path=None, premium_path=None):
    csv_path = csv_path or CSV_PATH
    premium_path = premium_path or CSV_PREMIUM
    new_links = get_href(query)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, encoding="utf-8-sig")
        old_pairs = set(zip(df_old.get("ticker", []), df_old.get("link", [])))
    else:
        df_old = pd.DataFrame(columns=["ticker","link","headline","pubdate","related_tickers","article"])
        old_pairs=set()

    if os.path.exists(premium_path):
        df_premium = pd.read_csv(premium_path, encoding="utf-8-sig")
        premium_pairs = set(zip(df_premium.get("ticker", []), df_premium.get("link", [])))
    else:
        df_premium = pd.DataFrame(columns=["ticker", "link"])
        premium_pairs = set()

    unique_links = [
        link for link in new_links
        if (query, link) not in old_pairs and (query, link) not in premium_pairs
    ]

    df_new = pd.DataFrame(unique_links, columns=["link"])
    df_new["ticker"] = query

    df_updated = pd.concat([df_old, df_new], ignore_index=True)
    df_updated = df_updated.drop_duplicates(subset=["ticker","link"], keep="first")
    _ensure_dirs()
    df_updated.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[{query}] 새로운 링크 {len(unique_links)}개 추가됨. 전체 {len(df_updated)}개 저장됨.")
    return df_updated

def save_premium(query, link, csv_path=None):
    csv_path = csv_path or CSV_PREMIUM
    if os.path.exists(csv_path):
        df_premium = pd.read_csv(csv_path, encoding="utf-8-sig")
    else:
        df_premium = pd.DataFrame(columns=["ticker", "link"])
    new_row = pd.DataFrame([{"ticker": query, "link": link}])
    df_premium = pd.concat([df_premium, new_row], ignore_index=True)
    df_premium = df_premium.drop_duplicates(subset=["ticker", "link"], keep="first")
    _ensure_dirs()
    df_premium.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"PREMIUM 저장됨: {link}")

def scrape_articles(df, query, csv_path="yahoo_links.csv"):
    """
    df: 이제는 특정 ticker만 필터링된 DataFrame을 받는다.
        (run_crawl()에서 df_all[df_all['ticker']==ticker] 로 잘라서 넘어옴)
    """
    chromedriver_autoinstaller.install()
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)

    # (선택) 네트워크 타임아웃 살짝 보수적으로
    try:
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(5)
        # RemoteConnection.set_timeout 경고 회피용
        driver.command_executor._client_config.timeout = 60  # 최신 selenium에서 권장 방식
    except Exception:
        pass

    i = 0
    while i < len(df):
        link = df.iloc[i]["link"]

        # 이미 headline 등 있으면 skip
        if "headline" in df.columns and pd.notnull(df.iloc[i].get("headline", None)):
            i += 1
            continue

        try:
            driver.get(link)
            time.sleep(2)

            # PREMIUM 체크
            try:
                head_str = driver.find_element(By.XPATH, '//*[@id="main-content-wrapper"]')
                is_premium = head_str.text.split('\n', 1)[0].strip()
            except NoSuchElementException:
                is_premium = ""

            if is_premium == "PREMIUM":
                print(f"Skip PREMIUM article: {link}")
                save_premium(query, link)   # PREMIUM 따로 저장
                df = df.drop(df.index[i]).reset_index(drop=True)
                continue  # drop했으니 같은 i에서 새 행 검사

            # 추가 필터 ① : Yahoo Finance Video
            try:
                head_str_2 = driver.find_element(By.CLASS_NAME, 'byline-attr-author.yf-1k5w6kz')
                if 'Yahoo Finance Video' in head_str_2.text:
                    print(f"Skip VIDEO article: {link}")
                    save_premium(query, link)
                    df = df.drop(df.index[i]).reset_index(drop=True)
                    continue
            except NoSuchElementException:
                pass

            # 추가 필터 ② : 특정 출처 필터링
            try:
                element = driver.find_element(
                    By.CSS_SELECTOR,
                    'div.cover-wrap div.top-header a.subtle-link[aria-label]'
                )
                aria_label_value = element.get_attribute('aria-label') or ''
                blocked_sources = [
                    'Motley Fool', 'Barchart', 'The Wall Street Journal',
                    'Zacks', 'Benzinga', 'Quartz'
                ]
                if any(src in aria_label_value for src in blocked_sources):
                    print(f"Skip SOURCE article ({aria_label_value}): {link}")
                    save_premium(query, link)
                    df = df.drop(df.index[i]).reset_index(drop=True)
                    continue
            except NoSuchElementException:
                pass

            # 기사 내용 추출 (여러 템플릿 대응)
            headline = None
            pubdate  = None
            article  = None

            # 1) 구(옛날) 템플릿 시도
            try:
                # CLASS_NAME은 하나만 써야 해서 'cover-headline'만 사용
                headline = driver.find_element(By.CLASS_NAME, "cover-headline").text
            except NoSuchElementException:
                pass
            try:
                pubdate = driver.find_element(By.CLASS_NAME, "byline-attr-meta-time").text
            except NoSuchElementException:
                pass
            try:
                article = driver.find_element(By.CLASS_NAME, "bodyItems-wrapper").text
            except NoSuchElementException:
                pass

            # 2) 새 CAAS 템플릿 시도 (headline이 아직 없을 때)
            if not headline:
                try:
                    # header.caas-title-wrapper > h1 구조
                    headline = driver.find_element(
                        By.CSS_SELECTOR,
                        "header.caas-title-wrapper h1"
                    ).text
                except NoSuchElementException:
                    try:
                        # 혹시 data-testid='Heading'만 있는 경우 대비
                        headline = driver.find_element(
                            By.CSS_SELECTOR,
                            "h1[data-testid='Heading']"
                        ).text
                    except NoSuchElementException:
                        pass

            if not pubdate:
                # time 태그에서 datetime/text 둘 중 하나 가져오기
                try:
                    t = driver.find_element(By.CSS_SELECTOR, "time.caas-attr-meta-time")
                    pubdate = t.get_attribute("datetime") or t.text
                except NoSuchElementException:
                    pass

            if not article:
                # 본문: caas-body 블록 여러 개 있을 수 있어서 전부 이어붙이기
                try:
                    bodies = driver.find_elements(By.CSS_SELECTOR, "div.caas-body")
                    if bodies:
                        article = "\n".join(b.text for b in bodies if b.text.strip())
                except NoSuchElementException:
                    pass

            # "더보기" 버튼 (옛 템플릿용) – 있으면 여전히 추가
            if article:
                try:
                    continue_button = driver.find_element(
                        By.CSS_SELECTOR,
                        "button[aria-label='Story Continues']"
                    )
                    continue_button.click()
                    time.sleep(1)
                    add_article = driver.find_element(By.CLASS_NAME, "read-more-wrapper").text
                    if add_article:
                        article += "\n" + add_article
                except NoSuchElementException:
                    pass

            # 최종적으로 제목/본문 둘 중 하나라도 없으면, 이 링크는 스킵
            if not headline or not article:
                print(f"[SKIP] headline/article missing → {link}")
                df = df.drop(df.index[i]).reset_index(drop=True)
                continue

            # 관련 티커
            try:
                tickers = driver.find_element(By.CLASS_NAME, 'carousel-top').text
            except NoSuchElementException:
                tickers = None

            # DataFrame 업데이트
            df.loc[i, "ticker"] = query
            df.loc[i, "headline"] = headline
            df.loc[i, "pubdate"] = pubdate
            df.loc[i, "related_tickers"] = tickers
            df.loc[i, "article"] = article

            i += 1

        except Exception as e:
            print(f"Error on {link}: {e}")
            # 문제 있는 행은 버리고 계속 진행
            df = df.drop(df.index[i]).reset_index(drop=True)

    # ticker별 df 내용을 원래 csv_path로 merge 저장
    try:
        full_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except FileNotFoundError:
        full_df = pd.DataFrame(columns=["ticker","link","headline","pubdate","related_tickers","article"])

    full_df_other = full_df[ full_df.get("ticker","") != query ]
    combined = pd.concat([full_df_other, df], ignore_index=True)

    _ensure_dirs()
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    driver.quit()
    print(f"[{query}] 크롤링 완료 후 저장됨.")

def cleanup_csv(csv_path=None):
    csv_path = csv_path or CSV_PATH
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if {"headline","pubdate","article"}.issubset(df.columns):
            mask = df[["headline", "pubdate", "article"]].isnull().all(axis=1)
            df = df.drop(df.index[mask]).reset_index(drop=True)
            _ensure_dirs()
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Null 행 {int(mask.sum())}개 삭제 완료")
        else:
            print("정리 스킵: 필요한 컬럼 없음(headline, pubdate, article)")

def run_crawl(csv_path=None, premium_path=None, use_dynamic_tickers=False, force_refresh_tickers=False):
    """
    수정된 버전:
    - ticker마다 전체 df_all을 받은 뒤, 그 ticker 것만 필터(df_tkr)해서 scrape_articles에 넘김
    - 이걸로 한 ticker마다 수만 건 다 도는 문제(타임아웃 유발) 막음
    """
    csv_path = csv_path or CSV_PATH
    premium_path = premium_path or CSV_PREMIUM
    tickers = nasdaq100_tickers
    if use_dynamic_tickers:
        try:
            tickers = get_nasdaq100_tickers(force_refresh=force_refresh_tickers)
            print(f"동적 티커 {len(tickers)}개 사용")
        except Exception as e:
            print("동적 티커 로딩 실패 → 하드코딩 리스트 사용:", e)

    cleanup_csv(csv_path)

    for ticker in tickers:
        df_all = update_csv(ticker, csv_path=csv_path, premium_path=premium_path)
        df_tkr = df_all[df_all["ticker"] == ticker].reset_index(drop=True)
        scrape_articles(df_tkr, ticker, csv_path=csv_path)

    # 전체 크롤링이 끝난 뒤, csv_path 내용을 crawled_latest.csv 로 스냅샷 저장
    if os.path.exists(csv_path):
        df_crawled = pd.read_csv(csv_path, encoding="utf-8-sig")
        _ensure_dirs()
        df_crawled.to_csv(CSV_CRAWLED_LATEST, index=False, encoding="utf-8-sig")
        print("saved:", CSV_CRAWLED_LATEST)

# ──────────────────────────────────────────────────────────────
# 전처리 & DB 적재  (articles 5컬럼만 업서트)
# ──────────────────────────────────────────────────────────────
def _apply_pubdate_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if "pubdate" in df.columns:
        df["pubdate_norm"] = df["pubdate"].apply(normalize_pubdate)
        df["pubdate"] = pd.to_datetime(df["pubdate_norm"], utc=True, errors="coerce")
    return df

def _apply_news_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo metadata CSV와 기존 full-article CSV를 같은 스키마로 맞춘다."""
    if "ticker" not in df.columns and "query_ticker" in df.columns:
        df["ticker"] = df["query_ticker"]
    if "headline" not in df.columns and "title" in df.columns:
        df["headline"] = df["title"]
    if "pubdate" not in df.columns and "provider_publish_time" in df.columns:
        df["pubdate"] = df["provider_publish_time"]
    return df

def upsert_minimal(df_ready: pd.DataFrame, db_path=None, table="articles"):
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    cols_real = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    common = [c for c in df_ready.columns if c in cols_real]
    df_tmp = df_ready[common].copy()
    df_tmp.to_sql("tmp", conn, if_exists="replace", index=False)
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({','.join(common)}) "
        f"SELECT {','.join(common)} FROM tmp;"
    )
    conn.execute("DROP TABLE tmp;")
    conn.commit()
    conn.close()

def preprocess_and_upsert(csv_in=None, db_path=None):
    csv_in = csv_in or CSV_PATH
    db_path = db_path or DB_PATH
    if not os.path.exists(csv_in):
        raise FileNotFoundError(f"CSV not found: {csv_in}")
    try:
        df = pd.read_csv(csv_in, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV is empty and has no columns: {csv_in}") from exc
    df.columns = [snake(c) for c in df.columns]
    df = _apply_news_column_aliases(df)

    # id 산출용 primary_url만 임시 계산
    url_cols = [c for c in df.columns if ("url" in c) or ("link" in c) or ("href" in c)]
    priority = [c for c in ["final_url","url","link","href"] if c in url_cols] + [c for c in url_cols if c not in ["final_url","url","link","href"]]
    df["primary_url"] = df.apply(lambda r: choose_primary(r, priority, url_cols), axis=1)

    before = len(df)
    df = df[~df["ticker"].isna() & ~df["headline"].isna()].copy()
    print(f"기본 필터링(ticker/headline 존재): {before} → {len(df)}")

    df = _apply_pubdate_normalization(df)
    df = parse_dates(df)

    pub_col = pick_pub_col(df)
    df = make_id(df, pub_col)

    want = ["id","headline","ticker","pubdate","summary"]
    for c in want:
        if c not in df.columns:
            df[c] = None
    df_ready = df[want].copy()

    upsert_minimal(df_ready, db_path=db_path)

# ──────────────────────────────────────────────────────────────
# 요약(LLM) – DB의 summary가 비어있을 때만 갱신
# ──────────────────────────────────────────────────────────────
SUMMARY_PROMPT_TMPL = """You are an expert financial analyst focused on {focus}.
News:
{article}
Instructions:
1) Extract objective, verifiable financial facts.
2) Use only facts explicitly present in the provided news text or metadata.
3) No opinions, predictions, or outside context.
Summary:"""

def _try_load_article_text_from_csv(id_row, csv_path_list):
    """가능하면 CSV에서 같은 기사 → headline/ticker/pubdate 기준 근접 매칭"""
    headline = id_row["headline"]; ticker = id_row["ticker"]; pubdate = id_row["pubdate"]
    for path in csv_path_list:
        if not path or not os.path.exists(path):
            continue
        try:
            x = pd.read_csv(path, encoding="utf-8-sig")
            x.columns = [snake(c) for c in x.columns]
            x = _apply_news_column_aliases(x)
            if "headline" not in x.columns:
                continue
            cand = x[x["headline"].astype(str)==str(headline)]
            if ticker is not None and "ticker" in x.columns:
                cand = cand[cand["ticker"].astype(str)==str(ticker)]
            if not cand.empty and "article" in cand.columns and cand["article"].notna().any():
                return str(cand.iloc[0]["article"])
            if not cand.empty:
                row = cand.iloc[0]
                parts = []
                for label, col in [
                    ("Title", "headline"),
                    ("Publisher", "publisher"),
                    ("Related tickers", "related_tickers"),
                    ("Published at", "pubdate"),
                    ("URL", "link"),
                ]:
                    if col in cand.columns and pd.notna(row.get(col)):
                        parts.append(f"{label}: {row.get(col)}")
                if parts:
                    return "\n".join(parts)
        except Exception:
            pass
    return None

def summarize_articles(db_path=None, model="gpt-4o-mini", per_minute=50, max_tokens=140):
    db_path = db_path or DB_PATH
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어야 합니다. pip install openai")
    api_key = get_openai_api_key(required=False)
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY 가 없습니다.")
    client = OpenAI(api_key=api_key)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, headline, pubdate, summary
        FROM articles
        WHERE (summary IS NULL OR LENGTH(TRIM(summary))=0)
    """).fetchall()

    print("요약 대상 행 수:", len(rows))
    if not rows:
        conn.close()
        return

    csv_candidates = [CSV_CRAWLED_LATEST, CSV_PATH]
    sleep_sec = max(0.0, 60.0 / max(per_minute, 1))
    done = 0
    for r in rows:
        focus = (r["ticker"] or "the company")
        article_text = _try_load_article_text_from_csv(r, csv_candidates)
        base_text = article_text if article_text else (r["headline"] or "")
        if not base_text:
            continue
        prompt = SUMMARY_PROMPT_TMPL.format(focus=focus, article=base_text[:8000])
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            summary = (resp.choices[0].message.content or "").strip()
            conn.execute("UPDATE articles SET summary=? WHERE id=?", (summary, r["id"]))
            conn.commit()
            done += 1
            print(f"[{done}/{len(rows)}] summarized id={r['id'][:8]}...")
        except Exception as e:
            print("요약 실패:", e)
        time.sleep(sleep_sec)
    conn.close()
    print("요약 완료")

# ──────────────────────────────────────────────────────────────
# 미리보기
# ──────────────────────────────────────────────────────────────
def preview_db(db_path=None, limit=10):
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM articles;").fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(articles);").fetchall()]
    print("총 행 수:", n)
    print("컬럼:", cols)
    df = pd.read_sql_query(f"""
    SELECT id, substr(headline,1,100) AS headline, ticker,
           substr(summary,1,120) AS summary_head, pubdate
    FROM articles
    ORDER BY COALESCE(pubdate, id) DESC
    LIMIT {int(limit)};
    """, conn)
    print(df)
    conn.close()

# =================================================================
# 통합 벡터 인덱스 (integrated_index: id, ivect만)
# =================================================================
DIM_TEXT = 384
DIM_META = 32
DIM_TOTAL = 256
W_TEXT, W_TICKER, W_TIME = 0.8, 0.1, 0.1
PROJ_SEED = 20241027

_proj_matrix = None
_concat_dim = None

def _get_proj_matrix(concat_dim):
    global _proj_matrix, _concat_dim
    if _proj_matrix is not None and _concat_dim == concat_dim:
        return _proj_matrix
    rng = np.random.default_rng(PROJ_SEED)
    _proj_matrix = (rng.normal(size=(concat_dim, DIM_TOTAL)) / math.sqrt(concat_dim)).astype('float32')
    _concat_dim = concat_dim
    return _proj_matrix

def _project_concat(cat: np.ndarray, proj: np.ndarray) -> np.ndarray:
    return np.sum(cat.astype("float64")[:, None] * proj.astype("float64"), axis=0).astype("float32")

def _deterministic_unit_vec(key: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype('float32')
    return v / (np.linalg.norm(v)+1e-8)

def _scalar_expand_unit(x: float, dim: int) -> np.ndarray:
    v = np.full(dim, float(x), dtype='float32')
    return v / (np.linalg.norm(v)+1e-8)

def _recency_score(pub_dt):
    if pub_dt is None:
        return 0.5
    try:
        dt = pd.to_datetime(pub_dt, utc=True)
    except Exception:
        dt = pd.to_datetime(pub_dt)
    days = (pd.Timestamp.utcnow() - dt).total_seconds()/86400.0
    return float(math.exp(-0.25*max(0.0, days)))

def _as_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype="float32").tobytes()

def _ivect_from_row(row, model):
    # row: {headline, ticker, pubdate, summary}
    text = ((row.get("summary") or "") + " " + (row.get("headline") or "")).strip() or (row.get("headline") or "")
    vt = model.encode([text], normalize_embeddings=True)[0]  # 384
    tkr = row.get("ticker") or "UNK"
    vtkr = _deterministic_unit_vec(f"ticker:{tkr}", DIM_META)  # 32
    r = _recency_score(row.get("pubdate"))
    vtime = _scalar_expand_unit(r, DIM_META)                  # 32
    cat = np.concatenate([W_TEXT*vt, W_TICKER*vtkr, W_TIME*vtime]).astype('float32')  # 384+32+32=448
    proj = _get_proj_matrix(cat.shape[0])                     # (448,256)
    v = _project_concat(cat, proj)                            # (256,)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def build_integrated_index(db_path=None, days: int = 30):
    """
    articles에서 summary가 채워진 애들만 뽑아서
    integrated_index(id, ivect) 에 저장/업데이트
    """
    db_path = db_path or DB_PATH
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.max_seq_length = 512

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"""
        SELECT id, headline, ticker, pubdate, summary
        FROM articles
        WHERE summary IS NOT NULL AND LENGTH(TRIM(summary))>0
          AND (pubdate IS NULL OR pubdate >= datetime('now','-{int(days)} days'))
    """).fetchall()

    if not rows:
        conn.close()
        print("build-index: 대상 기사 없음")
        return

    cur = conn.cursor()
    done = 0
    for r in rows:
        row = dict(r)
        v = _ivect_from_row(row, model)
        cur.execute("""
            INSERT INTO integrated_index (id, ivect)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET
              ivect=excluded.ivect
        """, (row["id"], _as_blob(v)))
        done += 1
        if done % 200 == 0:
            conn.commit()
            print(f"build-index: {done}건 진행 중…")
    conn.commit()
    conn.close()
    print(f"build-index: 완료 ({done}건 업서트)")

def _vec_from_blob(b):
    if b is None:
        return None
    if isinstance(b, memoryview):
        b = b.tobytes()
    v = np.frombuffer(b, dtype=np.float32)
    return v / (np.linalg.norm(v)+1e-8)

def preview_index(db_path=None, limit=10):
    """
    integrated_index(id, ivect)와 articles(id→headline 등) 조인해서
    사람 눈으로 확인할 수 있게 미리보기
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT a.id,
               a.ticker,
               substr(a.headline,1,60)  AS headline,
               substr(a.summary,1,80)   AS summary_head,
               a.pubdate,
               LENGTH(idx.ivect)        AS ivect_bytes
        FROM integrated_index idx
        JOIN articles a ON a.id = idx.id
        ORDER BY COALESCE(a.pubdate, a.id) DESC
        LIMIT {int(limit)};
    """, conn)
    print(df)
    conn.close()

def search_index(query: str, ticker: str|None = None, topk: int = 10, db_path=None):
    """
    1) 쿼리를 벡터화
    2) integrated_index에서 벡터 전부 가져와 코사인 유사도 계산
    3) 상위 topk개에 대해 articles 조인해서 리턴
    """
    db_path = db_path or DB_PATH
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.max_seq_length = 512

    # 쿼리 벡터 (ticker/time 정보는 0 벡터로 처리)
    vt = model.encode([query], normalize_embeddings=True)[0]
    zeros = _scalar_expand_unit(0.0, DIM_META)
    cat = np.concatenate([W_TEXT*vt, W_TICKER*zeros, W_TIME*zeros]).astype('float32')
    proj = _get_proj_matrix(cat.shape[0])
    qv = _project_concat(cat, proj)
    qv /= (np.linalg.norm(qv)+1e-8)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # integrated_index + articles 조인으로 메타데이터까지 한 번에
    rows = conn.execute("""
        SELECT idx.id, idx.ivect,
               a.headline, a.ticker, a.pubdate, a.summary
        FROM integrated_index idx
        JOIN articles a ON a.id = idx.id
        WHERE (? IS NULL OR a.ticker = ?)
    """, (ticker, ticker)).fetchall()
    conn.close()

    scored = []
    for r in rows:
        v = _vec_from_blob(r["ivect"])
        if v is None:
            continue
        sim = float(np.dot(qv, v))
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]

# ──────────────────────────────────────────────────────────────
# 오래된 기사 삭제 + 유사 기사 정리
# ──────────────────────────────────────────────────────────────
def purge_old_articles(db_path=None, days_keep: int = 14):
    """
    pubdate가 days_keep일보다 더 오래된(=15일째부터) 기사들은 삭제.
    articles, integrated_index 둘 다 정리.
    pubdate가 NULL이면 삭제하지 않음.
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cutoff_dt = pd.Timestamp.utcnow() - pd.Timestamp.utcnow().tz_localize("UTC").tz_localize(None) if False else None
    cutoff_dt = pd.Timestamp.utcnow() - pd.Timedelta(days=days_keep)
    cutoff_str = cutoff_dt.isoformat()

    old_rows = conn.execute("""
        SELECT id FROM articles
        WHERE pubdate IS NOT NULL AND pubdate < ?
    """, (cutoff_str,)).fetchall()
    old_ids = [r["id"] for r in old_rows]

    if old_ids:
        qmarks = ",".join(["?"]*len(old_ids))
        conn.execute(f"DELETE FROM integrated_index WHERE id IN ({qmarks})", old_ids)
        conn.execute(f"DELETE FROM articles WHERE id IN ({qmarks})", old_ids)
        conn.commit()
        print(f"purge_old_articles: {len(old_ids)}개 삭제(pubdate < {cutoff_str})")
    else:
        print("purge_old_articles: 삭제 대상 없음")

    conn.close()

def _quality_score(headline: str|None, summary: str|None):
    """
    요약이 길수록, 헤드라인이 어느 정도 길수록 점수를 높게 줌.
    """
    h = headline or ""
    s = summary or ""
    return len(s) + 0.5*len(h)

def dedupe_similar_articles(db_path=None, threshold: float = 0.9):
    """
    같은 ticker 안에서 내용이 거의 같은 기사(코사인 유사도 >= threshold)를 중복으로 판단.
    가장 점수 높은 것만 남기고 나머지는 삭제.
    삭제 시 articles와 integrated_index 둘 다 정리.
    """
    db_path = db_path or DB_PATH
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.max_seq_length = 512

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, headline, summary
        FROM articles
    """).fetchall()

    if not rows:
        print("dedupe_similar_articles: 정리 대상 없음 (articles 비어있음)")
        conn.close()
        return

    # ticker별로 묶기
    by_tkr = {}
    for r in rows:
        t = r["ticker"] or "UNK"
        by_tkr.setdefault(t, []).append(dict(r))

    to_delete = set()

    for tkr, items in by_tkr.items():
        if len(items) <= 1:
            continue

        # 텍스트 만들기 (summary 우선, 없으면 headline)
        texts = []
        for it in items:
            text = (it.get("summary") or "").strip()
            if not text:
                text = (it.get("headline") or "").strip()
            texts.append(text)

        # 임베딩
        embs = model.encode(texts, normalize_embeddings=True)  # shape = (N, 384)

        n = len(items)
        used = set()  # 이미 처리한 애들 기록

        for i in range(n):
            if i in used:
                continue
            cluster = [i]
            vi = embs[i]
            for j in range(i+1, n):
                if j in used:
                    continue
                vj = embs[j]
                sim = float(np.dot(vi, vj))
                if sim >= threshold:
                    cluster.append(j)

            if len(cluster) == 1:
                used.add(cluster[0])
                continue

            # cluster 내에서 품질 점수 가장 높은 애 하나만 keep
            best_idx = None
            best_score = -1e9
            for k in cluster:
                score_k = _quality_score(items[k].get("headline"), items[k].get("summary"))
                if score_k > best_score:
                    best_score = score_k
                    best_idx = k

            # best_idx만 keep, 나머지는 삭제 후보
            for k in cluster:
                used.add(k)
                if k != best_idx:
                    to_delete.add(items[k]["id"])

    if to_delete:
        to_delete = list(to_delete)
        qmarks = ",".join(["?"]*len(to_delete))
        conn.execute(f"DELETE FROM integrated_index WHERE id IN ({qmarks})", to_delete)
        conn.execute(f"DELETE FROM articles          WHERE id IN ({qmarks})", to_delete)
        conn.commit()
        print(f"dedupe_similar_articles: 중복 기사 {len(to_delete)}개 삭제 (threshold={threshold})")
    else:
        print("dedupe_similar_articles: 삭제할 중복 없음")

    conn.close()

def cleanup_db(db_path=None, days_keep: int = 14, sim_threshold: float = 0.9, preview_limit: int = 10):
    """
    1) 오래된 기사 삭제
    2) 중복 기사 정리
    3) 현재 상태 preview
    """
    db_path = db_path or DB_PATH
    purge_old_articles(db_path=db_path, days_keep=days_keep)
    dedupe_similar_articles(db_path=db_path, threshold=sim_threshold)
    preview_db(db_path=db_path, limit=preview_limit)

# ──────────────────────────────────────────────────────────────
# 샌드박스 전용 유틸(샘플 뽑기 & 데모)
# ──────────────────────────────────────────────────────────────
def make_sample_csv(src_csv: str, out_name: str = "sample3.csv", n: int = 3, ticker: str | None = None):
    """원본 CSV에서 상위 N개만 골라 샌드박스 DATA_DIR에 저장"""
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"src_csv not found: {src_csv}")
    try:
        df = pd.read_csv(src_csv, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"src_csv is empty and has no columns: {src_csv}") from exc
    if ticker:
        df = df[df.get("ticker", "").astype(str).str.upper() == ticker.upper()]
    df_out = df.head(int(n)).copy()
    dst = os.path.join(DATA_DIR, out_name)
    _ensure_dirs()
    df_out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"make-sample: saved {len(df_out)} rows → {dst}")
    return dst

def sandbox_demo(sample_csv_path: str, days_keep: int = 365, index_days: int = 365, preview_limit: int = 10):
    """
    샌드박스에서 샘플 CSV 기준으로 end-to-end 데모
    순서: init → preprocess → summarize → cleanup → build-index → preview-index
    """
    init_db(db_path=DB_PATH)
    preprocess_and_upsert(csv_in=sample_csv_path, db_path=DB_PATH)
    summarize_articles(db_path=DB_PATH)
    cleanup_db(db_path=DB_PATH, days_keep=days_keep, sim_threshold=0.9, preview_limit=preview_limit)
    build_integrated_index(db_path=DB_PATH, days=index_days)
    preview_index(db_path=DB_PATH, limit=preview_limit)

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="News DB pipeline (articles + integrated_index(id,ivect))")
    parser.add_argument("--base", default=None, help="샌드박스 베이스 폴더(미지정시 기본 BASE 사용)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    p_crawl = sub.add_parser("crawl")
    p_crawl.add_argument("--csv", default=None)
    p_crawl.add_argument("--use-dynamic", action="store_true")
    p_crawl.add_argument("--refresh-tickers", action="store_true")

    # 🔹 추가: 크롤링만 주기 반복
    p_auto_lite = sub.add_parser("autocrawl-lite", help="크롤링만 주기적으로 반복 실행")
    p_auto_lite.add_argument("--csv", default=None)
    p_auto_lite.add_argument("--interval-min", type=int, default=120)
    p_auto_lite.add_argument("--use-dynamic", action="store_true")
    p_auto_lite.add_argument("--refresh-tickers", action="store_true")

    p_auto = sub.add_parser("autocrawl")
    p_auto.add_argument("--csv", default=None)
    p_auto.add_argument("--interval-min", type=int, default=120)
    p_auto.add_argument("--use-dynamic", action="store_true")
    p_auto.add_argument("--refresh-tickers", action="store_true")

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--csv", default=None)

    sub.add_parser("summarize")

    p_prev = sub.add_parser("preview")
    p_prev.add_argument("--limit", type=int, default=10)

    # 인덱스
    p_bidx = sub.add_parser("build-index")
    p_bidx.add_argument("--days", type=int, default=30)

    p_pvidx = sub.add_parser("preview-index")
    p_pvidx.add_argument("--limit", type=int, default=10)

    p_sidx = sub.add_parser("search-index")
    p_sidx.add_argument("--query", required=True)
    p_sidx.add_argument("--topk", type=int, default=10)
    p_sidx.add_argument("--ticker", default=None)

    # 샌드박스 유틸
    p_mks = sub.add_parser("make-sample")
    p_mks.add_argument("--src", required=True, help="원본 CSV 경로(예: 기존 crawled_latest.csv)")
    p_mks.add_argument("--out-name", default="sample3.csv")
    p_mks.add_argument("--n", type=int, default=3)
    p_mks.add_argument("--ticker", default=None, help="특정 티커만 추출(Optional)")

    p_demo = sub.add_parser("sandbox-demo")
    p_demo.add_argument("--csv", required=True, help="샌드박스 CSV 경로(예: sample3.csv)")
    p_demo.add_argument("--days-keep", type=int, default=365)   # 데모용
    p_demo.add_argument("--index-days", type=int, default=365)
    p_demo.add_argument("--preview-limit", type=int, default=10)

    # cleanup-db (오래된 기사 삭제 + 중복 정리)
    p_clean = sub.add_parser("cleanup-db")
    p_clean.add_argument("--days-keep", type=int, default=14,
                         help="며칠치만 유지할지 (14면 14일 이전 기사=15일차부터 삭제)")
    p_clean.add_argument("--sim-threshold", type=float, default=0.9,
                         help="같은 ticker 내 코사인 유사도 임계값")
    p_clean.add_argument("--preview-limit", type=int, default=10)

    args = parser.parse_args()

    # --base 적용
    if args.base:
        reset_base(args.base)

    if args.cmd == "init":
        init_db(db_path=DB_PATH)

    elif args.cmd == "crawl":
        init_db(db_path=DB_PATH)
        run_crawl(
            csv_path=args.csv or CSV_PATH,
            use_dynamic_tickers=args.use_dynamic,
            force_refresh_tickers=args.refresh_tickers
        )

    elif args.cmd == "autocrawl-lite":
        init_db(db_path=DB_PATH)
        interval = max(1, int(args.interval_min))
        print(f"🌀 크롤링 전용 루프 시작: 매 {interval}분 간격 (동적티커={args.use_dynamic}, 강제리프레시={args.refresh_tickers})")
        try:
            while True:
                cycle_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n=== [Crawl cycle start @ {cycle_ts}] ===")
                try:
                    run_crawl(
                        csv_path=args.csv or CSV_PATH,
                        use_dynamic_tickers=args.use_dynamic,
                        force_refresh_tickers=args.refresh_tickers
                    )
                    print("✅ 이번 크롤 완료")
                except Exception as e:
                    print(f"[크롤 오류] {e}")
                # 소폭 지터 추가
                jitter = min(30, int(time.time()) % 17)
                sleep_sec = max(60, interval * 60 + jitter)
                next_ts = (datetime.now() + timedelta(seconds=sleep_sec)).strftime("%Y-%m-%d %H:%M:%S")
                print(f"💤 {sleep_sec}초 대기 → 다음 사이클: {next_ts}")
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            print("\n🛑 autocrawl-lite 종료")

    elif args.cmd == "autocrawl":
        init_db(db_path=DB_PATH)
        print(f"자동 크롤(간이): interval={args.interval_min}분")
        try:
            while True:
                run_crawl(
                    csv_path=args.csv or CSV_PATH,
                    use_dynamic_tickers=args.use_dynamic,
                    force_refresh_tickers=args.refresh_tickers
                )
                preprocess_and_upsert(csv_in=args.csv or CSV_PATH, db_path=DB_PATH)
                summarize_articles(db_path=DB_PATH)
                cleanup_db(db_path=DB_PATH, days_keep=14, sim_threshold=0.9, preview_limit=10)
                build_integrated_index(db_path=DB_PATH, days=30)
                time.sleep(max(60, args.interval_min*60))
        except KeyboardInterrupt:
            print("autocrawl 종료")

    elif args.cmd == "preprocess":
        init_db(db_path=DB_PATH)
        preprocess_and_upsert(csv_in=args.csv or CSV_PATH, db_path=DB_PATH)

    elif args.cmd == "summarize":
        summarize_articles(db_path=DB_PATH)

    elif args.cmd == "preview":
        preview_db(db_path=DB_PATH, limit=args.limit)

    elif args.cmd == "build-index":
        build_integrated_index(db_path=DB_PATH, days=args.days)

    elif args.cmd == "preview-index":
        preview_index(db_path=DB_PATH, limit=args.limit)

    elif args.cmd == "search-index":
        hits = search_index(
            query=args.query,
            ticker=args.ticker,
            topk=args.topk,
            db_path=DB_PATH
        )
        print(f"Top-{args.topk} for query: {args.query!r} (ticker={args.ticker})")
        for sim, r in hits:
            print(f"{sim: .3f} | {r['ticker']} | {str(r['pubdate'])[:19]} | {r['headline'][:80]}")

    elif args.cmd == "make-sample":
        make_sample_csv(
            src_csv=args.src,
            out_name=args.out_name,
            n=args.n,
            ticker=args.ticker
        )

    elif args.cmd == "sandbox-demo":
        init_db(db_path=DB_PATH)
        sandbox_demo(
            sample_csv_path=args.csv,
            days_keep=args.days_keep,
            index_days=args.index_days,
            preview_limit=args.preview_limit
        )

    elif args.cmd == "cleanup-db":
        cleanup_db(
            db_path=DB_PATH,
            days_keep=args.days_keep,
            sim_threshold=args.sim_threshold,
            preview_limit=args.preview_limit
        )

if __name__ == "__main__":
    main()
