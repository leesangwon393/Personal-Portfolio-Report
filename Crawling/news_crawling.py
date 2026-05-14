import argparse
import os
import random
import time
from typing import cast

import chromedriver_autoinstaller
import pandas as pd
import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


NASDAQ100_TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "AVGO",
    "GOOGL",
    "GOOG",
    "TSLA",
    "NFLX",
    "COST",
    "PLTR",
    "ASML",
    "TMUS",
    "CSCO",
    "AMD",
    "AZN",
    "LIN",
    "APP",
    "PEP",
    "SHOP",
    "INTU",
    "PDD",
    "BKNG",
    "MU",
    "QCOM",
    "TXN",
    "ISRG",
    "ARM",
    "AMGN",
    "ADBE",
    "LRCX",
    "GILD",
    "HON",
    "AMAT",
    "PANW",
    "KLAC",
    "CMCSA",
    "ADI",
    "ADP",
    "MELI",
    "INTC",
    "DASH",
    "CRWD",
    "VRTX",
    "CEG",
    "MSTR",
    "CDNS",
    "SBUX",
    "ORLY",
    "CTAS",
    "MDLZ",
    "SNPS",
    "TRI",
    "ABNB",
    "MAR",
    "ADSK",
    "PYPL",
    "MNST",
    "FTNT",
    "CSX",
    "WDAY",
    "AXON",
    "REGN",
    "AEP",
    "MRVL",
    "NXPI",
    "ROP",
    "FAST",
    "PCAR",
    "IDXX",
    "PAYX",
    "ROST",
    "DDOG",
    "CPRT",
    "WBD",
    "TEAM",
    "BKR",
    "TTWO",
    "ZS",
    "EXC",
    "XEL",
    "EA",
    "CCEP",
    "FANG",
    "KDP",
    "CSGP",
    "VRSK",
    "CHTR",
    "MCHP",
    "GEHC",
    "CTSH",
    "KHC",
    "ODFL",
    "DXCM",
    "TTD",
    "CDW",
    "BIIB",
    "ON",
    "LULU",
    "GFS",
]


def get_href(query: str, count: int = 20) -> list[str]:
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&newsCount={count}&start=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
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
            if link and link.startswith("https://finance.yahoo.com/news/"):
                results.append(link)

    return results[:count]


def _load_csv(path: str, columns: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def update_csv(
    query: str,
    csv_path: str = "yahoo_links.csv",
    premium_path: str = "yahoo_premium.csv",
    count: int = 20,
) -> pd.DataFrame:
    new_links = get_href(query, count=count)

    df_old = _load_csv(
        csv_path,
        ["ticker", "link", "headline", "pubdate", "related_tickers", "article"],
    )
    df_premium = _load_csv(premium_path, ["ticker", "link"])

    old_pairs = (
        set(zip(df_old["ticker"].astype(str), df_old["link"].astype(str)))
        if not df_old.empty
        else set()
    )
    premium_pairs = (
        set(zip(df_premium["ticker"].astype(str), df_premium["link"].astype(str)))
        if not df_premium.empty
        else set()
    )

    unique_links = [
        link
        for link in new_links
        if (query, link) not in old_pairs and (query, link) not in premium_pairs
    ]

    df_new = pd.DataFrame({"link": unique_links})
    df_new["ticker"] = query

    df_updated = pd.concat([df_old, df_new], ignore_index=True)
    df_updated = df_updated.drop_duplicates(subset=["ticker", "link"], keep="first")
    _ensure_parent_dir(csv_path)
    df_updated.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(
        f"[{query}] 새로운 링크 {len(unique_links)}개 추가됨. 전체 {len(df_updated)}개 저장됨."
    )
    return df_updated


def save_premium(query: str, link: str, csv_path: str = "yahoo_premium.csv") -> None:
    df_premium = _load_csv(csv_path, ["ticker", "link"])
    new_row = pd.DataFrame([{"ticker": query, "link": link}])
    df_premium = pd.concat([df_premium, new_row], ignore_index=True)
    df_premium = df_premium.drop_duplicates(subset=["ticker", "link"], keep="first")
    _ensure_parent_dir(csv_path)
    df_premium.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"PREMIUM 저장됨: {link}")


def _build_driver():
    chromedriver_autoinstaller.install()
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1400,1200")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(5)
        client_config = getattr(driver.command_executor, "_client_config", None)
        if client_config is not None:
            client_config.timeout = 60
    except Exception:
        pass
    return driver


def _safe_click(driver, element) -> bool:
    try:
        element.click()
        return True
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False


def _collect_article_text(driver) -> str:
    chunks: list[str] = []
    selectors = [
        (By.CLASS_NAME, "bodyItems-wrapper"),
        (By.CSS_SELECTOR, "div.caas-body"),
        (By.CSS_SELECTOR, "article"),
        (By.CSS_SELECTOR, "main"),
    ]
    for by, selector in selectors:
        try:
            elements = driver.find_elements(by, selector)
            for element in elements:
                text = (element.text or "").strip()
                if text and text not in chunks:
                    chunks.append(text)
        except Exception:
            continue
    return "\n".join(chunks).strip()


def _expand_story_controls(driver, max_clicks: int = 8) -> None:
    selectors = [
        (By.CSS_SELECTOR, "button[aria-label='Story Continues']"),
        (By.CSS_SELECTOR, "button[aria-label*='Next page']"),
        (By.CSS_SELECTOR, "button[aria-label*='Next Page']"),
        (By.XPATH, "//button[contains(., 'Story Continues')]"),
        (
            By.XPATH,
            "//button[contains(., 'Next page') or contains(., 'Next Page') or contains(., 'Next')]",
        ),
        (
            By.XPATH,
            "//a[contains(., 'Next page') or contains(., 'Next Page') or contains(., 'Story Continues')]",
        ),
    ]

    clicks = 0
    while clicks < max_clicks:
        clicked = False
        for by, selector in selectors:
            try:
                controls = driver.find_elements(by, selector)
            except Exception:
                continue
            for control in controls:
                label = f"{control.text} {control.get_attribute('aria-label') or ''}".strip()
                if not label:
                    continue
                label_lower = label.lower()
                if (
                    "story continues" not in label_lower
                    and "next page" not in label_lower
                    and label_lower != "next"
                ):
                    continue
                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", control
                    )
                except Exception:
                    pass
                if _safe_click(driver, control):
                    time.sleep(random.uniform(0.8, 1.5))
                    clicks += 1
                    clicked = True
                    break
            if clicked:
                break
        if not clicked:
            break


def scrape_articles(
    df: pd.DataFrame,
    query: str,
    csv_path: str = "yahoo_links.csv",
    premium_path: str = "yahoo_premium.csv",
) -> pd.DataFrame:
    driver = _build_driver()
    rows: list[dict[str, object]] = []

    for _, src in df.iterrows():
        link = str(src.get("link", "")).strip()
        if not link:
            continue
        try:
            driver.get(link)
            time.sleep(random.uniform(1.5, 2.5))

            try:
                head_str = driver.find_element(
                    By.XPATH, '//*[@id="main-content-wrapper"]'
                )
                is_premium = head_str.text.split("\n", 1)[0].strip()
            except NoSuchElementException:
                is_premium = ""

            if is_premium == "PREMIUM":
                print(f"Skip PREMIUM article: {link}")
                save_premium(query, link, csv_path=premium_path)
                continue

            headline = None
            pubdate = None
            article = None

            for selector in [
                (By.CLASS_NAME, "cover-headline"),
                (By.CSS_SELECTOR, "header.caas-title-wrapper h1"),
                (By.CSS_SELECTOR, "h1[data-testid='Heading']"),
            ]:
                if headline:
                    break
                try:
                    headline = driver.find_element(*selector).text
                except NoSuchElementException:
                    pass

            for selector in [
                (By.CLASS_NAME, "byline-attr-meta-time"),
                (By.CSS_SELECTOR, "time.caas-attr-meta-time"),
            ]:
                if pubdate:
                    break
                try:
                    el = driver.find_element(*selector)
                    pubdate = el.get_attribute("datetime") or el.text
                except NoSuchElementException:
                    pass

            article = _collect_article_text(driver)
            _expand_story_controls(driver)
            expanded_article = _collect_article_text(driver)
            if len(expanded_article) > len(article):
                article = expanded_article

            if not headline or not article:
                print(f"[SKIP] headline/article missing → {link}")
                continue

            try:
                tickers = driver.find_element(By.CLASS_NAME, "carousel-top").text
            except NoSuchElementException:
                tickers = None

            rows.append(
                {
                    "ticker": query,
                    "link": link,
                    "headline": headline,
                    "pubdate": pubdate,
                    "related_tickers": tickers,
                    "article": article,
                }
            )

        except Exception as e:
            print(f"Error on {link}: {e}")

    df_result = pd.DataFrame.from_records(
        rows,
        columns=["ticker", "link", "headline", "pubdate", "related_tickers", "article"],
    )
    full_df = _load_csv(
        csv_path,
        ["ticker", "link", "headline", "pubdate", "related_tickers", "article"],
    )
    if "ticker" in full_df.columns:
        full_df_other = full_df[full_df["ticker"].astype(str) != query]
    else:
        full_df_other = full_df.iloc[0:0].copy()
    combined = cast(
        pd.DataFrame, pd.concat([full_df_other, df_result], ignore_index=True)
    )
    _ensure_parent_dir(csv_path)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    driver.quit()
    print(f"[{query}] 크롤링 완료 후 저장됨.")
    return combined


def cleanup_csv(csv_path: str = "yahoo_links.csv") -> None:
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    needed = {"headline", "pubdate", "article"}
    if not needed.issubset(df.columns):
        print("정리 스킵: 필요한 컬럼 없음(headline, pubdate, article)")
        return
    clean_df = df.dropna(subset=["headline", "pubdate", "article"], how="all")
    null_count = int(len(df) - len(clean_df))
    df = clean_df.reset_index(drop=True)
    _ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Null 행 {null_count}개 삭제 완료")


def run_for_tickers(
    tickers: list[str],
    csv_path: str = "yahoo_links.csv",
    premium_path: str = "yahoo_premium.csv",
    count: int = 20,
) -> None:
    cleanup_csv(csv_path)
    for ticker in tickers:
        df = update_csv(
            ticker, csv_path=csv_path, premium_path=premium_path, count=count
        )
        if "ticker" in df.columns:
            df_ticker = df[df["ticker"].astype(str) == ticker].reset_index(drop=True)
        else:
            df_ticker = df.iloc[0:0].copy()
        scrape_articles(df_ticker, ticker, csv_path=csv_path, premium_path=premium_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Yahoo Finance news crawler grouped by ticker"
    )
    parser.add_argument("--csv", default="yahoo_links.csv", help="output CSV path")
    parser.add_argument(
        "--premium", default="yahoo_premium.csv", help="premium links CSV path"
    )
    parser.add_argument("--count", type=int, default=20, help="news links per ticker")
    parser.add_argument(
        "--tickers", nargs="*", default=NASDAQ100_TICKERS, help="tickers to crawl"
    )
    args = parser.parse_args()

    run_for_tickers(
        args.tickers,
        csv_path=args.csv,
        premium_path=args.premium,
        count=args.count,
    )


if __name__ == "__main__":
    main()
