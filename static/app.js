const form = document.querySelector("#portfolio-form");
const message = document.querySelector("#message");
const portfolioInput = document.querySelector("#portfolio");
const searchInput = document.querySelector("#ticker-search");
const quantityInput = document.querySelector("#ticker-quantity");
const addTickerButton = document.querySelector("#add-ticker");
const suggestions = document.querySelector("#ticker-suggestions");
const selectedTickers = document.querySelector("#selected-tickers");
const fields = {
  srisk: document.querySelector("#srisk"),
  category: document.querySelector("#category"),
  sigma: document.querySelector("#sigma"),
  mdd: document.querySelector("#mdd"),
  beta: document.querySelector("#beta"),
  hhi: document.querySelector("#hhi"),
  pie: document.querySelector("#allocation-pie"),
  totalValue: document.querySelector("#total-value"),
  priceWarning: document.querySelector("#price-warning"),
  sectors: document.querySelector("#sectors"),
  holdings: document.querySelector("#holdings"),
  reports: document.querySelector("#reports"),
};

let highlightedSuggestion = 0;
let activeTicker = "";
const pieColors = ["#1f6feb", "#0f766e", "#b76e00", "#c83b3b", "#6f42c1", "#2f6f4e", "#8f5c00", "#4f6378"];

function parsePortfolioText() {
  const portfolio = [];
  portfolioInput.value.split(",").forEach((item) => {
    const parts = item.trim().split(/\s+/);
    if (parts.length !== 2) return;
    const weight = Number(parts[1]);
    if (!parts[0] || Number.isNaN(weight)) return;
    portfolio.push({ ticker: parts[0].toUpperCase(), weight });
  });
  return portfolio;
}

function writePortfolioText(portfolio) {
  portfolioInput.value = portfolio
    .map((item) => `${item.ticker} ${Number(item.weight).toFixed(2)}`)
    .join(", ");
  renderSelectedTickers();
}

function renderSelectedTickers() {
  selectedTickers.innerHTML = "";
  parsePortfolioText().forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "ticker-chip";
    chip.textContent = `${item.ticker} ${Number(item.weight).toLocaleString()} shares`;
    chip.addEventListener("click", () => {
      writePortfolioText(
        parsePortfolioText().filter((current) => current.ticker !== item.ticker)
      );
    });
    selectedTickers.appendChild(chip);
  });
}

function scoreTicker(option, query) {
  const ticker = option.Ticker.toLowerCase();
  const sector = String(option.Sector || "").toLowerCase();
  const q = query.toLowerCase();
  if (ticker === q) return 0;
  if (ticker.startsWith(q)) return 1;
  if (ticker.includes(q)) return 2;
  if (sector.includes(q)) return 3;
  return 99;
}

function getMatches(query) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return tickerOptions
    .map((option) => ({ ...option, score: scoreTicker(option, q) }))
    .filter((option) => option.score < 99)
    .sort((a, b) => a.score - b.score || a.Ticker.localeCompare(b.Ticker))
    .slice(0, 8);
}

function renderSuggestions(query) {
  const matches = getMatches(query);
  suggestions.innerHTML = "";
  highlightedSuggestion = Math.min(highlightedSuggestion, Math.max(matches.length - 1, 0));
  suggestions.classList.toggle("open", matches.length > 0);

  matches.forEach((option, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = index === highlightedSuggestion ? "suggestion active" : "suggestion";
    button.innerHTML = `<strong>${option.Ticker}</strong><span>${option.Sector || "Unknown"}</span>`;
    button.addEventListener("click", () => selectTicker(option.Ticker));
    suggestions.appendChild(button);
  });
}

function selectTicker(ticker) {
  activeTicker = ticker;
  searchInput.value = ticker;
  suggestions.classList.remove("open");
}

function addTickerToPortfolio() {
  const typed = searchInput.value.trim().toUpperCase();
  const exact = tickerOptions.find((option) => option.Ticker === typed);
  const fallback = getMatches(typed)[0];
  const ticker = activeTicker || exact?.Ticker || fallback?.Ticker;
  const quantity = Number(quantityInput.value);

  if (!ticker) {
    message.textContent = "Choose a valid ticker.";
    message.className = "error";
    return;
  }
  if (!quantity || quantity <= 0) {
    message.textContent = "Enter a positive quantity.";
    message.className = "error";
    return;
  }

  const portfolio = parsePortfolioText().filter((item) => item.ticker !== ticker);
  portfolio.push({ ticker, weight: quantity });
  writePortfolioText(portfolio);
  searchInput.value = "";
  activeTicker = "";
  renderSuggestions("");
  message.textContent = `${ticker} added`;
  message.className = "success";
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
  return Number(value).toFixed(4);
}

function renderBadge(category) {
  fields.category.textContent = category;
  fields.category.className = `badge ${String(category).toLowerCase()}`;
}

function renderSectors(sectors) {
  fields.sectors.innerHTML = "";
  sectors.forEach((item) => {
    const row = document.createElement("div");
    row.className = "sector-row";
    row.innerHTML = `
      <div class="sector-meta">
        <span>${item.sector}</span>
        <strong>${formatPercent(item.weight)}</strong>
      </div>
      <div class="bar"><span style="width: ${Math.max(2, item.weight * 100)}%"></span></div>
    `;
    fields.sectors.appendChild(row);
  });
}

function renderHoldings(holdings) {
  fields.holdings.innerHTML = "";
  const allocationByTicker = new Map(
    (window.currentAllocation || []).map((item) => [item.Ticker, item])
  );
  holdings.forEach((row) => {
    const allocation = allocationByTicker.get(row.Ticker) || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.Ticker}</td>
      <td>${Number(allocation.Quantity || 0).toLocaleString()}</td>
      <td>${formatCurrency(allocation.Price || 0)}</td>
      <td>${formatCurrency(allocation.MarketValue || 0)}</td>
      <td>${formatPercent(row.Weight)}</td>
      <td>${row.Sector}</td>
      <td>${formatNumber(row.Sigma)}</td>
      <td>${formatNumber(row.MDD)}</td>
      <td>${formatNumber(row.Beta)}</td>
    `;
    fields.holdings.appendChild(tr);
  });
}

function formatCurrency(value) {
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function renderPie(allocation) {
  if (!allocation.length) {
    fields.pie.style.background = "#edf1f6";
    fields.pie.innerHTML = "";
    return;
  }
  let cursor = 0;
  const slices = allocation.map((item, index) => {
    const start = cursor;
    const end = cursor + item.Weight * 100;
    cursor = end;
    return `${pieColors[index % pieColors.length]} ${start}% ${end}%`;
  });
  fields.pie.style.background = `conic-gradient(${slices.join(", ")})`;
  fields.pie.innerHTML = allocation
    .slice(0, 4)
    .map(
      (item, index) =>
        `<span><i style="background:${pieColors[index % pieColors.length]}"></i>${item.Ticker} ${formatPercent(item.Weight)}</span>`
    )
    .join("");
}

function renderReports(reports) {
  fields.reports.innerHTML = "";
  if (!reports.length) {
    fields.reports.innerHTML = `<p class="empty">No reports available.</p>`;
    return;
  }

  reports.forEach((report) => {
    const article = document.createElement("article");
    article.className = "report-card";
    const financial = report.financial.map((item) => `<li>${item}</li>`).join("");
    const news = report.news.length
      ? report.news
          .map(
            (item) => `
              <li>
                <strong>${item.headline}</strong>
                <span>${item.pubdate}</span>
                <p>${item.summary}</p>
              </li>
            `
          )
          .join("")
      : `<li><p>저장된 뉴스 요약이 없습니다.</p></li>`;

    article.innerHTML = `
      <header>
        <div>
          <h3>${report.ticker}</h3>
          <p>${report.sector} · ${formatPercent(report.weight)}</p>
        </div>
      </header>
      <section>
        <h4>Overview</h4>
        <p>${report.overview}</p>
      </section>
      <section>
        <h4>Financial Analysis</h4>
        <ul>${financial}</ul>
      </section>
      <section>
        <h4>News Impact</h4>
        <p>${report.news_impact}</p>
        <ul class="news-list">${news}</ul>
      </section>
      <section>
        <h4>Outlook</h4>
        <p>${report.outlook}</p>
      </section>
    `;
    fields.reports.appendChild(article);
  });
}

function renderResult(result) {
  window.currentAllocation = result.allocation || [];
  fields.srisk.textContent = formatNumber(result.srisk);
  fields.sigma.textContent = formatNumber(result.sigma);
  fields.mdd.textContent = formatNumber(result.mdd);
  fields.beta.textContent = formatNumber(result.beta);
  fields.hhi.textContent = formatNumber(result.hhi);
  fields.totalValue.textContent = formatCurrency(result.total_value || 0);
  fields.priceWarning.textContent = result.price_warning || "";
  renderBadge(result.category);
  renderPie(result.allocation || []);
  renderSectors(result.sectors);
  renderHoldings(result.holdings);
  renderReports(result.reports || initialReports || []);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "Analyzing...";
  message.className = "";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        portfolio: document.querySelector("#portfolio").value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Analysis failed.");
    }
    renderResult(payload);
    message.textContent = "Updated";
    message.className = "success";
  } catch (error) {
    message.textContent = error.message;
    message.className = "error";
  }
});

searchInput.addEventListener("input", () => {
  activeTicker = "";
  highlightedSuggestion = 0;
  renderSuggestions(searchInput.value);
});

searchInput.addEventListener("keydown", (event) => {
  const openSuggestions = [...suggestions.querySelectorAll(".suggestion")];
  if (!openSuggestions.length) return;

  if (event.key === "ArrowDown") {
    event.preventDefault();
    highlightedSuggestion = (highlightedSuggestion + 1) % openSuggestions.length;
    renderSuggestions(searchInput.value);
  }
  if (event.key === "ArrowUp") {
    event.preventDefault();
    highlightedSuggestion =
      (highlightedSuggestion - 1 + openSuggestions.length) % openSuggestions.length;
    renderSuggestions(searchInput.value);
  }
  if (event.key === "Enter") {
    event.preventDefault();
    openSuggestions[highlightedSuggestion]?.click();
  }
});

addTickerButton.addEventListener("click", addTickerToPortfolio);
portfolioInput.addEventListener("input", renderSelectedTickers);

renderResult(initialResult);
renderSelectedTickers();
