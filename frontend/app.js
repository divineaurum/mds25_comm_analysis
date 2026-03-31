// BASE_URL: пустая строка при запуске через FastAPI (relative URLs),
// полный адрес — при открытии файла напрямую (file://)
const BASE_URL =
  window.location.protocol === "file:"
    ? "http://localhost:8000"
    : "";

const productSelect = document.getElementById("product-select");
const dateFrom = document.getElementById("date-from");
const dateTo = document.getElementById("date-to");
const statsInfo = document.getElementById("stats-info");
const analyzeBtn = document.getElementById("analyze-btn");
const loadingEl = document.getElementById("loading");
const resultsEl = document.getElementById("results");
const topicsGrid = document.getElementById("topics-grid");
const resultTitle = document.getElementById("results-title");
const resultsMeta = document.getElementById("results-meta");
const errorMsg = document.getElementById("error-msg");

function showLoading() {
  loadingEl.hidden = false;
  resultsEl.hidden = true;
  errorMsg.hidden = true;
}

function hideLoading() {
  loadingEl.hidden = true;
}

function showError(text) {
  errorMsg.textContent = text;
  errorMsg.hidden = false;
  resultsEl.hidden = true;
}

async function loadProducts() {
  try {
    const response = await fetch(`${BASE_URL}/products`);
    if (!response.ok) {
      throw new Error(`Ошибка ${response.status}: не удалось получить список продуктов`);
    }
    const products = await response.json();
    productSelect.innerHTML = '<option value="">Выберите продукт...</option>';
    products.forEach(p => {
      const option = document.createElement("option");
      option.value = p.id;
      option.textContent = p.name;
      productSelect.appendChild(option);
    });
    productSelect.disabled = false;
  } catch (err) {
    productSelect.innerHTML = '<option value="">Ошибка загрузки</option>';
    showError(`Не удалось подключиться к серверу: ${err.message}`);
  }
}

async function loadProductStats(productId) {
  statsInfo.hidden = true;
  analyzeBtn.disabled = true;
  if (!productId) return;

  try {
    const response = await fetch(`${BASE_URL}/products/${productId}/stats`);
    if (!response.ok) throw new Error(response.statusText);
    const stats = await response.json();

    dateFrom.min = stats.date_min;
    dateFrom.max = stats.date_max;
    dateTo.min = stats.date_min;
    dateTo.max = stats.date_max;
    dateFrom.value = stats.date_min;
    dateTo.value = stats.date_max;

    statsInfo.textContent =
      `Доступно ${stats.total_comments.toLocaleString("ru")} комментариев ` +
      `с ${formatDate(stats.date_min)} по ${formatDate(stats.date_max)}`;
    statsInfo.hidden = false;
    analyzeBtn.disabled = false;
  } catch (err) {
    showError(`Ошибка при загрузке статистики: ${err.message}`);
  }
}

async function runAnalysis() {
  const productId = productSelect.value;
  if (!productId) return;

  // Блокируем на время анализа (1–3 минуты) — дублирующий запрос сломает результаты
  analyzeBtn.disabled = true;
  showLoading();

  try {
    const response = await fetch(`${BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_id: productId,
        date_from: dateFrom.value,
        date_to: dateTo.value,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || response.statusText);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError(`Ошибка анализа: ${err.message}`);
  } finally {
    hideLoading();
    analyzeBtn.disabled = false;
  }
}

function renderResults(data) {
  resultTitle.textContent = data.product;
  resultsMeta.textContent =
    `${data.period} · ${data.total_comments.toLocaleString("ru")} комментариев`;

  topicsGrid.innerHTML = "";

  if (!data.topics || data.topics.length === 0) {
    topicsGrid.innerHTML =
      '<p style="color: var(--color-text-muted); text-align: center; padding: 24px">' +
      "Темы не найдены. Попробуйте расширить период.</p>";
  } else {
    data.topics.forEach(topic => {
      topicsGrid.appendChild(buildTopicCard(topic));
    });
  }

  resultsEl.hidden = false;
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildTopicCard(topic) {
  const severityMap = {
    "высокая": { cls: "high", label: "Высокая критичность" },
    "средняя": { cls: "medium", label: "Средняя критичность" },
    "низкая": { cls: "low", label: "Низкая критичность" },
  };
  const sev = severityMap[topic.severity] || { cls: "medium", label: topic.severity };

  const card = document.createElement("div");
  card.className = `topic-card topic-card--${sev.cls}`;

  const header = document.createElement("div");
  header.className = "topic-card__header";

  const title = document.createElement("h3");
  title.className = "topic-card__title";
  title.textContent = topic.topic_name;

  const badges = document.createElement("div");
  badges.className = "topic-card__badges";

  badges.appendChild(makeBadge(sev.label, `badge--${sev.cls}`));

  if (topic.is_new_topic) {
    badges.appendChild(makeBadge("Новая тема", "badge--new"));
  }

  badges.appendChild(
    makeBadge(`${topic.count} (${topic.percentage.toFixed(1)}%)`, "badge--count")
  );

  header.appendChild(title);
  header.appendChild(badges);
  card.appendChild(header);

  const bar = document.createElement("div");
  bar.className = "progress-bar";
  const fill = document.createElement("div");
  fill.className = "progress-bar__fill";
  fill.style.width = `${Math.min(topic.percentage, 100)}%`;
  bar.appendChild(fill);
  card.appendChild(bar);

  const desc = document.createElement("p");
  desc.className = "topic-card__description";
  desc.textContent = topic.description;
  card.appendChild(desc);

  const catEl = document.createElement("p");
  catEl.className = "topic-card__category";
  catEl.innerHTML =
    `<strong>Категория:</strong> ${escapeHtml(topic.existing_category)}`;
  card.appendChild(catEl);

  if (topic.example_comments && topic.example_comments.length > 0) {
    const details = document.createElement("details");
    details.className = "examples-details";

    const summary = document.createElement("summary");
    summary.textContent =
      `Примеры комментариев (${topic.example_comments.length})`;
    details.appendChild(summary);

    const ul = document.createElement("ul");
    ul.className = "examples-list";
    topic.example_comments.forEach(comment => {
      const li = document.createElement("li");
      li.textContent = comment;
      ul.appendChild(li);
    });
    details.appendChild(ul);
    card.appendChild(details);
  }

  return card;
}

function makeBadge(text, extraClass) {
  const span = document.createElement("span");
  span.className = `badge ${extraClass}`;
  span.textContent = text;
  return span;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatDate(isoDate) {
  if (!isoDate) return "";
  const [y, m, d] = isoDate.split("-");
  return `${d}.${m}.${y}`;
}

productSelect.addEventListener("change", () => {
  loadProductStats(productSelect.value);
});

analyzeBtn.addEventListener("click", runAnalysis);

document.addEventListener("DOMContentLoaded", loadProducts);
