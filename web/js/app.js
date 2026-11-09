const grid = document.querySelector("#movie-grid");
const statusLine = document.querySelector("#catalog-status");
const search = document.querySelector("#movie-search");
const detailsDialog = document.querySelector("#movie-dialog");
const ratingDialog = document.querySelector("#rating-dialog");
const selectionMessage = document.querySelector("#selection-message");
const selectionBar = document.querySelector("#selection-bar");
const rateButton = document.querySelector("#rate-selected");
const resultsSection = document.querySelector("#recommendations");
const resultGrid = document.querySelector("#recommendation-grid");
const resultStatus = document.querySelector("#recommendation-status");
const loadingPanel = document.querySelector("#recommendation-loading");
let movies = [];
let catalogSize = 0;
let searchTimer = null;
let searchController = null;
let searchVersion = 0;
const movieById = new Map();
const selected = new Map();
let currentDetail = null;
let loading = false;
let requestVersion = 0;
let activeRequest = null;

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}
function image(movie, decorative = false) {
  const img = document.createElement("img");
  img.alt = decorative ? "" : movie.title;
  if (movie.poster_url?.startsWith("https://image.tmdb.org/")) img.src = movie.poster_url;
  else img.hidden = true;
  img.addEventListener("error", () => img.remove(), { once: true });
  return img;
}
function year(movie) {
  return movie.release_date?.slice(0, 4) || String(movie.year || "—");
}
function rating(movie) {
  return typeof movie.tmdb_rating === "number"
    ? movie.tmdb_rating.toLocaleString("tr-TR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })
    : "—";
}
function normalize(text) {
  return String(text ?? "").toLocaleLowerCase("tr-TR")
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/ı/g, "i");
}
function persistSelections() {
  try { sessionStorage.setItem("film-atlasi-preferences", JSON.stringify([...selected])); } catch (_) {}
}
async function restoreSelections() {
  try {
    const stored = JSON.parse(sessionStorage.getItem("film-atlasi-preferences") || "[]");
    if (!Array.isArray(stored)) return;
    const valid = stored.slice(0, 10).filter(item =>
      Array.isArray(item) && item.length === 2 &&
      Number.isInteger(item[0]) && item[0] > 0 &&
      Number.isInteger(item[1]) && item[1] >= 1 && item[1] <= 5);
    const restored = await Promise.all(valid.map(async ([id, value]) => {
      let movie = movieById.get(id);
      if (!movie) {
        try {
          const response = await fetch("/api/movies/" + id);
          if (!response.ok) return null;
          movie = await response.json();
          if (movie.movie_id !== id) return null;
        } catch (_) { return null; }
      }
      return { id, value, movie };
    }));
    for (const item of restored) {
      if (!item) continue;
      movieById.set(item.id, item.movie);
      selected.set(item.id, item.value);
    }
    persistSelections();
  } catch (_) {}
}
function invalidateResults() {
  requestVersion += 1;
  activeRequest?.abort();
  activeRequest = null;
  loading = false;
  resultsSection.hidden = true;
  loadingPanel.hidden = true;
  resultGrid.replaceChildren();
}
function updateSelectionUI() {
  document.body.classList.toggle("has-selection", selected.size > 0);
  selectionBar.hidden = selected.size === 0;
  document.querySelector("#selected-count").textContent = selected.size + " film seçtin";
  document.querySelector("#selected-help").textContent = selected.size < 3
    ? (3 - selected.size) + " film daha seçerek başlayabilirsin." : "Puanlamaya hazırsın.";
  rateButton.disabled = selected.size < 3 || loading;
  rateButton.textContent = loading ? "Hazırlanıyor…" : "Puanla ve öneri al →";
  document.querySelector("#edit-preferences").disabled = loading;
  const thumbs = document.querySelector("#selection-thumbs");
  thumbs.replaceChildren();
  for (const id of [...selected.keys()].slice(0, 5)) thumbs.append(image(movieById.get(id), true));
  for (const button of document.querySelectorAll("[data-select-id]")) {
    const id = Number(button.dataset.selectId);
    const chosen = selected.has(id);
    const movie = movieById.get(id);
    button.setAttribute("aria-pressed", String(chosen));
    button.setAttribute("aria-label", movie.title + (chosen ? " seçimden çıkar" : " seç"));
    button.textContent = chosen ? "✓" : "+";
    button.closest(".movie-card").classList.toggle("is-selected", chosen);
  }
  if (currentDetail) {
    const chosen = selected.has(currentDetail.movie_id);
    const button = document.querySelector("#detail-select");
    button.textContent = chosen ? "Seçimden çıkar" : "İzledim, filmi seç";
    button.setAttribute("aria-pressed", String(chosen));
  }
}
function toggleMovie(movie) {
  if (selected.has(movie.movie_id)) selected.delete(movie.movie_id);
  else {
    if (selected.size >= 10) {
      selectionMessage.textContent = "En fazla 10 film seçebilirsin. Önce bir filmi seçiminden çıkar.";
      return;
    }
    selected.set(movie.movie_id, 5);
  }
  selectionMessage.textContent = "";
  invalidateResults();
  persistSelections();
  updateSelectionUI();
}
function openDetails(movie) {
  currentDetail = movie;
  const poster = document.querySelector("#detail-poster");
  poster.hidden = !movie.poster_url;
  if (movie.poster_url) poster.src = movie.poster_url;
  else poster.removeAttribute("src");
  poster.alt = movie.title;
  document.querySelector("#detail-title").textContent = movie.title;
  document.querySelector("#detail-meta").textContent = [
    year(movie), movie.runtime ? movie.runtime + " dk" : null,
    movie.tmdb_rating != null ? "TMDB ★ " + rating(movie) : null,
  ].filter(Boolean).join(" · ");
  document.querySelector("#detail-genres").textContent = (movie.genres || []).join(" / ");
  document.querySelector("#detail-overview").textContent = movie.overview || "Bu film için açıklama şu anda yüklenemiyor.";
  const source = document.querySelector("#detail-source");
  source.hidden = !movie.tmdb_id;
  if (movie.tmdb_id) source.href = "https://www.themoviedb.org/movie/" + movie.tmdb_id;
  else source.removeAttribute("href");
  updateSelectionUI();
  detailsDialog.showModal();
}
function createMovieCard(movie, selectable) {
  const card = element("article", "movie-card");
  const cover = element("div", "movie-cover");
  const open = element("button", "poster-open");
  open.type = "button";
  open.setAttribute("aria-label", movie.title + " detaylarını aç");
  open.append(element("span", "poster-fallback", "film atlası"));
  const poster = image(movie);
  poster.loading = "lazy";
  open.append(poster);
  open.addEventListener("click", () => openDetails(movie));
  cover.append(open);
  if (selectable) {
    const select = element("button", "select-movie", "+");
    select.type = "button";
    select.dataset.selectId = movie.movie_id;
    select.addEventListener("click", () => toggleMovie(movie));
    cover.append(select);
  }
  const title = element("h3", "movie-title", movie.title);
  title.title = movie.title;
  const meta = element("div", "movie-meta");
  meta.append(element("span", "", year(movie)));
  if (movie.tmdb_rating != null) meta.append(element("span", "movie-rating", "★ " + rating(movie)));
  card.append(cover, title, meta);
  return card;
}
function renderCatalog(items) {
  grid.replaceChildren(...items.map(movie => createMovieCard(movie, true)));
  statusLine.textContent = items.length
    ? search.value.trim().length >= 2
      ? items.length + " eşleşme · + ile seç veya detayını aç." +
        (items.length >= 12 ? " Aramalarda en fazla 12 sonuç gösterilir." : "")
      : catalogSize.toLocaleString("tr-TR") + " film içinde ara · Öne çıkan " + items.length + " film."
    : "Eşleşme bulunamadı. Filmin özgün adını veya farklı bir türü deneyebilirsin.";
  updateSelectionUI();
}
function renderHero(items) {
  document.querySelector("#poster-wall").replaceChildren(...items.slice(0, 24).map(movie => image(movie, true)));
  const preview = document.querySelector("#hero-preview");
  preview.replaceChildren();
  for (const movie of items.slice(0, 3)) {
    const row = element("div", "preview-movie");
    const copy = element("div");
    const meta = element("p", "", year(movie) + " · ");
    meta.append(element("span", "preview-rating", "★ " + rating(movie)));
    copy.append(element("h3", "", movie.title), meta);
    row.append(image(movie), copy);
    preview.append(row);
  }
}
function openRatings() {
  if (selected.size < 3 || loading) return;
  const rows = document.querySelector("#rating-rows");
  rows.replaceChildren();
  for (const [id, value] of selected) {
    const movie = movieById.get(id);
    const row = element("div", "rating-row");
    const copy = element("div", "rating-copy");
    const score = element("p", "", value + " / 5");
    score.setAttribute("aria-live", "polite");
    copy.append(element("h3", "", movie.title), score);
    const group = element("div", "star-group");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", movie.title + " için puanın");
    for (let star = 1; star <= 5; star += 1) {
      const button = element("button", "star-button", "★");
      button.type = "button";
      button.setAttribute("aria-label", star + " yıldız");
      button.setAttribute("aria-pressed", String(star === value));
      button.classList.toggle("is-lit", star <= value);
      button.addEventListener("click", () => {
        selected.set(id, star);
        score.textContent = star + " / 5";
        for (const [index, sibling] of [...group.children].entries()) {
          sibling.classList.toggle("is-lit", index < star);
          sibling.setAttribute("aria-pressed", String(index + 1 === star));
        }
        invalidateResults();
        persistSelections();
        updateSelectionUI();
      });
      group.append(button);
    }
    row.append(image(movie), copy, group);
    rows.append(row);
  }
  ratingDialog.showModal();
}
function scrollToResults() {
  resultsSection.scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    block: "start",
  });
}
async function requestRecommendations() {
  const version = ++requestVersion;
  const controller = new AbortController();
  activeRequest = controller;
  loading = true;
  updateSelectionUI();
  resultsSection.hidden = false;
  loadingPanel.hidden = false;
  resultGrid.replaceChildren();
  resultStatus.classList.remove("is-error");
  resultStatus.textContent = "Verdiğin puanlarla film listen hazırlanıyor.";
  scrollToResults();
  try {
    const response = await fetch("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        ratings: [...selected].map(([movie_id, rating]) => ({ movie_id, rating })),
        n: 10,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string"
      ? data.detail : "Seçimlerini ve puanlarını kontrol edip tekrar dene.");
    if (version !== requestVersion) return;
    if (!Array.isArray(data.movies) || !data.movies.length) {
      throw new Error("Bu seçimlerle öneri oluşturulamadı. Puanlarını düzenleyip tekrar dene.");
    }
    for (const movie of data.movies) movieById.set(movie.movie_id, movie);
    resultGrid.replaceChildren(...data.movies.map(movie => createMovieCard(movie, false)));
    resultStatus.textContent = data.input_count + " film için verdiğin puanlardan " +
      data.movies.length + " öneri oluşturuldu. Seçtiğin filmler listeden çıkarıldı." +
      (data.metadata_missing ? " Bazı afişler şu anda yüklenemiyor." : "");
  } catch (error) {
    if (version !== requestVersion || error.name === "AbortError") return;
    resultStatus.classList.add("is-error");
    resultStatus.textContent = error instanceof TypeError
      ? "Öneriler şu anda yüklenemiyor. Biraz sonra tekrar dene." : error.message;
  } finally {
    if (version === requestVersion) {
      loading = false;
      activeRequest = null;
      loadingPanel.hidden = true;
      updateSelectionUI();
    }
  }
}
search.maxLength = 100;
search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchController?.abort();
  searchController = null;
  const version = ++searchVersion;
  const query = search.value.trim();
  grid.removeAttribute("aria-busy");
  if (!query || normalize(query).length < 2) {
    renderCatalog(movies);
    if (query) statusLine.textContent = "Aramak için en az 2 karakter yaz.";
    return;
  }
  grid.replaceChildren();
  statusLine.textContent = "Filmler aranıyor…";
  searchTimer = setTimeout(async () => {
    const controller = new AbortController();
    searchController = controller;
    grid.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(
        "/api/movies?query=" + encodeURIComponent(query) + "&limit=12",
        { signal: controller.signal },
      );
      if (!response.ok) throw new Error("Arama isteği başarısız");
      const data = await response.json();
      if (version !== searchVersion) return;
      if (!Array.isArray(data)) throw new Error("Arama yanıtı geçersiz");
      for (const movie of data) movieById.set(movie.movie_id, movie);
      renderCatalog(data);
    } catch (error) {
      if (version !== searchVersion || error.name === "AbortError") return;
      grid.replaceChildren();
      statusLine.textContent = "Arama şu anda tamamlanamadı. Tekrar dene.";
    } finally {
      if (version === searchVersion) {
        searchController = null;
        grid.removeAttribute("aria-busy");
      }
    }
  }, 450);
});
document.querySelector("#detail-select").addEventListener("click", () => {
  if (currentDetail) toggleMovie(currentDetail);
});
document.querySelector("#clear-selection").addEventListener("click", () => {
  selected.clear();
  selectionMessage.textContent = "";
  invalidateResults();
  persistSelections();
  updateSelectionUI();
});
rateButton.addEventListener("click", openRatings);
document.querySelector("#edit-preferences").addEventListener("click", openRatings);
document.querySelector("#rating-form").addEventListener("submit", event => {
  event.preventDefault();
  ratingDialog.close();
  requestRecommendations();
});
for (const [dialog, buttonId] of [
  [detailsDialog, "#close-dialog"], [ratingDialog, "#close-ratings"],
]) {
  document.querySelector(buttonId).addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
}
async function loadCatalog() {
  try {
    const response = await fetch("/api/movies");
    if (!response.ok) throw new Error("Katalog isteği başarısız");
    const data = await response.json();
    if (!Array.isArray(data) || !data.length) throw new Error("Katalog boş");
    movies = data;
    const size = Number(response.headers.get("X-Catalog-Size"));
    catalogSize = Number.isInteger(size) && size > 0 ? size : movies.length;
    search.placeholder = catalogSize.toLocaleString("tr-TR") + " filmde ad veya tür ara…";
    for (const movie of movies) movieById.set(movie.movie_id, movie);
    await restoreSelections();
    renderHero(movies);
    renderCatalog(movies);
  } catch (error) {
    statusLine.textContent = "Film kataloğu yüklenemedi. Sayfayı yenileyerek tekrar dene.";
    document.querySelector("#hero-preview").textContent = "Katalog şu anda yüklenemiyor.";
    console.error(error);
  }
}
loadCatalog();
