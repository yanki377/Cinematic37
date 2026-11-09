import re
from html import escape
from pathlib import Path

import streamlit as st

from recommender.artifacts import load_artifacts
from recommender.onboarding import recommend_from_ratings


PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Film Atlası",
    page_icon="🎬",
    layout="wide",
)

st.html(PROJECT_DIR / "assets" / "style.css")


@st.cache_resource
def get_bundle():
    return load_artifacts(
        PROJECT_DIR / "artifacts" / "final_model"
    )


bundle = get_bundle()

titles = (
    bundle["ranking"]
    .set_index("movieId")["title"]
    .to_dict()
)

catalog_ids = sorted(
    titles,
    key=lambda movie_id: titles[movie_id].casefold(),
)

with st.sidebar:
    st.html("""
        <div class="brand">
            <span class="brand-icon">F</span>
            Film Atlası
        </div>
    """)

    st.caption("Film zevkini tanıyalım.")
    st.divider()

    selected_ids = st.multiselect(
        "Sevdiğin filmleri seç",
        options=catalog_ids,
        format_func=lambda movie_id: titles[movie_id],
        max_selections=10,
        placeholder="Film adını yazarak ara...",
        key="favorite_movies",
    )

    st.caption(
        "3–10 film seç. Aşağıdaki puanları "
        "kendi tercihlerine göre değiştirebilirsin."
    )

    # Film seçimi formun dışında: yeni seçilen filmlerin
    # puan alanları hemen görünsün.
    with st.form("rating_form", border=False):
        movie_ratings = {}

        for movie_id in selected_ids:
            movie_ratings[movie_id] = st.slider(
                titles[movie_id],
                min_value=0.5,
                max_value=5.0,
                value=5.0,
                step=0.5,
                key=f"rating_{movie_id}",
            )

        n_recommendations = st.slider(
            "Kaç film önerelim?",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
        )

        submitted = st.form_submit_button(
            "Bana film öner →",
            type="primary",
            width="stretch",
            disabled=len(selected_ids) < 3,
        )

    st.divider()
    st.caption("Tercihlerin bu tarayıcı oturumunda tutulur.")

# Film seçimi değiştiyse eski seçimlerin listesini kaldır.
previous_result = st.session_state.get("personal_result")

if previous_result is not None:
    previous_ids = set(previous_result["ratings"])

    if previous_ids != set(selected_ids):
        del st.session_state["personal_result"]

if submitted:
    with st.spinner("Seçimlerinden film profilin oluşturuluyor..."):
        recommendations = recommend_from_ratings(
            bundle=bundle,
            movie_ratings=movie_ratings,
            n=n_recommendations,
        )

    st.session_state["personal_result"] = {
        "frame": recommendations,
        "ratings": movie_ratings.copy(),
    }

movie_count = f"{len(bundle['ranking']):,}".replace(",", ".")

st.html(f"""
    <section class="hero">
        <div class="eyebrow">Senin seçimlerin, senin film listen</div>

        <h1>Bu gece<br><span>ne izlesek?</span></h1>

        <p class="hero-description">
            Sevdiğin filmleri seç, puanlarını ver.
            Seçimlerinden oluşan kişisel profilinle
            sıradaki hikâyeni keşfet.
        </p>

        <div class="stats">
            <div class="stat">
                <strong>{movie_count}</strong>
                <span>film</span>
            </div>
            <div class="stat">
                <strong>3–10</strong>
                <span>film seçerek başla</span>
            </div>
            <div class="stat">
                <strong>{len(selected_ids)}</strong>
                <span>seçtiğin film</span>
            </div>
        </div>
    </section>
""")

if "personal_result" not in st.session_state:
    st.info(
        "Başlamak için sol panelden en az 3 film seç, "
        "puanlarını belirle ve “Bana film öner” düğmesine bas."
    )
    st.stop()

view = st.session_state["personal_result"]
recommendations = view["frame"]

st.html(f"""
    <div class="results-heading">
        <div>
            <div class="eyebrow">Film tercihlerinden yola çıkarak</div>
            <h2>Seçimlerine göre bir liste.</h2>
        </div>

        <div class="list-badge">
            {len(view["ratings"])} tercih ·
            {len(recommendations)} öneri
        </div>
    </div>
""")

with st.expander("Bu liste için verdiğin puanlar"):
    for movie_id, rating in view["ratings"].items():
        st.write(f"{titles[movie_id]} — {rating:.1f}/5")

palette = [
    "#72484d",
    "#315967",
    "#574974",
    "#7a5b32",
    "#395f50",
    "#405e83",
]

cards = []

for position, (_, row) in enumerate(
    recommendations.iterrows(),
    start=1,
):
    original_title = str(row["title"])

    year_match = re.search(
        r"\s*\((\d{4})\)\s*$",
        original_title,
    )

    year = year_match.group(1) if year_match else "—"

    title = (
        original_title[:year_match.start()].strip()
        if year_match
        else original_title
    )

    title = re.sub(
        r"^(.+), (The|A|An)$",
        r"\2 \1",
        title,
    )

    monogram = "".join(
        word[0] for word in title.split()[:2]
    ).upper()

    accent = palette[(position - 1) % len(palette)]
    rating = float(row["mean_rating"])
    count = f"{int(row['rating_count']):,}".replace(",", ".")

    cards.append(f"""
        <article class="movie-card">
            <div class="movie-art" style="--accent: {accent};">
                <span class="rank-badge">#{position:02d}</span>
                <span class="monogram" aria-hidden="true">
                    {escape(monogram)}
                </span>
                <span class="year">{year}</span>
            </div>

            <div class="movie-body">
                <h3 class="movie-title" title="{escape(title, quote=True)}">
                    {escape(title)}
                </h3>

                <div class="movie-footer">
                    <span class="rating">
                        ★ {rating:.2f} <small>/ 5</small>
                    </span>
                    <span class="rating-count">
                        {count} puanlama
                    </span>
                </div>
            </div>
        </article>
    """)

st.html(
    '<div class="movie-grid">'
    + "".join(cards)
    + "</div>"
)

st.caption(
    "Kartlardaki puanlar veri setindeki ortalama film puanlarıdır. "
    "Seçtiğin filmler önerilere dahil edilmez."
)