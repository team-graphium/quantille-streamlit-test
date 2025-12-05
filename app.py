import os
from typing import Dict

import streamlit as st
from openai import OpenAI

from persona_engine import (
    PersonaEngine,
    FACTOR_NAME_MAP,
    STYLE_SENTENCES,
    generate_persona_prompt_for_profile,
)

# ---------- KONFIG ----------

ARTIFACTS_PATH_DEFAULT = "./persona_artifacts_v1.npz"

FACTOR_CODES = [
    "REF", "FOG", "SZAM", "GYAK",
    "VÁL", "REN", "KUT", "SZOC",
    "MŰV", "EMO", "TEM", "RUG",
    "CSAP", "KAP", "LÁT", "HAT",
]

STYLE_CODES = list(STYLE_SENTENCES.keys())  # VISSZ, URA, KER, KIE, ELK, ALK, VER, KOM, MEG


# ---------- HELPER: ENGINE & OPENAI CACHE ----------

@st.cache_resource
def load_engine(artifacts_path: str) -> PersonaEngine:
    return PersonaEngine(
        mode="artifacts",
        artifacts_path=artifacts_path,
    )


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


# ---------- APP ----------

def main():
    st.set_page_config(
        page_title="Persona Engine – Személyiség-leírás demó",
        layout="wide",
    )

    st.title("🧠 Persona Engine – Személyiség-leírás demó")

    st.markdown(
        """
        Add meg a faktorok 1–8 közötti értékeit, az app pedig egy LLM segítségével
        **rövid személyiség-összefoglalót** készít.
        """
    )

    # ---- Sidebar: beállítások ----
    st.sidebar.header("⚙️ Beállítások")

    artifacts_path = st.sidebar.text_input(
        "Artifacts fájl elérési útja",
        value=ARTIFACTS_PATH_DEFAULT,
    )

    work_env = st.sidebar.text_input(
        "Munkahelyi környezet (work_env)",
        value="iroda",
    )

    st.sidebar.subheader("🔑 OpenAI API")
    openai_api_key = st.sidebar.text_input(
        "OPENAI_API_KEY",
        type="password",
        help="Add meg az OpenAI API kulcsot (GPT-4/5 modellekhez).",
    )
    model_name = st.sidebar.text_input(
        "Model neve",
        value="gpt-5.1",  # vagy "gpt-4.1", "gpt-5.1", stb.
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("A módosítások után kattints a „Leírás generálása” gombra.")

    # ---- Fő layout: bal – input mezők, jobb – output ----
    col_left, col_right = st.columns([1.1, 1.3])

    # ---------- PROFIL INPUT: sima boxok ----------
    with col_left:
        st.header("📊 Faktorértékek (1–8)")

        st.markdown("### Alap faktorok")
        profile_levels: Dict[str, float] = {}

        for code in FACTOR_CODES:
            display_name = FACTOR_NAME_MAP.get(code, code)
            # minimális default logika, csak hogy legyen valami életszerű induló érték
            default_val = 4
            if code in ["GYAK", "REN"]:
                default_val = 6

            val = st.number_input(
                f"{display_name} ({code})",
                min_value=1,
                max_value=8,
                value=default_val,
                step=1,
            )
            profile_levels[code] = float(val)

        st.markdown("### Stílusfaktorok")
        st.caption("5–8 között jelenik meg a leírásban kiegészítő mondatként.")

        style_levels: Dict[str, float] = {}
        for code in STYLE_CODES:
            display_name = FACTOR_NAME_MAP.get(code, code)
            val = st.number_input(
                f"{display_name} ({code})",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
            )
            style_levels[code] = float(val)

        generate_btn = st.button("🚀 Leírás generálása", type="primary")

    # ---------- OUTPUT OLDAL ----------
    with col_right:
        st.header("📝 Generált leírás")

        if generate_btn:
            # --- guardok ---
            if not artifacts_path:
                st.error("Add meg az artifacts fájl elérési útját!")
                return

            if not openai_api_key:
                st.error("Add meg az OpenAI API kulcsot a sidebarban!")
                return

            # Engine betöltés
            try:
                engine = load_engine(artifacts_path)
            except Exception as e:
                st.error(f"Nem sikerült betölteni az artifacts fájlt: {e}")
                return

            # Persona prompt + snippets
            with st.spinner("Persona prompt generálása..."):
                try:
                    prompt, factor_snippets = generate_persona_prompt_for_profile(
                        engine,
                        profile_levels=profile_levels,
                        style_levels=style_levels,
                        work_env=work_env,
                        factor_name_map=FACTOR_NAME_MAP,
                    )
                except Exception as e:
                    st.error(f"Hiba a persona prompt generálásánál: {e}")
                    return

            # LLM hívás
            with st.spinner("LLM válasz generálása..."):
                try:
                    client = get_openai_client(openai_api_key)
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )
                    persona_text = completion.choices[0].message.content
                except Exception as e:
                    st.error(f"Hiba az LLM hívás közben: {e}")
                    st.subheader("Debug – prompt")
                    st.code(prompt)
                    return

            # --- Eredmény megjelenítése ---
            st.subheader("📄 Személyiség-összefoglaló")
            st.write(persona_text)

            with st.expander("🔍 Debug: LLM-nek adott prompt"):
                st.code(prompt)

            with st.expander("🧩 Debug: faktor-szintek & snippetek"):
                st.json(factor_snippets)

        else:
            st.info("Töltsd ki a faktor mezőket bal oldalon, add meg az API kulcsot, majd kattints a „Leírás generálása” gombra.")


if __name__ == "__main__":
    main()
