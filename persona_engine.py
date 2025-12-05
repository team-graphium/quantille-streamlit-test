# persona_engine.py
# =========================
#   KIEGÉSZÍTŐ STÍLUSFAKTOROK – STATIKUS MONDATOK (csak 5–8 értékhez)
# =========================
# =========================
#   KIEGÉSZÍTŐ STÍLUSFAKTOROK – STATIKUS MONDATOK (csak 5–8 értékhez)
# =========================

FACTOR_NAME_MAP = {
    "REF": "Reflexió",
    "FOG": "Fogalmi gondolkodás",
    "SZAM": "Gondolkodás számokban",
    "GYAK": "Gyakorlati",
    "VÁL": "Vállalkozói",
    "REN": "Rendszerező",
    "KUT": "Kutatói",
    "SZOC": "Szociális",
    "MŰV": "Művészi",
    "EMO": "Emocionalitás",
    "TEM": "Tempó",
    "RUG": "Rugalmasság",
    "CSAP": "Csapatmunka",
    "KAP": "Kapcsolódás",
    "LÁT": "Látásmód",
    "HAT": "Határozottság",
    # stílusfaktorok display nevei
    "VISSZ": "Visszahúzódó kommunikáció",
    "URA": "Uralkodó kommunikáció",
    "KER": "Kerülőutas kommunikáció",
    "KIE": "Kiegyensúlyozott kommunikáció",
    "ELK": "Konfliktuskerülés",
    "ALK": "Alkalmazkodó konfliktuskezelés",
    "VER": "Versengő konfliktuskezelés",
    "KOM": "Kompromisszumkereső stílus",
    "MEG": "Megoldásfókuszú konfliktuskezelés",
}

STYLE_SENTENCES = {
    # Kommunikációs stílus
    "VISSZ": {  # visszahúzódó
        5: "Kommunikációjában jól érzékelhető a visszafogottság: inkább figyel és kérdez, ritkábban hoz erős, egyértelmű állításokat.",
        6: "Kommunikációja kifejezetten visszahúzódó; jellemzően megfigyelő pozícióból van jelen, óvatosan szólal meg és kerüli a hangsúlyos szerepet.",
        7: "Kommunikációjában dominánsan megjelenik a csendes, tartózkodó jelenlét: ritkán vállal kezdeményező, véleményformáló szerepet.",
        8: "Kommunikációja erősen visszahúzódó; szinte végig megfigyelőként marad jelen, nagyon ritkán foglal nyíltan állást vagy vállal látható szerepet.",
    },
    "URA": {  # uralkodó
        5: "Kommunikációjában gyakran határozott és erős jelenlétű, időnként átveszi a beszélgetések irányítását.",
        6: "Kommunikációja jól érzékelhetően domináns: markáns megfogalmazásokat használ, és könnyen irányító pozícióba kerül a beszélgetésekben.",
        7: "Kommunikációjában kifejezetten uralkodó stílus jelenik meg; határozottan viszi a beszélgetéseket és erősen formálja a döntési helyzeteket.",
        8: "Kommunikációja nagyon domináns, erősen alakítja a csoportdinamikát, és rendszerint ő az, aki meghatározza a beszélgetések irányát.",
    },
    "KER": {  # kerülőutas
        5: "Kommunikációjában gyakran megjelennek finomabb, indirekt megfogalmazások, időnként célozgatva jelzi a véleményét.",
        6: "Kommunikációjára jellemző a burkolt, kerülőutas stílus: üzeneteit sokszor diplomatikusan becsomagolva fogalmazza meg.",
        7: "Kommunikációjában erősen jelen van a kerülőutas működés: indirekt jelzések, célozgatás és taktikusan adagolt információk kísérik a mondandóját.",
        8: "Kommunikációja kifejezetten kerülőutas; nyílt kimondás helyett gyakran célozgat, hallgatással vagy finom jelzésekkel fejezi ki az álláspontját.",
    },
    "KIE": {  # kiegyensúlyozott
        5: "Kommunikációjában érzékelhető az asszertív, kiegyensúlyozott hang: igyekszik tényszerű, nyugodt módon érvelni.",
        6: "Kommunikációja alapvetően asszertív és nyugodt; tiszteletteljes, konstruktív hangnemben fejezi ki a véleményét és bevonja a másik felet.",
        7: "Kommunikációjában erősen jelen van az asszertív, kiegyensúlyozott működés: tárgyszerű, ítélkezésmentes, és tudatosan teret ad a párbeszédnek.",
        8: "Kommunikációja kimondottan kiegyensúlyozott és érett; magabiztosan, nyugodtan érvel, miközben következetesen figyel a másik fél szempontjaira is.",
    },

    # Konfliktuskezelési stílus
    "ELK": {  # elkerülés
        5: "Konfliktushelyzetekben gyakran halogatja a nyílt szembenézést, és inkább kitér a feszültséget okozó témák elől.",
        6: "Konfliktuskezelésében erősen jelen van az elkerülés: sokszor kivonul a nehezebb helyzetekből, vagy igyekszik nem tudomást venni a problémáról.",
        7: "Konfliktusokban kifejezetten kerüli a konfrontációt; ritkán vállal nyílt állásfoglalást, inkább elhúzza vagy elengedi a helyzeteket.",
        8: "Konfliktuskezelésében nagyon markáns az elkerülés: a nehéz helyzeteket rendszerint elodázza vagy kikerüli, így a feszültségek könnyen bent maradnak a rendszerben.",
    },
    "ALK": {  # alkalmazkodás
        5: "Konfliktushelyzetekben hajlamos engedni a saját szempontjaiból a kapcsolat megőrzése érdekében.",
        6: "Konfliktuskezelésében erősen jelen van az alkalmazkodás: sokszor a másik megoldását fogadja el, hogy a viszony harmonikus maradjon.",
        7: "Konfliktusokban kifejezetten kapcsolatvédő; gyakran háttérbe helyezi saját érdekeit, csak hogy elkerülje a tartós feszültséget.",
        8: "Konfliktuskezelésében nagyon erős az önfeladó alkalmazkodás: rendszerint lemond a saját szempontjairól, ha ezzel békét tud fenntartani.",
    },
    "VER": {  # versengés
        5: "Konfliktushelyzetekben határozottan képviseli a saját álláspontját, és nem riad vissza attól, hogy ezt ütköztesse másokéval.",
        6: "Konfliktuskezelésében erősen jelen van a versengő stílus: aktívan törekszik a saját érdekei érvényesítésére.",
        7: "Konfliktusokban kifejezetten versengő módon működik; erősen nyomja a saját megoldását, és nehezen enged a pozíciójából.",
        8: "Konfliktushelyzetekben nagyon domináns, versengő módon jelenik meg; markánsan a saját érdekei mentén mozgatja a helyzeteket.",
    },
    "KOM": {  # kompromisszumkeresés
        5: "Konfliktushelyzetekben törekszik arra, hogy mindkét fél számára elfogadható köztes megoldást találjon.",
        6: "Konfliktuskezelésében erősen jelen van a kompromisszumkeresés: kész engedni bizonyos pontokon, ha a másik fél is tesz lépéseket.",
        7: "Konfliktusokban tudatosan a középutat keresi; alkuképes, és figyel arra, hogy minden fél kapjon valamit a megoldásból.",
        8: "Konfliktuskezelésében nagyon erős a kompromisszumorientált működés: strukturáltan keresi a win-win megoldásokat és az egyensúlyt a felek érdekei között.",
    },
    "MEG": {  # megoldásfókusz
        5: "Konfliktushelyzetekben jellemző rá, hogy igyekszik a tényekre és a lehetséges megoldásokra terelni a figyelmet.",
        6: "Konfliktuskezelésében erősen jelen van a megoldásfókusz: elemző, jövőorientált módon keresi a továbblépési lehetőségeket.",
        7: "Konfliktusokban kifejezetten megoldásorientált; nem ragad le a hibakeresésnél, inkább alternatívákat épít a felek számára.",
        8: "Konfliktuskezelésében nagyon erős, érett megoldásfókusz működik: tényalapú, nyitott, win-win szemlélettel dolgozik még feszült helyzetekben is.",
    },
}


# persona_engine.py

from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from inference import (
    compute_factor_centroids,
    compute_factor_pos_calibration,
    compute_factor_scores_for_texts,
    sample_texts_for_profile_simple,
)

FACTORS_FILE = "./data/factors_regenerated.jsonl"

EngineMode = Literal["online", "artifacts"]


class PersonaEngine:
    """
    Kétmódú engine:

    - mode="online":
        - Betölti a SentenceTransformer modellt,
        - Centroidokat számol,
        - POS kalibrációt számol,
        - Anchor szövegeket betölti,
        - Anchor score-okat kiszámolja.

    - mode="artifacts":
        - Nem tölt be modellt,
        - Egy korábban elmentett .npz fájlból betölti:
            - centroids
            - pos_calib
            - anchor_texts
            - anchor_scores

    A generate_persona_prompt_for_profile csak ezeket használja, így
    runtime-ban elég az artifacts mód.
    """

    def __init__(
        self,
        mode: EngineMode = "online",
        model_path: Optional[str] = None,
        factors_file: str = FACTORS_FILE,
        artifacts_path: Optional[str] = None,
    ):
        self.mode: EngineMode = mode
        self.factors_file = factors_file

        self.model: Optional[SentenceTransformer] = None
        self.centroids: Dict[str, np.ndarray] = {}
        self.pos_calib: Dict[str, Tuple[float, float]] = {}
        self.anchor_texts: List[str] = []
        self.anchor_scores: List[Dict[str, Dict[str, float]]] = []

        if mode == "online":
            if model_path is None:
                raise ValueError("mode='online' esetén kötelező a model_path paraméter.")

            # 1) Modell betöltése
            self.model = SentenceTransformer(model_path)

            # 2) Centroidok
            self.centroids = compute_factor_centroids(
                self.model,
                self.factors_file,
            )

            # 3) POS kalibráció
            self.pos_calib = compute_factor_pos_calibration(
                self.model,
                self.centroids,
                self.factors_file,
                low_target=2.0,
                high_target=7.0,
            )

            # 4) Anchor szövegek
            self.anchor_texts = self._load_anchor_texts(self.factors_file)

            # 5) Anchor score-ok
            self.anchor_scores = compute_factor_scores_for_texts(
                self.model,
                self.centroids,
                self.anchor_texts,
                pos_calib=self.pos_calib,
            )

        elif mode == "artifacts":
            if artifacts_path is None:
                raise ValueError("mode='artifacts' esetén kötelező az artifacts_path paraméter.")
            self._load_artifacts(artifacts_path)

        else:
            raise ValueError(f"Ismeretlen mode: {mode}")

    # ---------- OFFLINE HELPER: anchor szövegek betöltése ----------

    @staticmethod
    def _load_anchor_texts(path: str) -> List[str]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        texts: List[str] = []

        for rec in raw:
            lows = rec.get("factor_characteristic_low") or []
            highs = rec.get("factor_characteristic_high") or []

            texts.extend([t.strip() for t in lows if t.strip()])
            texts.extend([t.strip() for t in highs if t.strip()])

        return texts

    # ---------- ARTEFAKT MENTÉS / BETÖLTÉS ----------

    def save_artifacts(self, path: str) -> None:
        """
        Elmenti az engine állapotának lényeges részeit egy .npz fájlba.

        Csak akkor értelmes hívni, ha:
          - mode="online",
          - és már ki vannak számolva:
              - centroids
              - pos_calib
              - anchor_texts
              - anchor_scores
        """
        if not self.centroids:
            raise ValueError("Nincsenek centroids – biztos futott már az online init?")
        if not self.anchor_texts or not self.anchor_scores:
            raise ValueError("Hiányos anchor_texts / anchor_scores – online módban futtatva?")

        np.savez_compressed(
            path,
            centroids=self.centroids,
            pos_calib=self.pos_calib,
            anchor_texts=np.array(self.anchor_texts, dtype=object),
            anchor_scores=np.array(self.anchor_scores, dtype=object),
        )
        print(f"[PersonaEngine] Artefaktok elmentve ide: {path}")

    def _load_artifacts(self, path: str) -> None:
        """
        Artefaktok betöltése .npz fájlból (runtime mód).
        """
        data = np.load(path, allow_pickle=True)

        # dict-ek 0-dim object array-ként jönnek vissza
        self.centroids = data["centroids"].item()
        self.pos_calib = data["pos_calib"].item()

        # listák visszaalakítása
        self.anchor_texts = list(data["anchor_texts"].tolist())
        self.anchor_scores = list(data["anchor_scores"].tolist())

        # runtime módban nincs modell
        self.model = None


def build_factor_snippets_for_profile(
    engine: PersonaEngine,
    profile_levels: Dict[str, float],
    rel_threshold: float = 0.3,
    top_k_factors: int = 3,
    n_extreme: int = 3,
    n_mid: int = 1,
    style_levels: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Egy személy faktorprofiljára kiválaszt faktoronként néhány tipikus mondatot.

    + opcionálisan: a 9 kiegészítő stílusfaktorhoz 1-1 statikus mondatot ad hozzá,
      HA az értékük legalább 5 (5, 6, 7 vagy 8).
    """
    # 1) Alap mintavétel a 16 faktorhoz (embedding alapú)
    # samples = sample_texts_for_profile_simple(
    #     profile_levels=profile_levels,
    #     texts=engine.anchor_texts,
    #     all_scores=engine.anchor_scores,
    #     rel_threshold=rel_threshold,
    #     top_k_factors=top_k_factors,
    #     n_extreme=n_extreme,
    #     n_mid=n_mid,
    # )
    samples = sample_texts_for_profile_simple(
        profile_levels=profile_levels,
        texts=engine.anchor_texts,
        all_scores=engine.anchor_scores,
        rel_threshold=0.3,
        top_k_factors=3,
        n_extreme=3,
        n_mid=1,
        rerank_by_profile=True,
        rerank_rel_threshold=0.3,
        min_rel_for_rerank=0.6,
        top_n_per_bin_before_rerank=5,
    )

    out: Dict[str, Any] = {}

    for factor, lvl in profile_levels.items():
        factor_samples = samples.get(factor, [])
        s_list = []
        for text, bin_id, fs, label in factor_samples:
            s_list.append({
                "label": label,
                "bin": bin_id,
                "pos": float(fs["pos"]),
                "rel": float(fs["rel"]),
                "text": text,
            })

        out[factor] = {
            "level": float(lvl),
            "samples": s_list,
        }

    # 2) Kiegészítő 9 faktor – statikus mondatok csak 5–8 érték esetén
    if style_levels:
        for style_code, lvl in style_levels.items():
            if style_code not in STYLE_SENTENCES:
                continue

            level_int = int(round(float(lvl)))
            if level_int < 5:
                # csak 5, 6, 7, 8 esetén adunk mintát
                continue
            if level_int > 8:
                level_int = 8

            sentence_map = STYLE_SENTENCES[style_code]
            text = sentence_map.get(level_int)
            if not text:
                continue

            bin_id = level_int
            label = "STYLE"

            out[style_code] = {
                "level": float(lvl),
                "samples": [
                    {
                        "label": label,
                        "bin": bin_id,
                        "pos": float(lvl),  # egyszerűen a tényleges szintet adjuk vissza
                        "rel": 1.0,         # statikus, nem relevancia-alapú
                        "text": text,
                    }
                ],
            }

    return out



def render_factors_for_prompt(
    factor_snippets: Dict[str, Any],
    factor_name_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Az LLM-nek szánt faktor-blokkot generálja:

    === GYAK (level=6.0) ===
      [HIGH] bin=6, pos=6.44, rel=0.90 | ...
      ...
    """
    lines: List[str] = []

    for factor, info in factor_snippets.items():
        level = info["level"]
        samples = info["samples"]

        if factor_name_map:
            display_name = factor_name_map.get(factor, factor)
        else:
            display_name = factor

        lines.append(f"=== {display_name} (level={level:.1f}) ===")
        for s in samples:
            label = s["label"]
            bin_id = s["bin"]
            pos = s["pos"]
            rel = s["rel"]
            text = s["text"]
            lines.append(
                f"  [{label}] bin={bin_id}, pos={pos:.2f}, rel={rel:.2f} | {text}"
            )
        lines.append("")

    return "\n".join(lines).strip()


BASE_PERSONA_PROMPT_HU = """[Feladat]: Foglald össze pár mondatban a következő személyiséget!

[Szerep & Kontextus]:
Pszichológia és HR a fő szakirányod. Komplex viselkedési minták azonosításával foglalkozol, amiket különféle személyiségi faktorok elemzéséből vonsz le.
A fő feladatod, hogy támogasd a vezetőket azzal, hogy rövid, gyakorlatias leírásokat adsz a munkavállalókról.

[Skála értelmezés]:
- A faktorok 1–8 közötti skálán mozognak, ahol 1–3: alacsony, 4–5: közepes, 6–8: magas.
- A közepes tartományt ne írd le nagyon szélsőségként.

Az elemzésből a következő jellemzők jöttek:

{FAKTOR_BLOKK}

Munkahelyi környezete: {WORK_ENV}

[Elvárt eredmény]:
- Egy rövid, 3–5 mondatos összefoglaló, ami leírja a munkavállaló működését. Nem muszáj mindent belesűrítened és általánosíthatsz.
- Ha a leírásban szerepelnek kommunikációs vagy konfliktuskezelési stílusra utaló mondatok is, ezeket legfeljebb 1–2 rövid mondatban említsd meg, inkább árnyalatként, ne a leírás fő fókuszaként.
- A fő hangsúly maradjon a többi faktor által jelzett működésen.
- Írj 1–2 motivációs lehetőséget (hogyan lehet jól motiválni).
- Írj 1–2 fejlesztést célzó gondolatot (mire érdemes figyelni, mit lehet fejleszteni).
- Maradj konstruktív, HR-kompatibilis nyelvben: ne patologizálj, ne használj minősítő jelzőket (pl. "rossz", "használhatatlan"), inkább viselkedést írj le.
"""


def build_persona_prompt(
    factor_snippets: Dict[str, Any],
    work_env: str = "iroda",
    factor_name_map: Optional[Dict[str, str]] = None,
) -> str:
    faktor_blokk = render_factors_for_prompt(
        factor_snippets,
        factor_name_map=factor_name_map,
    )
    prompt = BASE_PERSONA_PROMPT_HU.format(
        FAKTOR_BLOKK=faktor_blokk,
        WORK_ENV=work_env,
    )
    return prompt

def generate_persona_prompt_for_profile(
    engine: PersonaEngine,
    profile_levels: Dict[str, float],
    work_env: str = "iroda",
    factor_name_map: Optional[Dict[str, str]] = None,
    style_levels: Optional[Dict[str, float]] = None,  # 👈 ÚJ
) -> Tuple[str, Dict[str, Any]]:
    factor_snippets = build_factor_snippets_for_profile(
        engine,
        profile_levels=profile_levels,
        rel_threshold=0.3,
        top_k_factors=3,
        n_extreme=3,
        n_mid=1,
        style_levels=style_levels,
    )

    prompt = build_persona_prompt(
        factor_snippets=factor_snippets,
        work_env=work_env,
        factor_name_map=factor_name_map,
    )
    return prompt, factor_snippets

