"""App de demostración: clasificación de cobertura del suelo en Villa Hayes.

Uso:  streamlit run app.py
Necesita model.pkl y datos_muestra.csv en la misma carpeta.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="Cobertura del suelo · Villa Hayes", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "model.pkl")


@st.cache_data
def cargar_muestra():
    return pd.read_csv(BASE / "datos_muestra.csv")


paquete = cargar_modelo()
pipe = paquete["pipeline"]
FEATURES = paquete["features"]
A_CODIGO = paquete["a_codigo"]
NOMBRES = paquete["nombres"]
COLORES = paquete["colores"]


def predecir(df):
    proba = pipe.predict_proba(df[FEATURES])
    idx = proba.argmax(axis=1)
    out = df.copy()
    out["clase_predicha"] = [A_CODIGO[i] for i in idx]
    out["cobertura"] = out["clase_predicha"].map(NOMBRES)
    out["confianza"] = proba.max(axis=1).round(3)
    return out, proba


def hex_a_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


st.title("Clasificación de cobertura del suelo · Villa Hayes")
st.caption(
    f"Sentinel-2 (temporada húmeda y seca) · etiquetas MapBiomas Chaco · modelo: "
    f"{paquete['modelo']} · F1-macro en test espacial: {paquete['metricas']['f1_macro_test']:.2f}"
)

tab_lote, tab_pixel, tab_roi = st.tabs(["Clasificar un lote", "Explorar un píxel", "ROI"])

# ---------------------------------------------------------------- lote
with tab_lote:
    st.subheader("Clasificar una tabla de píxeles")
    st.write(
        "Subí un CSV con las 37 features espectrales (el mismo formato que exporta el "
        "notebook de extracción), o usá la muestra de 100 píxeles del test."
    )
    archivo = st.file_uploader("CSV de píxeles", type="csv")
    datos = pd.read_csv(archivo) if archivo else cargar_muestra()
    if not archivo:
        st.info("Usando datos_muestra.csv (100 píxeles que el modelo no vio al entrenar).")

    faltan = [f for f in FEATURES if f not in datos.columns]
    if faltan:
        st.error(f"Al archivo le faltan {len(faltan)} columnas, por ejemplo: {faltan[:5]}")
        st.stop()

    res, _ = predecir(datos)

    c1, c2, c3 = st.columns(3)
    c1.metric("Píxeles clasificados", len(res))
    c2.metric("Confianza media", f"{res['confianza'].mean():.0%}")
    if "clase" in res.columns:
        acierto = (res["clase"] == res["clase_predicha"]).mean()
        c3.metric("Coincidencia con MapBiomas", f"{acierto:.0%}")

    if {"lon", "lat"}.issubset(res.columns):
        res["color"] = res["clase_predicha"].map(lambda c: hex_a_rgb(COLORES[c]) + [210])
        capa = pdk.Layer(
            "ScatterplotLayer", data=res, get_position="[lon, lat]",
            get_fill_color="color", get_radius=900, pickable=True,
        )
        vista = pdk.ViewState(latitude=res["lat"].mean(), longitude=res["lon"].mean(), zoom=7.5)
        st.pydeck_chart(pdk.Deck(
            layers=[capa], initial_view_state=vista, map_style=None,
            tooltip={"text": "{cobertura} ({confianza})"},
        ))
        st.markdown(" · ".join(
            f"<span style='color:{COLORES[c]}'>●</span> {NOMBRES[c]}" for c in sorted(NOMBRES)
        ), unsafe_allow_html=True)

    st.bar_chart(res["cobertura"].value_counts())

    columnas = ["cobertura", "confianza"] + (["clase"] if "clase" in res else []) + \
               [c for c in ["anio", "lon", "lat"] if c in res]
    vista_tabla = res[columnas].copy()
    if "clase" in vista_tabla:
        vista_tabla["clase"] = vista_tabla["clase"].map(NOMBRES)
        vista_tabla = vista_tabla.rename(columns={"clase": "MapBiomas"})
    st.dataframe(vista_tabla, width="stretch")
    st.download_button("Descargar resultado (CSV)",
                       res.drop(columns=["color"], errors="ignore").to_csv(index=False),
                       "clasificacion.csv", "text/csv")

# ---------------------------------------------------------------- píxel
with tab_pixel:
    st.subheader("¿Qué pasa si cambio la firma de un píxel?")
    st.write(
        "Elegí un píxel real y modificá sus índices principales. El modelo reclasifica en "
        "tiempo real: sirve para ver qué variables lo hacen cambiar de opinión."
    )
    muestra = cargar_muestra()
    etiquetas = [
        f"#{i} · {NOMBRES[c]} ({a})" for i, (c, a) in enumerate(zip(muestra["clase"], muestra["anio"]))
    ]
    elegido = st.selectbox("Píxel de partida", range(len(muestra)), format_func=lambda i: etiquetas[i])
    fila = muestra.iloc[[elegido]].copy()

    col_a, col_b = st.columns(2)
    with col_a:
        ndvi_h = st.slider("NDVI temporada húmeda", -1.0, 1.0, float(fila["NDVI_humeda"].iloc[0]), 0.01)
        ndvi_s = st.slider("NDVI temporada seca", -1.0, 1.0, float(fila["NDVI_seca"].iloc[0]), 0.01)
    with col_b:
        mndwi_h = st.slider("MNDWI temporada húmeda (agua)", -1.0, 1.0, float(fila["MNDWI_humeda"].iloc[0]), 0.01)
        ndbi_s = st.slider("NDBI temporada seca (construido)", -1.0, 1.0, float(fila["NDBI_seca"].iloc[0]), 0.01)

    fila["NDVI_humeda"], fila["NDVI_seca"] = ndvi_h, ndvi_s
    fila["dNDVI"] = ndvi_h - ndvi_s
    fila["MNDWI_humeda"], fila["NDBI_seca"] = mndwi_h, ndbi_s

    res_p, proba_p = predecir(fila)
    st.metric("Clase predicha", res_p["cobertura"].iloc[0],
              f"confianza {res_p['confianza'].iloc[0]:.0%}")
    st.caption(f"Etiqueta MapBiomas de este píxel: {NOMBRES[int(muestra['clase'].iloc[elegido])]}")
    probs = pd.Series(proba_p[0], index=[NOMBRES[A_CODIGO[i]] for i in range(len(proba_p[0]))])
    st.bar_chart(probs)

# ---------------------------------------------------------------- ROI
with tab_roi:
    st.subheader("Retorno de la inversión frente a la fotointerpretación manual")
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Área a mapear (km²)", value=17650, step=500)
        tarifa = st.number_input("Tarifa del técnico SIG (USD/h)", value=30, step=1)
        seg_km2 = st.number_input("Interpretación + digitalización manual (segundos por km²)",
                                  value=40, step=5)
    with c2:
        h_desarrollo = st.number_input("Horas de desarrollo del pipeline (único)", value=60, step=5)
        h_mapa = st.number_input("Horas por cada mapa nuevo con ML", value=6, step=1)

    escenarios = {"A · Mapa único": 1, "B · Anual (5 años)": 5, "C · Trimestral (5 años)": 20}
    filas = []
    for nombre, n in escenarios.items():
        manual = n * area * seg_km2 / 3600 * tarifa
        ml = (h_desarrollo + n * h_mapa) * tarifa
        filas.append({"Escenario": nombre, "Mapas": n, "Manual (USD)": round(manual),
                      "Con ML (USD)": round(ml), "Ahorro (USD)": round(manual - ml),
                      "ROI": f"{(manual - ml) / ml:.0%}" if ml else "—"})
    tabla = pd.DataFrame(filas)
    st.caption(f"Un mapa manual del área: {area * seg_km2 / 3600:,.0f} horas de trabajo")
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.bar_chart(tabla.set_index("Escenario")[["Manual (USD)", "Con ML (USD)"]])
    st.caption("Supuestos editables. El mapa automático tiene un error medible que un técnico "
               "debe revisar en las zonas críticas.")
