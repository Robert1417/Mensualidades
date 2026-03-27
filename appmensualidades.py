import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Explorador de Datos", layout="wide")

st.title("📊 Explorador de Datos Inteligente")

# ================================
# CARGA DE ARCHIVO
# ================================
archivo = st.file_uploader("Sube tu archivo Excel o CSV", type=["xlsx", "csv"])

@st.cache_data
def cargar_datos(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if archivo is not None:
    df = cargar_datos(archivo)

    st.success("Archivo cargado correctamente")

    # ================================
    # KPIs
    # ================================
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", len(df))
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos", df.isna().sum().sum())

    # ================================
    # SIDEBAR FILTROS
    # ================================
    st.sidebar.header("Filtros")

    df_filtrado = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            valores = st.sidebar.multiselect(f"{col}", df[col].dropna().unique())
            if valores:
                df_filtrado = df_filtrado[df_filtrado[col].isin(valores)]

    # ================================
    # SELECTORES DE GRÁFICA
    # ================================
    st.subheader("Generador de gráficas")

    columnas = df_filtrado.columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        x = st.selectbox("Eje X", columnas)

    with col2:
        y = st.selectbox("Eje Y", columnas)

    with col3:
        tipo = st.selectbox("Tipo", ["line", "bar", "scatter", "histogram", "box"])

    # ================================
    # GRAFICAS
    # ================================
    try:
        if tipo == "line":
            fig = px.line(df_filtrado, x=x, y=y)
        elif tipo == "bar":
            fig = px.bar(df_filtrado, x=x, y=y)
        elif tipo == "scatter":
            fig = px.scatter(df_filtrado, x=x, y=y)
        elif tipo == "histogram":
            fig = px.histogram(df_filtrado, x=x)
        else:
            fig = px.box(df_filtrado, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning("No se pudo generar la gráfica con esas columnas")

    # ================================
    # TABLA
    # ================================
    st.subheader("Datos filtrados")
    st.dataframe(df_filtrado, use_container_width=True)

    # ================================
    # DESCARGA
    # ================================
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar datos filtrados",
        csv,
        "datos_filtrados.csv",
        "text/csv"
    )

else:
    st.info("Sube un archivo para comenzar")
