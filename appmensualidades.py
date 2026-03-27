import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Explorador df_cc", layout="wide")

st.title("📊 Explorador de Mensualidades")
st.write("Sube tu archivo CSV o Excel para explorar la base.")

@st.cache_data
def cargar_datos(archivo):
    nombre = archivo.name.lower()

    if nombre.endswith(".csv"):
        # Intento 1: utf-8 con coma
        try:
            df = pd.read_csv(archivo, encoding="utf-8", sep=",")
            if df.shape[1] == 1:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding="utf-8", sep=";")
            return df
        except Exception:
            pass

        # Intento 2: latin-1 con coma
        archivo.seek(0)
        try:
            df = pd.read_csv(archivo, encoding="latin-1", sep=",")
            if df.shape[1] == 1:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding="latin-1", sep=";")
            return df
        except Exception as e:
            raise ValueError(f"No se pudo leer el CSV: {e}")

    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        try:
            xls = pd.ExcelFile(archivo)
            hojas = xls.sheet_names
            return xls, hojas
        except Exception as e:
            raise ValueError(f"No se pudo leer el Excel: {e}")

    else:
        raise ValueError("Formato no soportado. Sube CSV o Excel.")

archivo = st.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls"])

if archivo is not None:
    try:
        resultado = cargar_datos(archivo)

        if isinstance(resultado, tuple):
            xls, hojas = resultado
            hoja = st.selectbox("Selecciona la hoja", hojas)
            df = pd.read_excel(xls, sheet_name=hoja)
        else:
            df = resultado

        df.columns = [str(c).strip() for c in df.columns]

        st.success("Archivo cargado correctamente")

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", len(df))
        col2.metric("Columnas", df.shape[1])
        col3.metric("Nulos", int(df.isna().sum().sum()))

        st.subheader("Columnas detectadas")
        st.write(df.columns.tolist())

        st.subheader("Vista previa")
        st.dataframe(df.head(50), use_container_width=True)

        st.subheader("Filtros")

        df_filtrado = df.copy()

        columnas_objeto = df_filtrado.select_dtypes(include="object").columns.tolist()
        columnas_numericas = df_filtrado.select_dtypes(include=["number"]).columns.tolist()

        with st.sidebar:
            st.header("Filtros")

            for col in columnas_objeto:
                valores_unicos = df_filtrado[col].dropna().astype(str).unique().tolist()
                if 0 < len(valores_unicos) <= 50:
                    seleccion = st.multiselect(f"{col}", sorted(valores_unicos))
                    if seleccion:
                        df_filtrado = df_filtrado[df_filtrado[col].astype(str).isin(seleccion)]

        st.subheader("Generador de gráfica")

        columnas = df_filtrado.columns.tolist()

        c1, c2, c3 = st.columns(3)
        with c1:
            x = st.selectbox("Eje X", columnas)
        with c2:
            y = st.selectbox("Eje Y", [None] + columnas, index=0)
        with c3:
            tipo = st.selectbox("Tipo de gráfica", ["bar", "line", "scatter", "histogram", "box"])

        try:
            if tipo == "bar":
                if y is None:
                    st.warning("Para barras selecciona también un eje Y.")
                else:
                    fig = px.bar(df_filtrado, x=x, y=y)
                    st.plotly_chart(fig, use_container_width=True)

            elif tipo == "line":
                if y is None:
                    st.warning("Para línea selecciona también un eje Y.")
                else:
                    fig = px.line(df_filtrado, x=x, y=y)
                    st.plotly_chart(fig, use_container_width=True)

            elif tipo == "scatter":
                if y is None:
                    st.warning("Para dispersión selecciona también un eje Y.")
                else:
                    fig = px.scatter(df_filtrado, x=x, y=y)
                    st.plotly_chart(fig, use_container_width=True)

            elif tipo == "histogram":
                fig = px.histogram(df_filtrado, x=x)
                st.plotly_chart(fig, use_container_width=True)

            elif tipo == "box":
                if y is None:
                    st.warning("Para box selecciona también un eje Y.")
                else:
                    fig = px.box(df_filtrado, x=x, y=y)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"No se pudo generar la gráfica: {e}")

        st.subheader("Datos filtrados")
        st.dataframe(df_filtrado, use_container_width=True)

        csv_descarga = df_filtrado.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar datos filtrados",
            data=csv_descarga,
            file_name="df_filtrado.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Sube un archivo para comenzar.")
