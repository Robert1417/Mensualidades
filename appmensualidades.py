import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cosechas Mensualidades", layout="wide")

# =========================================================
# CONFIG
# =========================================================
DATA_PATH_PARQUET = "df_cc.parquet"
DATA_PATH_CSV = "data/df_cc.csv"

COLUMNAS_ESPERADAS = {
    "Referencia": "Referencia",
    "Credito": "Credito",
    "Tipo de comision": "Tipo de comision",
    "Fecha de cobro": "Fecha de cobro",
    "Monto": "Monto",
    "Apartado Mensual": "Apartado Mensual",
    "Vehiculo de ahorro": "Vehiculo de ahorro",
    "Deuda inicial Fija": "Deuda inicial Fija",
    "Mes_Año": "Mes_Año",
    "Mes_Cobro": "Mes_Cobro",
    "AM/DB": "AM/DB",
    "Rango_AM_DB": "Rango_AM_DB",
    "Rango_de_deuda": "Rango_de_deuda",
    "FECHA DE BAJA": "FECHA DE BAJA",
    "FECHA DE GRADUACIÓN": "FECHA DE GRADUACIÓN",
    "CM Condonada $": "CM Condonada $",
    "CE Condonada $": "CE Condonada $",
    "MES_BASE": "MES_BASE",
    "monthly_payment": "monthly_payment",
    "begin_of_program": "begin_of_program",
    "type": "type",
}

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_mes_anyo(valor):
    """
    Espera formatos como:
    - 06-2025
    - 6-2025
    - 06/2025
    """
    if pd.isna(valor):
        return pd.NaT

    s = str(valor).strip().replace("/", "-")
    try:
        return pd.to_datetime("01-" + s, format="%d-%m-%Y", errors="coerce")
    except Exception:
        return pd.NaT


def meses_entre(fecha_inicio, fecha_fin):
    if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
        return np.nan
    return (fecha_fin.year - fecha_inicio.year) * 12 + (fecha_fin.month - fecha_inicio.month)


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalizar_columnas(df)

    # Crear columnas faltantes si no existen
    for col in COLUMNAS_ESPERADAS:
        if col not in df.columns:
            df[col] = np.nan

    # Tipos
    cols_numericas = [
        "Monto",
        "Apartado Mensual",
        "Deuda inicial Fija",
        "Mes_Cobro",
        "AM/DB",
        "CM Condonada $",
        "CE Condonada $",
        "monthly_payment",
    ]
    for col in cols_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Fecha de cobro"] = pd.to_datetime(df["Fecha de cobro"], errors="coerce")
    df["FECHA DE BAJA"] = pd.to_datetime(df["FECHA DE BAJA"], errors="coerce")
    df["FECHA DE GRADUACIÓN"] = pd.to_datetime(df["FECHA DE GRADUACIÓN"], errors="coerce")
    df["begin_of_program"] = pd.to_datetime(df["begin_of_program"], errors="coerce")

    df["Referencia"] = df["Referencia"].astype(str).str.strip()
    df["Tipo de comision"] = df["Tipo de comision"].astype(str).str.strip()

    # Originación
    df["fecha_originacion"] = df["Mes_Año"].apply(parse_mes_anyo)

    # Condonado: si cualquiera de las 2 tiene valor distinto de nulo y distinto de 0
    df["Condonado"] = np.where(
        (
            (df["CM Condonada $"].fillna(0) != 0)
            | (df["CE Condonada $"].fillna(0) != 0)
        ),
        "Sí",
        "No"
    )

    # Crédito
    df["Credito"] = df["Credito"].astype(str).str.strip().str.lower().map(
        {"true": "True", "false": "False", "verdadero": "True", "falso": "False"}
    ).fillna(df["Credito"].astype(str))

    return df


@st.cache_data(show_spinner=False)
def cargar_datos_fijos():
    if os.path.exists(DATA_PATH_PARQUET):
        df = pd.read_parquet(DATA_PATH_PARQUET)
        return preparar_df(df), f"Base cargada desde {DATA_PATH_PARQUET}"

    if os.path.exists(DATA_PATH_CSV):
        try:
            df = pd.read_csv(DATA_PATH_CSV, encoding="utf-8", sep=",")
            if df.shape[1] == 1:
                df = pd.read_csv(DATA_PATH_CSV, encoding="utf-8", sep=";")
        except Exception:
            try:
                df = pd.read_csv(DATA_PATH_CSV, encoding="latin-1", sep=",")
                if df.shape[1] == 1:
                    df = pd.read_csv(DATA_PATH_CSV, encoding="latin-1", sep=";")
            except Exception as e:
                raise ValueError(f"No se pudo leer la base fija CSV: {e}")
        return preparar_df(df), f"Base cargada desde {DATA_PATH_CSV}"

    return None, "No se encontró base fija en data/."


@st.cache_data(show_spinner=False)
def cargar_desde_upload(archivo):
    nombre = archivo.name.lower()

    if nombre.endswith(".parquet"):
        df = pd.read_parquet(archivo)
    elif nombre.endswith(".csv"):
        try:
            df = pd.read_csv(archivo, encoding="utf-8", sep=",")
            if df.shape[1] == 1:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding="utf-8", sep=";")
        except Exception:
            archivo.seek(0)
            df = pd.read_csv(archivo, encoding="latin-1", sep=",")
            if df.shape[1] == 1:
                archivo.seek(0)
                df = pd.read_csv(archivo, encoding="latin-1", sep=";")
    elif nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        xls = pd.ExcelFile(archivo)
        hoja = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=hoja)
    else:
        raise ValueError("Formato no soportado")

    return preparar_df(df)


def construir_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla única por Referencia para análisis de baja / graduación / condonación.
    """
    agg = df.groupby("Referencia", dropna=False).agg(
        fecha_originacion=("fecha_originacion", "min"),
        Mes_Año=("Mes_Año", "first"),
        Credito=("Credito", "first"),
        FECHA_DE_BAJA=("FECHA DE BAJA", "max"),
        FECHA_DE_GRADUACION=("FECHA DE GRADUACIÓN", "max"),
        Condonado=("Condonado", lambda x: "Sí" if (x == "Sí").any() else "No"),
        Deuda_inicial_Fija=("Deuda inicial Fija", "max"),
        Apartado_Mensual=("Apartado Mensual", "max"),
        Rango_AM_DB=("Rango_AM_DB", "first"),
        Rango_de_deuda=("Rango_de_deuda", "first"),
    ).reset_index()

    agg["mes_baja"] = agg.apply(
        lambda r: meses_entre(r["fecha_originacion"], r["FECHA_DE_BAJA"]), axis=1
    )
    agg["mes_graduacion"] = agg.apply(
        lambda r: meses_entre(r["fecha_originacion"], r["FECHA_DE_GRADUACION"]), axis=1
    )

    agg["estado_cliente"] = np.select(
        [
            agg["FECHA_DE_GRADUACION"].notna(),
            agg["FECHA_DE_BAJA"].notna(),
        ],
        [
            "Graduado",
            "Baja",
        ],
        default="Activo"
    )

    return agg


def resumen_cosechas(clientes: pd.DataFrame, df_cobros: pd.DataFrame) -> pd.DataFrame:
    base_clientes = clientes.groupby("Mes_Año", dropna=False).agg(
        clientes=("Referencia", "nunique"),
        condonados=("Condonado", lambda x: (x == "Sí").sum()),
        graduados=("estado_cliente", lambda x: (x == "Graduado").sum()),
        bajas=("estado_cliente", lambda x: (x == "Baja").sum()),
        activos=("estado_cliente", lambda x: (x == "Activo").sum()),
    ).reset_index()

    base_cobros = df_cobros.groupby("Mes_Año", dropna=False).agg(
        monto_total=("Monto", "sum"),
        transacciones=("Monto", "size"),
    ).reset_index()

    out = base_clientes.merge(base_cobros, on="Mes_Año", how="left")
    out["monto_total"] = out["monto_total"].fillna(0)
    out["transacciones"] = out["transacciones"].fillna(0).astype(int)
    out["pct_condonados"] = np.where(out["clientes"] > 0, out["condonados"] / out["clientes"], 0.0)
    out["pct_graduados"] = np.where(out["clientes"] > 0, out["graduados"] / out["clientes"], 0.0)
    out["pct_bajas"] = np.where(out["clientes"] > 0, out["bajas"] / out["clientes"], 0.0)
    return out.sort_values("Mes_Año")


def curva_cobros(df: pd.DataFrame, acumulado=False) -> pd.DataFrame:
    out = (
        df.groupby(["Mes_Año", "Mes_Cobro", "Tipo de comision", "Condonado"], dropna=False)["Monto"]
        .sum()
        .reset_index()
        .sort_values(["Mes_Año", "Tipo de comision", "Condonado", "Mes_Cobro"])
    )

    if acumulado:
        out["Monto"] = (
            out.groupby(["Mes_Año", "Tipo de comision", "Condonado"])["Monto"]
            .cumsum()
        )
    return out


def curva_evento(clientes: pd.DataFrame, columna_mes: str, etiqueta: str) -> pd.DataFrame:
    tmp = clientes[clientes[columna_mes].notna()].copy()
    tmp[columna_mes] = tmp[columna_mes].astype(int)

    out = (
        tmp.groupby(["Mes_Año", columna_mes, "Condonado"], dropna=False)["Referencia"]
        .nunique()
        .reset_index(name="Clientes")
        .sort_values(["Mes_Año", "Condonado", columna_mes])
    )

    base = (
        clientes.groupby(["Mes_Año", "Condonado"], dropna=False)["Referencia"]
        .nunique()
        .reset_index(name="Base_Clientes")
    )

    out = out.merge(base, on=["Mes_Año", "Condonado"], how="left")
    out["Pct"] = np.where(out["Base_Clientes"] > 0, out["Clientes"] / out["Base_Clientes"], 0.0)
    out["Evento"] = etiqueta
    return out


# =========================================================
# CARGA DE DATOS
# =========================================================
st.title("📊 Dashboard de Cosechas - Mensualidades")

df, origen_msg = cargar_datos_fijos()

with st.expander("Carga de datos", expanded=False):
    st.write(origen_msg)
    archivo = st.file_uploader(
        "Si quieres reemplazar temporalmente la base fija, sube otro archivo aquí",
        type=["parquet", "csv", "xlsx", "xls"]
    )
    if archivo is not None:
        df = cargar_desde_upload(archivo)
        st.success("Usando archivo subido manualmente en esta sesión")

if df is None:
    st.warning("No hay base fija cargada. Sube un archivo o agrega data/df_cc.parquet al repo.")
    st.stop()

clientes = construir_clientes(df)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Filtros")

cosechas_disponibles = sorted([x for x in df["Mes_Año"].dropna().astype(str).unique().tolist()])
tipos_disponibles = sorted([x for x in df["Tipo de comision"].dropna().astype(str).unique().tolist()])
condonado_opts = ["Todos", "Sí", "No"]
credito_opts = ["Todos"] + sorted([x for x in clientes["Credito"].dropna().astype(str).unique().tolist()])

cosechas_sel = st.sidebar.multiselect("Cosecha (Mes_Año)", cosechas_disponibles, default=cosechas_disponibles[:5] if len(cosechas_disponibles) > 5 else cosechas_disponibles)
tipos_sel = st.sidebar.multiselect("Tipo de comisión", tipos_disponibles, default=tipos_disponibles)
condonado_sel = st.sidebar.selectbox("Condonado", condonado_opts, index=0)
credito_sel = st.sidebar.selectbox("Crédito", credito_opts, index=0)
ver_pct = st.sidebar.checkbox("Ver bajas / graduaciones en porcentaje", value=True)

df_f = df.copy()
clientes_f = clientes.copy()

if cosechas_sel:
    df_f = df_f[df_f["Mes_Año"].astype(str).isin(cosechas_sel)]
    clientes_f = clientes_f[clientes_f["Mes_Año"].astype(str).isin(cosechas_sel)]

if tipos_sel:
    df_f = df_f[df_f["Tipo de comision"].astype(str).isin(tipos_sel)]

if condonado_sel != "Todos":
    df_f = df_f[df_f["Condonado"] == condonado_sel]
    clientes_f = clientes_f[clientes_f["Condonado"] == condonado_sel]

if credito_sel != "Todos":
    clientes_f = clientes_f[clientes_f["Credito"].astype(str) == credito_sel]
    refs_validas = set(clientes_f["Referencia"])
    df_f = df_f[df_f["Referencia"].isin(refs_validas)]

# =========================================================
# KPIs
# =========================================================
resumen = resumen_cosechas(clientes_f, df_f)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clientes únicos", f"{clientes_f['Referencia'].nunique():,}")
k2.metric("Cobro total", f"{df_f['Monto'].sum():,.0f}")
k3.metric("Bajas", f"{(clientes_f['estado_cliente'] == 'Baja').sum():,}")
k4.metric("Graduados", f"{(clientes_f['estado_cliente'] == 'Graduado').sum():,}")
k5.metric("Condonados", f"{(clientes_f['Condonado'] == 'Sí').sum():,}")

st.subheader("Resumen por cosecha")
st.dataframe(resumen, use_container_width=True)

# =========================================================
# COBROS
# =========================================================
st.subheader("Curva de cobros por cosecha")

tabs = st.tabs([
    "Cobro mensual",
    "Cobro acumulado",
    "Bajas",
    "Graduaciones",
    "Clientes"
])

with tabs[0]:
    cobros_mes = curva_cobros(df_f, acumulado=False)
    if not cobros_mes.empty:
        fig = px.line(
            cobros_mes,
            x="Mes_Cobro",
            y="Monto",
            color="Mes_Año",
            line_dash="Condonado",
            facet_col="Tipo de comision",
            markers=True,
            title="Monto cobrado por mes desde originación"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cobros_mes, use_container_width=True)
    else:
        st.info("No hay datos para la vista de cobro mensual.")

with tabs[1]:
    cobros_acum = curva_cobros(df_f, acumulado=True)
    if not cobros_acum.empty:
        fig = px.line(
            cobros_acum,
            x="Mes_Cobro",
            y="Monto",
            color="Mes_Año",
            line_dash="Condonado",
            facet_col="Tipo de comision",
            markers=True,
            title="Cobro acumulado por mes desde originación"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cobros_acum, use_container_width=True)
    else:
        st.info("No hay datos para la vista de cobro acumulado.")

with tabs[2]:
    bajas = curva_evento(clientes_f, "mes_baja", "Baja")
    if not bajas.empty:
        ycol = "Pct" if ver_pct else "Clientes"
        fig = px.line(
            bajas,
            x="mes_baja",
            y=ycol,
            color="Mes_Año",
            line_dash="Condonado",
            markers=True,
            title="Curva de bajas por mes desde originación"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bajas, use_container_width=True)
    else:
        st.info("No hay clientes con baja para los filtros seleccionados.")

with tabs[3]:
    grads = curva_evento(clientes_f, "mes_graduacion", "Graduación")
    if not grads.empty:
        ycol = "Pct" if ver_pct else "Clientes"
        fig = px.line(
            grads,
            x="mes_graduacion",
            y=ycol,
            color="Mes_Año",
            line_dash="Condonado",
            markers=True,
            title="Curva de graduación por mes desde originación"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grads, use_container_width=True)
    else:
        st.info("No hay clientes graduados para los filtros seleccionados.")

with tabs[4]:
    clientes_det = clientes_f.copy()
    st.dataframe(clientes_det, use_container_width=True)

# =========================================================
# DESCARGAS
# =========================================================
st.subheader("Descargas")

c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "Descargar cobros filtrados",
        data=df_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="cobros_filtrados.csv",
        mime="text/csv"
    )

with c2:
    st.download_button(
        "Descargar clientes filtrados",
        data=clientes_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="clientes_filtrados.csv",
        mime="text/csv"
    )

with c3:
    st.download_button(
        "Descargar resumen cosechas",
        data=resumen.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumen_cosechas.csv",
        mime="text/csv"
    )
