import os
import re
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard Anual - Mensualidades", layout="wide")

DATA_PATH_PARQUET = "data/df_cc.parquet"
DATA_PATH_CSV = "data/df_cc.csv"


def quitar_tildes(texto):
    if pd.isna(texto):
        return texto
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def normalizar_nombre_col(col):
    col = quitar_tildes(col)
    col = str(col).strip().lower()
    col = re.sub(r"\s+", " ", col)
    return col


def renombrar_columnas_estandar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    originales = df.columns.tolist()
    normalizadas = [normalizar_nombre_col(c) for c in originales]
    df.columns = normalizadas

    mapa = {
        "referencia": "Referencia",
        "credito": "Credito",
        "tipo de comision": "Tipo de comision",
        "fecha de cobro": "Fecha de cobro",
        "monto": "Monto",
        "apartado mensual": "Apartado Mensual",
        "vehiculo de ahorro": "Vehiculo de ahorro",
        "deuda inicial fija": "Deuda inicial Fija",
        "mes_año": "Mes_Año",
        "mes_ano": "Mes_Año",
        "mes año": "Mes_Año",
        "mes ano": "Mes_Año",
        "mes_cobro": "Mes_Cobro",
        "mes cobro": "Mes_Cobro",
        "am/db": "AM/DB",
        "rango_am_db": "Rango_AM_DB",
        "rango am db": "Rango_AM_DB",
        "rango de deuda": "Rango_de_deuda",
        "rango_de_deuda": "Rango_de_deuda",
        "fecha de baja": "FECHA DE BAJA",
        "fecha de graduacion": "FECHA DE GRADUACIÓN",
        "fecha de graduación": "FECHA DE GRADUACIÓN",
        "cm condonada $": "CM Condonada $",
        "ce condonada $": "CE Condonada $",
        "mes_base": "MES_BASE",
        "mes base": "MES_BASE",
        "monthly_payment": "monthly_payment",
        "begin_of_program": "begin_of_program",
        "type": "type",
    }

    df = df.rename(columns=mapa)
    return df


def parse_mes_anyo(valor):
    if pd.isna(valor):
        return pd.NaT

    s = str(valor).strip().replace("/", "-")

    formatos = [
        "%m-%Y",
        "%Y-%m",
        "%d-%m-%Y",
    ]

    for fmt in formatos:
        try:
            if fmt == "%m-%Y":
                return pd.to_datetime("01-" + s, format="%d-%m-%Y", errors="raise")
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass

    return pd.to_datetime(s, errors="coerce")


def meses_entre(fecha_inicio, fecha_fin):
    if pd.isna(fecha_inicio) or pd.isna(fecha_fin):
        return np.nan
    return (fecha_fin.year - fecha_inicio.year) * 12 + (fecha_fin.month - fecha_inicio.month)


def preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_estandar(df)

    columnas_necesarias = [
        "Referencia", "Credito", "Tipo de comision", "Fecha de cobro", "Monto",
        "Apartado Mensual", "Vehiculo de ahorro", "Deuda inicial Fija",
        "Mes_Año", "Mes_Cobro", "AM/DB", "Rango_AM_DB", "Rango_de_deuda",
        "FECHA DE BAJA", "FECHA DE GRADUACIÓN", "CM Condonada $", "CE Condonada $",
        "MES_BASE", "monthly_payment", "begin_of_program", "type"
    ]

    for col in columnas_necesarias:
        if col not in df.columns:
            df[col] = np.nan

    numericas = [
        "Monto", "Apartado Mensual", "Deuda inicial Fija", "Mes_Cobro",
        "AM/DB", "CM Condonada $", "CE Condonada $", "monthly_payment"
    ]
    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Fecha de cobro"] = pd.to_datetime(df["Fecha de cobro"], errors="coerce")
    df["FECHA DE BAJA"] = pd.to_datetime(df["FECHA DE BAJA"], errors="coerce")
    df["FECHA DE GRADUACIÓN"] = pd.to_datetime(df["FECHA DE GRADUACIÓN"], errors="coerce")
    df["begin_of_program"] = pd.to_datetime(df["begin_of_program"], errors="coerce")

    df["Referencia"] = df["Referencia"].astype(str).str.strip()
    df["Tipo de comision"] = df["Tipo de comision"].astype(str).str.strip()
    df["Mes_Año"] = df["Mes_Año"].astype(str).str.strip()

    df["fecha_originacion"] = df["Mes_Año"].apply(parse_mes_anyo)

    df["Anio_Cosecha"] = df["fecha_originacion"].dt.year
    df["Anio_Cosecha"] = df["Anio_Cosecha"].astype("Int64").astype(str)
    df.loc[df["fecha_originacion"].isna(), "Anio_Cosecha"] = "Sin año"

    df["Condonado"] = np.where(
        (df["CM Condonada $"].fillna(0) != 0) | (df["CE Condonada $"].fillna(0) != 0),
        "Sí",
        "No"
    )

    df["Credito"] = (
        df["Credito"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "true": "True",
            "false": "False",
            "verdadero": "True",
            "falso": "False"
        })
    )

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


def construir_clientes(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = [
        "Referencia", "fecha_originacion", "Mes_Año", "Anio_Cosecha", "Credito",
        "FECHA DE BAJA", "FECHA DE GRADUACIÓN", "Condonado",
        "Deuda inicial Fija", "Apartado Mensual", "Rango_AM_DB", "Rango_de_deuda"
    ]

    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas para construir clientes: "
            + ", ".join(faltantes)
            + f". Columnas detectadas: {list(df.columns)}"
        )

    clientes = df.groupby("Referencia", dropna=False).agg(
        fecha_originacion=("fecha_originacion", "min"),
        Mes_Año=("Mes_Año", "first"),
        Anio_Cosecha=("Anio_Cosecha", "first"),
        Credito=("Credito", "first"),
        FECHA_DE_BAJA=("FECHA DE BAJA", "max"),
        FECHA_DE_GRADUACION=("FECHA DE GRADUACIÓN", "max"),
        Condonado=("Condonado", lambda x: "Sí" if (x == "Sí").any() else "No"),
        Deuda_inicial_Fija=("Deuda inicial Fija", "max"),
        Apartado_Mensual=("Apartado Mensual", "max"),
        Rango_AM_DB=("Rango_AM_DB", "first"),
        Rango_de_deuda=("Rango_de_deuda", "first"),
    ).reset_index()

    clientes["mes_baja"] = clientes.apply(
        lambda r: meses_entre(r["fecha_originacion"], r["FECHA_DE_BAJA"]), axis=1
    )
    clientes["mes_graduacion"] = clientes.apply(
        lambda r: meses_entre(r["fecha_originacion"], r["FECHA_DE_GRADUACION"]), axis=1
    )

    clientes["estado_cliente"] = np.select(
        [
            clientes["FECHA_DE_GRADUACION"].notna(),
            clientes["FECHA_DE_BAJA"].notna(),
        ],
        [
            "Graduado",
            "Baja",
        ],
        default="Activo"
    )

    return clientes


def aplicar_filtros(df, clientes, anios_sel, tipos_sel, credito_sel):
    df_f = df.copy()
    clientes_f = clientes.copy()

    if anios_sel:
        df_f = df_f[df_f["Anio_Cosecha"].isin(anios_sel)]
        clientes_f = clientes_f[clientes_f["Anio_Cosecha"].isin(anios_sel)]

    if tipos_sel:
        df_f = df_f[df_f["Tipo de comision"].isin(tipos_sel)]

    if credito_sel != "Todos":
        clientes_f = clientes_f[clientes_f["Credito"] == credito_sel]
        refs_validas = set(clientes_f["Referencia"])
        df_f = df_f[df_f["Referencia"].isin(refs_validas)]

    return df_f, clientes_f


def resumen_anual(clientes: pd.DataFrame, df_cobros: pd.DataFrame) -> pd.DataFrame:
    base_clientes = clientes.groupby("Anio_Cosecha", dropna=False).agg(
        clientes=("Referencia", "nunique"),
        condonados=("Condonado", lambda x: (x == "Sí").sum()),
        graduados=("estado_cliente", lambda x: (x == "Graduado").sum()),
        bajas=("estado_cliente", lambda x: (x == "Baja").sum()),
        activos=("estado_cliente", lambda x: (x == "Activo").sum()),
        deuda_total=("Deuda_inicial_Fija", "sum"),
    ).reset_index()

    base_cobros = df_cobros.groupby("Anio_Cosecha", dropna=False).agg(
        cobro_total=("Monto", "sum"),
        transacciones=("Monto", "size"),
    ).reset_index()

    out = base_clientes.merge(base_cobros, on="Anio_Cosecha", how="left")
    out["cobro_total"] = out["cobro_total"].fillna(0)
    out["transacciones"] = out["transacciones"].fillna(0).astype(int)
    out["pct_bajas"] = np.where(out["clientes"] > 0, out["bajas"] / out["clientes"], 0)
    out["pct_graduados"] = np.where(out["clientes"] > 0, out["graduados"] / out["clientes"], 0)
    out["pct_condonados"] = np.where(out["clientes"] > 0, out["condonados"] / out["clientes"], 0)
    return out.sort_values("Anio_Cosecha")


def curva_cobros_anual(df_f: pd.DataFrame, clientes_f: pd.DataFrame, modo="Suma total", acumulado=False):
    base_deuda = (
        clientes_f.groupby(["Anio_Cosecha", "Condonado"], dropna=False)
        .agg(
            clientes=("Referencia", "nunique"),
            deuda_total=("Deuda_inicial_Fija", "sum")
        )
        .reset_index()
    )

    cobros = (
        df_f.groupby(["Anio_Cosecha", "Mes_Cobro", "Condonado"], dropna=False)
        .agg(
            monto_total=("Monto", "sum"),
            clientes_con_cobro=("Referencia", "nunique")
        )
        .reset_index()
        .sort_values(["Anio_Cosecha", "Condonado", "Mes_Cobro"])
    )

    cobros = cobros.merge(base_deuda, on=["Anio_Cosecha", "Condonado"], how="left")

    if modo == "Suma total":
        cobros["Valor"] = cobros["monto_total"]
    elif modo == "Promedio por cliente":
        cobros["Valor"] = np.where(cobros["clientes"] > 0, cobros["monto_total"] / cobros["clientes"], 0)
    elif modo == "Promedio ponderado por deuda":
        cobros["Valor"] = np.where(cobros["deuda_total"] > 0, cobros["monto_total"] / cobros["deuda_total"], 0)

    if acumulado:
        cobros["Valor"] = cobros.groupby(["Anio_Cosecha", "Condonado"])["Valor"].cumsum()

    return cobros


def curva_evento_anual(clientes_f: pd.DataFrame, columna_mes: str):
    tmp = clientes_f[clientes_f[columna_mes].notna()].copy()
    if tmp.empty:
        return tmp

    tmp[columna_mes] = tmp[columna_mes].astype(int)

    eventos = (
        tmp.groupby(["Anio_Cosecha", columna_mes, "Condonado"], dropna=False)["Referencia"]
        .nunique()
        .reset_index(name="clientes_evento")
        .sort_values(["Anio_Cosecha", "Condonado", columna_mes])
    )

    base = (
        clientes_f.groupby(["Anio_Cosecha", "Condonado"], dropna=False)["Referencia"]
        .nunique()
        .reset_index(name="base_clientes")
    )

    eventos = eventos.merge(base, on=["Anio_Cosecha", "Condonado"], how="left")
    eventos["pct"] = np.where(eventos["base_clientes"] > 0, eventos["clientes_evento"] / eventos["base_clientes"], 0)
    return eventos


st.title("📊 Dashboard Anual de Cobros, Bajas y Graduaciones")

df, origen_msg = cargar_datos_fijos()
if df is None:
    st.error("No encontré la base fija. Debes subir data/df_cc.parquet o data/df_cc.csv al repo.")
    st.stop()

with st.expander("Diagnóstico de columnas", expanded=False):
    st.write("Fuente:", origen_msg)
    st.write("Columnas detectadas:")
    st.write(list(df.columns))
    st.write("Vista previa:")
    st.dataframe(df.head(10), use_container_width=True)

clientes = construir_clientes(df)

st.sidebar.header("Filtros")

anios_disponibles = sorted([x for x in df["Anio_Cosecha"].dropna().unique().tolist() if x != "Sin año"])
tipos_disponibles = sorted(df["Tipo de comision"].dropna().astype(str).unique().tolist())
credito_opts = ["Todos"] + sorted(clientes["Credito"].dropna().astype(str).unique().tolist())

anios_sel = st.sidebar.multiselect("Año de cosecha", anios_disponibles, default=anios_disponibles)
tipos_sel = st.sidebar.multiselect("Tipos de comisión a incluir", tipos_disponibles, default=tipos_disponibles)
credito_sel = st.sidebar.selectbox("Crédito", credito_opts, index=0)

modo_metrica = st.sidebar.selectbox(
    "Métrica de cobro",
    ["Suma total", "Promedio por cliente", "Promedio ponderado por deuda"]
)

ver_pct_eventos = st.sidebar.checkbox("Ver bajas/graduaciones en porcentaje", value=True)

df_f, clientes_f = aplicar_filtros(df, clientes, anios_sel, tipos_sel, credito_sel)
resumen = resumen_anual(clientes_f, df_f)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clientes únicos", f"{clientes_f['Referencia'].nunique():,}")
k2.metric("Cobro total", f"{df_f['Monto'].sum():,.0f}")
k3.metric("Bajas", f"{(clientes_f['estado_cliente'] == 'Baja').sum():,}")
k4.metric("Graduados", f"{(clientes_f['estado_cliente'] == 'Graduado').sum():,}")
k5.metric("Condonados", f"{(clientes_f['Condonado'] == 'Sí').sum():,}")

st.subheader("Resumen anual")
st.dataframe(resumen, use_container_width=True)

tabs = st.tabs(["Cobro mensual", "Cobro acumulado", "Bajas", "Graduaciones", "Detalle clientes"])

with tabs[0]:
    cobro_mensual = curva_cobros_anual(df_f, clientes_f, modo=modo_metrica, acumulado=False)
    if not cobro_mensual.empty:
        fig = px.line(
            cobro_mensual,
            x="Mes_Cobro",
            y="Valor",
            color="Anio_Cosecha",
            line_dash="Condonado",
            markers=True,
            title="Cobros por mes desde originación, agrupado por año"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cobro_mensual, use_container_width=True)
    else:
        st.info("No hay datos para la curva de cobro mensual.")

with tabs[1]:
    cobro_acum = curva_cobros_anual(df_f, clientes_f, modo=modo_metrica, acumulado=True)
    if not cobro_acum.empty:
        fig = px.line(
            cobro_acum,
            x="Mes_Cobro",
            y="Valor",
            color="Anio_Cosecha",
            line_dash="Condonado",
            markers=True,
            title="Cobro acumulado por mes desde originación, agrupado por año"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cobro_acum, use_container_width=True)
    else:
        st.info("No hay datos para la curva de cobro acumulado.")

with tabs[2]:
    bajas = curva_evento_anual(clientes_f, "mes_baja")
    if not bajas.empty:
        ycol = "pct" if ver_pct_eventos else "clientes_evento"
        fig = px.line(
            bajas,
            x="mes_baja",
            y=ycol,
            color="Anio_Cosecha",
            line_dash="Condonado",
            markers=True,
            title="Bajas por mes desde originación, agrupado por año"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bajas, use_container_width=True)
    else:
        st.info("No hay bajas para los filtros seleccionados.")

with tabs[3]:
    grads = curva_evento_anual(clientes_f, "mes_graduacion")
    if not grads.empty:
        ycol = "pct" if ver_pct_eventos else "clientes_evento"
        fig = px.line(
            grads,
            x="mes_graduacion",
            y=ycol,
            color="Anio_Cosecha",
            line_dash="Condonado",
            markers=True,
            title="Graduaciones por mes desde originación, agrupado por año"
        )
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grads, use_container_width=True)
    else:
        st.info("No hay graduaciones para los filtros seleccionados.")

with tabs[4]:
    st.dataframe(clientes_f, use_container_width=True))
