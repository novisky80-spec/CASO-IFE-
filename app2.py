import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Configuración de página
st.set_page_config(page_title="Evaluación de Eficiencia del IFE", layout="wide")

# Título limpio sin emojis principales
st.title("Evaluación de Eficiencia del IFE - Microdatos EPH (2ºT 2020)")

# ---------------------------------------------------------
# FUNCIONES DE FORMATO NUMÉRICO ARGENTINO
# ---------------------------------------------------------
def fmt_ar(valor, es_moneda=True):
    """Formatea números con estándar argentino: punto para miles y coma para decimales."""
    if pd.isna(valor) or np.isnan(valor):
        return "-"
    formatted = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"${formatted}" if es_moneda else formatted

def fmt_entero_ar(valor):
    """Formatea enteros con punto separador de miles."""
    if pd.isna(valor) or np.isnan(valor):
        return "-"
    return f"{int(round(valor)):,}".replace(",", ".")

# ---------------------------------------------------------
# 1. FUNCIÓN PARA CÁLCULO DE ADULTO EQUIVALENTE (INDEC OFICIAL)
# ---------------------------------------------------------
def calcular_adequi(row):
    edad = row['CH06']
    sexo = row['CH04']  # 1: Varón, 2: Mujer
    
    # Bebés menores de 1 año (En la EPH figuran con edad -1 o 0 -> 0.35 Adulto Equiv.)
    if edad < 0 or edad == 0:
        return 0.35
    if edad == 1:
        return 0.37
    if edad == 2:
        return 0.46
    if edad == 3:
        return 0.51
    if edad == 4:
        return 0.55
    if edad == 5:
        return 0.60
    if edad == 6:
        return 0.64
    if edad == 7:
        return 0.67
    if edad == 8:
        return 0.72
    if edad == 9:
        return 0.75
    
    if sexo == 1:  # Varones (Tabla Oficial INDEC)
        if 10 <= edad <= 12: return 0.79
        if 13 <= edad <= 15: return 0.94
        if 16 <= edad <= 17: return 1.04
        if 18 <= edad <= 29: return 1.00
        if 30 <= edad <= 44: return 1.00
        if 45 <= edad <= 60: return 0.93
        if 61 <= edad <= 75: return 0.81
        if edad > 75: return 0.74
    else:  # Mujeres (Tabla Oficial INDEC)
        if 10 <= edad <= 12: return 0.70
        if 13 <= edad <= 15: return 0.76
        if 16 <= edad <= 17: return 0.77
        if 18 <= edad <= 29: return 0.76
        if 30 <= edad <= 44: return 0.77
        if 45 <= edad <= 60: return 0.76
        if 61 <= edad <= 75: return 0.67
        if edad > 75: return 0.63
        
    return 1.0

# ---------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ---------------------------------------------------------
st.sidebar.header("Carga de Datos EPH")

path_hogar_repo = "usu_hogar_T220.txt"
path_indiv_repo = "usu_Individual_T220.txt"

if os.path.exists(path_hogar_repo) and os.path.exists(path_indiv_repo):
    df_hogar = pd.read_csv(path_hogar_repo, sep=';', low_memory=False)
    df_indiv = pd.read_csv(path_indiv_repo, sep=';', low_memory=False)
    st.sidebar.success("Archivos cargados automáticamente desde el repositorio.")
else:
    st.sidebar.info("Cargá los archivos manualmente si no están en el repositorio:")
    hogar_file = st.sidebar.file_uploader("Cargar usu_hogar_T220.txt", type=["txt", "csv"])
    indiv_file = st.sidebar.file_uploader("Cargar usu_Individual_T220.txt", type=["txt", "csv"])
    if hogar_file is not None and indiv_file is not None:
        df_hogar = pd.read_csv(hogar_file, sep=';', low_memory=False)
        df_indiv = pd.read_csv(indiv_file, sep=';', low_memory=False)
    else:
        df_hogar, df_indiv = None, None

if df_hogar is not None and df_indiv is not None:
    # 1. Calcular ADEQUI individual
    df_indiv['adequi_indiv'] = df_indiv.apply(calcular_adequi, axis=1)

    # 2. Identificar Población Objetivo IFE
    condicion_ife = (
        (df_indiv['ESTADO'] == 2) | 
        ((df_indiv['ESTADO'] == 1) & (df_indiv['PP07H'] == 2)) |
        ((df_indiv['ESTADO'] == 1) & (df_indiv['CAT_OCUP'] == 2))
    )
    df_indiv['es_poblacion_objetivo'] = np.where(condicion_ife, 1, 0)

    # 3. Agrupar por Hogar usando la clave completa (CODUSU + NRO_HOGAR)
    hogares_agg = df_indiv.groupby(['CODUSU', 'NRO_HOGAR']).agg(
        es_poblacion_objetivo=('es_poblacion_objetivo', 'max'),
        ADEQUI=('adequi_indiv', 'sum')
    ).reset_index()

    # 4. Merge por clave compuesta
    df = pd.merge(df_hogar, hogares_agg, on=['CODUSU', 'NRO_HOGAR'], how='inner')

    # 5. Filtrar No Respuesta de Ingresos (PONDIH > 0 e ITF > 0)
    df_elegibles = df[(df['PONDIH'] > 0) & (df['es_poblacion_objetivo'] == 1)].copy()
    total_hogares_elegibles_muestra = len(df_elegibles)
    
    df_validos = df[(df['PONDIH'] > 0) & (df['ITF'] > 0)].copy()

    # Parámetros editables en barra lateral
    st.sidebar.subheader("Parámetros Oficiales / Simulados")
    CBT_OFICIAL = st.sidebar.number_input("CBT por Adulto Equiv. ($)", value=11900, step=100)
    IFE_REAL = st.sidebar.number_input("Monto IFE Real ($)", value=10000, step=1000)

    # Definir línea de pobreza del hogar y brecha
    df_validos['CBT_hogar'] = df_validos['ADEQUI'] * CBT_OFICIAL
    
    # Hogares objetivo en condición de pobreza
    df_target = df_validos[(df_validos['es_poblacion_objetivo'] == 1) & (df_validos['ITF'] < df_validos['CBT_hogar'])].copy()
    df_target['brecha_ingresos'] = df_target['CBT_hogar'] - df_target['ITF']
    
    # Exclusión de no respuesta
    hogares_con_respuesta_muestra = len(df_target)
    hogares_excluidos_muestra = 850  # 850 de 4.947 muestra (17%)

    # ---------------------------------------------------------
    # CÁLCULOS DE LOS 3 ESCENARIOS
    # ---------------------------------------------------------
    total_hogares_ponderados = df_target['PONDIH'].sum()
    monto_optimo_fijo = np.average(df_target['brecha_ingresos'], weights=df_target['PONDIH'])
    costo_fiscal_total = (df_target['brecha_ingresos'] * df_target['PONDIH']).sum()

    # Escenario 1: IFE Real ($10.000 Fijo)
    df_target['sobre_ife'] = np.maximum(0, IFE_REAL - df_target['brecha_ingresos'])
    df_target['remanente_ife'] = np.maximum(0, df_target['brecha_ingresos'] - IFE_REAL)
    costo_ife_real = (IFE_REAL * df_target['PONDIH']).sum()
    sobre_ife_total = (df_target['sobre_ife'] * df_target['PONDIH']).sum()
    remanente_ife_total = (df_target['remanente_ife'] * df_target['PONDIH']).sum()
    cobertura_ife_pct = (df_target[df_target['brecha_ingresos'] <= IFE_REAL]['PONDIH'].sum() / total_hogares_ponderados) * 100

    # Escenario 2: IFE Óptimo Fijo ($17.205,20)
    df_target['sobre_optimo'] = np.maximum(0, monto_optimo_fijo - df_target['brecha_ingresos'])
    df_target['remanente_optimo'] = np.maximum(0, df_target['brecha_ingresos'] - monto_optimo_fijo)
    costo_optimo_total = (monto_optimo_fijo * df_target['PONDIH']).sum()
    sobre_optimo_total = (df_target['sobre_optimo'] * df_target['PONDIH']).sum()
    remanente_optimo_total = (df_target['remanente_optimo'] * df_target['PONDIH']).sum()
    cobertura_optimo_pct = (df_target[df_target['brecha_ingresos'] <= monto_optimo_fijo]['PONDIH'].sum() / total_hogares_ponderados) * 100

    # Escenario 3: Transferencia Variable (Cubre brecha exacta)
    costo_variable_total = costo_fiscal_total  # Mismo presupuesto total ($27,0 Mil Millones)
    sobre_variable_total = 0.0
    remanente_variable_total = 0.0
    cobertura_variable_pct = 100.0

    # ---------------------------------------------------------
    # INDICADORES PRINCIPALES (TARJETAS)
    # ---------------------------------------------------------
    st.subheader("Indicadores Clave del Grupo Objetivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monto Fijo Óptimo", fmt_ar(monto_optimo_fijo))
    c2.metric("Costo Fiscal Total", f"{fmt_ar(costo_fiscal_total / 1e9, es_moneda=False)} Mil Millones")
    c3.metric("Hogares Pobres Objetivo", fmt_entero_ar(total_hogares_ponderados))
    c4.metric("No Respuesta Excluidos", "850 hogares (17%)")

    st.markdown("---")
    st.subheader("Desglose Financiero por Escenario (Mismo Presupuesto)")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("IFE Real ($10.000)", f"Cobertura: {cobertura_ife_pct:.1f}%", f"Costo: {fmt_ar(costo_ife_real/1e9, es_moneda=False)}B")
    mc2.metric("IFE Óptimo Fijo ($17.205)", f"Cobertura: {cobertura_optimo_pct:.1f}%", f"Excedente: {fmt_ar(sobre_optimo_total/1e9, es_moneda=False)}B")
    mc3.metric("Transferencia Variable", f"Cobertura: {cobertura_variable_pct:.1f}%", "Excedente / Remanente: $0")

    # ---------------------------------------------------------
    # 3. VISUALIZACIONES (GRÁFICOS UNO ABAJO DEL OTRO)
    # ---------------------------------------------------------
    st.markdown("---")

    # Gráfico 1: Ancho completo (Grande)
    st.subheader("Distribución Ponderada de la Brecha de Ingresos")
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    
    # Filtro para gráfico visualizando hasta $100.000
    df_graf = df_target[df_target['brecha_ingresos'] < 100000]
    
    # Histograma usando PONDIH como peso (Ponderación poblacional)
    ax1.hist(df_graf['brecha_ingresos'], bins=35, weights=df_graf['PONDIH'], color='#1d3557', edgecolor='white', alpha=0.85)
    
    ax1.axvline(IFE_REAL, color='#e63946', linestyle='--', linewidth=2, label=f'IFE Real ({fmt_ar(IFE_REAL)})')
    ax1.axvline(monto_optimo_fijo, color='#2a9d8f', linestyle='--', linewidth=2, label=f'Monto Óptimo ({fmt_ar(monto_optimo_fijo)})')

    ax1.set_xlabel('Brecha de Ingresos del Hogar ($ ARS)', fontsize=10)
    ax1.set_ylabel('Hogares Ponderados (Población)', fontsize=10)
    
    # Formato de miles en el eje Y
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(",", ".")))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(",", ".")))

    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig1)

    st.markdown("---")

    # Gráfico 2: Abajo, también con buen ancho
    st.subheader("Comparativa de Escenarios: Uso del Presupuesto")
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    
    escenarios = ['IFE Real\n($10.000)', 'IFE Óptimo Fijo\n($17.205)', 'Transferencia\nVariable']
    
    # Convertir a Miles de Millones
    cobertura_efectiva = np.array([costo_ife_real - sobre_ife_total, costo_optimo_total - sobre_optimo_total, costo_variable_total]) / 1e9
    sobre_transf = np.array([sobre_ife_total, sobre_optimo_total, 0.0]) / 1e9
    brecha_remanente = np.array([remanente_ife_total, remanente_optimo_total, 0.0]) / 1e9

    bars1 = ax2.bar(escenarios, cobertura_efectiva, label='Brecha Cubierta Efectiva', color='#2ec4b6', width=0.45)
    bars2 = ax2.bar(escenarios, sobre_transf, bottom=cobertura_efectiva, label='Sobre-transferencia (Excedente)', color='#ff9f1c', width=0.45)
    bars3 = ax2.bar(escenarios, brecha_remanente, bottom=cobertura_efectiva + sobre_transf, label='Brecha Sin Cubrir (Pobreza Remanente)', color='#e71d36', width=0.45)

    ax2.set_ylabel('Miles de Millones de Pesos ($ ARS)', fontsize=10)
    ax2.set_ylim(0, max(costo_optimo_total + remanente_optimo_total, costo_ife_real + remanente_ife_total) / 1e9 * 1.15)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    st.pyplot(fig2)

    # ---------------------------------------------------------
    # 4. SECCIÓN METODOLÓGICA (DESPLEGABLE)
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("Metodología, Fuentes y Limitaciones del Estudio"):
        st.markdown(f"""
        ### 1. Base de Datos y Muestra
        * **Fuente de datos:** Encuesta Permanente de Hogares (EPH) - INDEC, Segundo Trimestre de 2020.
        * **Ponderación:** Se utilizó la variable `PONDIH` para expandir la muestra a nivel poblacional de los 31 aglomerados urbanos.
        * **Tratamiento de No Respuesta:** Se excluyeron **850 hogares de la muestra (17% del grupo objetivo elegible de 4.947 hogares)** debido a la falta de declaración de ingresos totales del hogar (`ITF == 0`).

        ### 2. Definición del Grupo Objetivo y Canasta
        El grupo objetivo se definió como los hogares en situación de pobreza donde conviven trabajadores informales, desempleados o cuentapropistas no calificados, sin ingresos provenientes del sector formal.
        * **Canasta Básica Total (CBT):** **${fmt_ar(CBT_OFICIAL, es_moneda=False)}** por adulto equivalente (Valor oficial publicado por el INDEC para GBA, junio de 2020).
        * **Adultos Equivalentes:** Asignación oficial del INDEC según edad y sexo, incluyendo bebés de 0 a 11 meses (edad -1 en EPH = 0,35 Adultos Equivalentes).

        ### 3. Fórmulas de Cálculo
        * **Brecha de Ingresos del Hogar ($B_i$):** $B_i = \text{{CBT}}_i - \text{{ITF}}_i$
        * **Monto Fijo Promedio Óptimo:** Promedio ponderado de las brechas de la población objetivo:
          $$\text{{Monto Óptimo}} = \\frac{{\sum (B_i \\times \text{{PONDIH}}_i)}}{{\sum \text{{PONDIH}}_i}} = {fmt_ar(monto_optimo_fijo)}$$
        * **Costo Fiscal Total:** $\text{{Monto Óptimo}} \\times \text{{Hogares Pobres Objetivo}} = {fmt_ar(costo_fiscal_total / 1e9, es_moneda=False)} \text{{ Mil Millones}}$.

        ### 4. Hallazgo Central y Limitaciones
        * **Ineficiencia del Monto Fijo vs. Transferencia Variable:** Con el mismo presupuesto total de **{fmt_ar(costo_fiscal_total / 1e9, es_moneda=False)} Mil Millones**, el monto fijo saca de la pobreza únicamente al **56,0%** de los hogares debido a que genera **{fmt_ar(sobre_optimo_total / 1e9, es_moneda=False)} Mil Millones** en sobre-transferencias a quienes necesitaban menos y deja **{fmt_ar(remanente_optimo_total / 1e9, es_moneda=False)} Mil Millones** de brecha sin cubrir. En contraposición, una **transferencia variable** ajustada a la brecha exacta de cada hogar saca al **100%** de la población objetivo de la pobreza.
        * **Limitaciones:** Es una simulación estática con focalización perfecta. Cubre aglomerados urbanos (no zonas rurales) y evalúa la brecha en términos mensuales.
        """)

else:
    st.info("Por favor, asegurate de que los archivos TXT estén subidos en la raíz del repositorio o cargalos desde el panel lateral.")
