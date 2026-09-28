import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Evaluación de Eficiencia del IFE", layout="wide")

st.title("📊 Evaluación de Eficiencia del IFE - Microdatos EPH (2ºT 2020)")

# ---------------------------------------------------------
# 1. FUNCIÓN PARA CÁLCULO DE ADULTO EQUIVALENTE (INDEC)
# ---------------------------------------------------------
def calcular_adequi(row):
    edad = row['CH06']
    sexo = row['CH04']  # 1: Varón, 2: Mujer
    
    if edad < 0:
        return 0
    if edad == 0:
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
        return 0.66
    if edad == 8:
        return 0.68
    if edad == 9:
        return 0.69
    
    if sexo == 1: # Varones
        if 10 <= edad <= 12: return 0.70
        if 13 <= edad <= 14: return 0.76
        if 15 <= edad <= 17: return 0.83
        if 18 <= edad <= 29: return 1.00
        if 30 <= edad <= 45: return 1.00
        if 46 <= edad <= 60: return 1.00
        if 61 <= edad <= 75: return 0.83
        if edad > 75: return 0.74
    else: # Mujeres
        if 10 <= edad <= 12: return 0.70
        if 13 <= edad <= 14: return 0.70
        if 15 <= edad <= 17: return 0.70
        if 18 <= edad <= 29: return 0.77
        if 30 <= edad <= 45: return 0.77
        if 46 <= edad <= 60: return 0.76
        if 61 <= edad <= 75: return 0.67
        if edad > 75: return 0.63
        
    return 1.0

# ---------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ---------------------------------------------------------
st.sidebar.header("Carga de Datos EPH")

# Intentar cargar directamente desde el repositorio si existen
path_hogar_repo = "usu_hogar_T220.txt"
path_indiv_repo = "usu_Individual_T220.txt"

if os.path.exists(path_hogar_repo) and os.path.exists(path_indiv_repo):
    df_hogar = pd.read_csv(path_hogar_repo, sep=';', low_memory=False)
    df_indiv = pd.read_csv(path_indiv_repo, sep=';', low_memory=False)
    st.sidebar.success("✅ Archivos cargados automáticamente desde el repositorio.")
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

    # 3. Agrupar por Hogar usando CODUSU + NRO_HOGAR
    hogares_agg = df_indiv.groupby(['CODUSU', 'NRO_HOGAR']).agg(
        es_poblacion_objetivo=('es_poblacion_objetivo', 'max'),
        ADEQUI=('adequi_indiv', 'sum')
    ).reset_index()

    # 4. Merge por clave compuesta
    df = pd.merge(df_hogar, hogares_agg, on=['CODUSU', 'NRO_HOGAR'], how='inner')

    # 5. Filtrar No Respuesta de Ingresos (PONDIH > 0 e ITF > 0)
    df_validos = df[(df['PONDIH'] > 0) & (df['ITF'] > 0)].copy()

    # Parámetros editables en barra lateral
    st.sidebar.subheader("Parámetros Oficiales / Simulados")
    CBT_OFICIAL = st.sidebar.number_input("CBT por Adulto Equiv. ($)", value=11900, step=100)
    IFE_REAL = st.sidebar.number_input("Monto IFE Real ($)", value=10000, step=1000)

    # Definir línea de pobreza del hogar y Brecha
    df_validos['CBT_hogar'] = df_validos['ADEQUI'] * CBT_OFICIAL
    
    # Hogares objetivo en condición de pobreza
    df_target = df_validos[(df_validos['es_poblacion_objetivo'] == 1) & (df_validos['ITF'] < df_validos['CBT_hogar'])].copy()
    df_target['brecha_ingresos'] = df_target['CBT_hogar'] - df_target['ITF']

    # Métricas clave
    monto_optimo_fijo = np.average(df_target['brecha_ingresos'], weights=df_target['PONDIH'])
    costo_fiscal_total = (df_target['brecha_ingresos'] * df_target['PONDIH']).sum()

    st.subheader("📌 Indicadores Clave")
    c1, c2, c3 = st.columns(3)
    c1.metric("Monto Fijo Promedio Óptimo", f"${monto_optimo_fijo:,.2f}")
    c2.metric("Costo Fiscal Total Mensual", f"${costo_fiscal_total / 1e9:,.2f} Mil Millones")
    c3.metric("Hogares Objetivo Pobres Analizados", f"{df_target['PONDIH'].sum():,.0f}")

    # ---------------------------------------------------------
    # 3. VISUALIZACIONES EN DOS COLUMNAS
    # ---------------------------------------------------------
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.subheader("📈 Histograma de Brechas de Ingresos")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        brechas_grafico = df_target[df_target['brecha_ingresos'] < 100000]['brecha_ingresos']

        ax1.hist(brechas_grafico, bins=40, color='#3498db', edgecolor='black', alpha=0.7)
        
        ax1.axvspan(0, IFE_REAL, color='red', alpha=0.15, label='Sobre-cobertura')
        ax1.axvspan(IFE_REAL, 100000, color='orange', alpha=0.15, label='Sub-cobertura')

        ax1.axvline(IFE_REAL, color='red', linestyle='--', linewidth=2, label=f'IFE Real (${IFE_REAL:,.0f})')
        ax1.axvline(monto_optimo_fijo, color='green', linestyle='--', linewidth=2, label=f'Óptimo (${monto_optimo_fijo:,.0f})')

        ax1.set_xlabel('Brecha de Ingresos ($)', fontsize=10)
        ax1.set_ylabel('Frecuencia de Hogares', fontsize=10)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig1)

    with col_graf2:
        st.subheader("📊 Cobertura de la Brecha por Escenario")
        cubiertos_ife = df_target[df_target['brecha_ingresos'] <= IFE_REAL]['PONDIH'].sum()
        cubiertos_optimo = df_target[df_target['brecha_ingresos'] <= monto_optimo_fijo]['PONDIH'].sum()
        total_pobres = df_target['PONDIH'].sum()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        categorias = ['Con IFE Real', 'Con Monto Óptimo']
        porcentajes = [(cubiertos_ife / total_pobres) * 100, (cubiertos_optimo / total_pobres) * 100]

        bars = ax2.bar(categorias, porcentajes, color=['#e74c3c', '#2ecc71'], width=0.5)
        ax2.set_ylabel('% Hogares que Salen de la Pobreza', fontsize=10)
        ax2.set_ylim(0, 100)

        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig2)

else:
    st.info("Por favor, asegurate de que los archivos TXT estén subidos en la raíz del repositorio o cargalos desde el panel lateral.")