import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools

st.set_page_config(layout="wide")
st.title("🎮 Dashboard Interactivo: Juegos con Lootboxes en Steam")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("processed_games_final.csv")
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['Release Year'] = df['Release date'].dt.year
    df['FreeToPlay'] = df['FreeToPlay'].astype(bool)
    df['HasLootboxKeywords'] = df['HasLootboxKeywords'].astype(bool)
    return df

df = cargar_datos()
df_loot = df[df['HasLootboxKeywords'] == True]

st.sidebar.header("🔍 Filtros globales")

# Modelo de negocio
tipo_juego = st.sidebar.radio("Tipo de juego", ["Todos", "Solo F2P", "Solo P2P"])
if tipo_juego == "Solo F2P":
    df_loot = df_loot[df_loot['FreeToPlay'] == True]
elif tipo_juego == "Solo P2P":
    df_loot = df_loot[df_loot['FreeToPlay'] == False]

# Géneros
generos_invalidos = ['Free to Play', 'Multiplayer', 'Single-player', 'Controller Support']
generos_all = df_loot['Genres'].dropna().str.split(',')
generos_flat = [g.strip() for g in itertools.chain.from_iterable(generos_all) if g.strip() not in generos_invalidos]
generos_unicos = sorted(set(generos_flat))
generos_seleccionados = st.sidebar.multiselect("Géneros", generos_unicos)
if generos_seleccionados:
    df_loot = df_loot[df_loot['Genres'].fillna('').apply(lambda x: any(g in x for g in generos_seleccionados))]

# Tags
tags_all = df_loot['Tags'].dropna().str.split(',')
tags_flat = [t.strip() for t in itertools.chain.from_iterable(tags_all)]
tags_unicos = sorted(set(tags_flat))
tags_seleccionados = st.sidebar.multiselect("Etiquetas (Tags)", tags_unicos)
if tags_seleccionados:
    df_loot = df_loot[df_loot['Tags'].fillna('').apply(lambda x: any(t in x for t in tags_seleccionados))]

# Año
min_year, max_year = int(df_loot['Release Year'].min()), int(df_loot['Release Year'].max())
rango_anios = st.sidebar.slider("Rango de años", min_year, max_year, (min_year, max_year))
df_loot = df_loot[(df_loot['Release Year'] >= rango_anios[0]) & (df_loot['Release Year'] <= rango_anios[1])]

# Peak CCU
ccu_min, ccu_max = int(df_loot['Peak CCU'].min()), int(df_loot['Peak CCU'].max())
ccu_rango = st.sidebar.slider("Peak CCU", ccu_min, ccu_max, (ccu_min, ccu_max))
df_loot = df_loot[df_loot['Peak CCU'].between(ccu_rango[0], ccu_rango[1])]

# Precio
precio_min = float(df_loot['Price'].min())
precio_max = float(df_loot['Price'].max())
rango_precio = st.sidebar.slider("Precio (USD)", precio_min, precio_max, (precio_min, precio_max))
df_loot = df_loot[df_loot['Price'].between(rango_precio[0], rango_precio[1])]

# Edad requerida
edad_range = st.sidebar.slider("Edad mínima requerida", int(df_loot['Required age'].min()), int(df_loot['Required age'].max()), (0, 18))
df_loot = df_loot[df_loot['Required age'].between(edad_range[0], edad_range[1])]

# Tiempo promedio jugado
if 'Average playtime forever' in df_loot.columns:
    tmin = int(df_loot['Average playtime forever'].min())
    tmax = int(df_loot['Average playtime forever'].max())
    rango_playtime = st.sidebar.slider("Tiempo promedio jugado (min)", tmin, tmax, (tmin, tmax))
    df_loot = df_loot[df_loot['Average playtime forever'].between(rango_playtime[0], rango_playtime[1])]

# Plataformas
plataformas = st.sidebar.multiselect("Plataformas", ['Windows', 'Mac', 'Linux'])
if 'Windows' in plataformas:
    df_loot = df_loot[df_loot['Windows'] == True]
if 'Mac' in plataformas:
    df_loot = df_loot[df_loot['Mac'] == True]
if 'Linux' in plataformas:
    df_loot = df_loot[df_loot['Linux'] == True]

# Métrica resumen
st.metric("🎯 Juegos que cumplen los filtros", len(df_loot))

# Hipótesis 1
st.subheader("🧪 Hipótesis 1: F2P con lootboxes atraen más jugadores que P2P")
f2p_filtered = df_loot[df_loot['FreeToPlay'] == True]
p2p_filtered = df_loot[df_loot['FreeToPlay'] == False]

if not f2p_filtered.empty or not p2p_filtered.empty:
    fig1, ax1 = plt.subplots()
    sns.barplot(
        x=["F2P", "P2P"],
        y=[
            f2p_filtered['Peak CCU'].mean() if not f2p_filtered.empty else 0,
            p2p_filtered['Peak CCU'].mean() if not p2p_filtered.empty else 0
        ],
        ax=ax1,
        palette=["lightgreen", "salmon"]
    )
    ax1.set_ylabel("Peak CCU promedio")
    ax1.set_title("Comparación de jugadores concurrentes")
    st.pyplot(fig1)
else:
    st.info("No hay datos suficientes para comparar F2P vs P2P con los filtros actuales.")

# Hipótesis 2
st.subheader("📈 Hipótesis 2: Evolución temporal de juegos con lootboxes")
total_por_anio = df[(df['Release Year'] >= rango_anios[0]) & (df['Release Year'] <= rango_anios[1])].groupby('Release Year').size()
lootbox_por_anio = df_loot.groupby('Release Year').size()
proporcion = (lootbox_por_anio / total_por_anio * 100).dropna()

if not proporcion.empty:
    fig2, ax2 = plt.subplots()
    ax2.plot(proporcion.index, proporcion.values, marker='o', color='orange')
    ax2.set_title("Porcentaje de juegos con lootboxes por año")
    ax2.set_xlabel("Año")
    ax2.set_ylabel("Porcentaje (%)")
    ax2.grid(True)
    st.pyplot(fig2)
else:
    st.info("No hay datos suficientes para mostrar evolución temporal.")

# Hipótesis 3
st.subheader("🎮 Hipótesis 3: Géneros con más jugadores concurrentes")
genero_ccu = defaultdict(int)
for _, row in df_loot[['Genres', 'Peak CCU']].dropna().iterrows():
    generos = [g.strip() for g in str(row['Genres']).split(',') if g.strip() not in generos_invalidos]
    for g in generos:
        genero_ccu[g] += row['Peak CCU']

if genero_ccu:
    top_generos = sorted(genero_ccu.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, values = zip(*top_generos)
    fig3, ax3 = plt.subplots()
    ax3.bar(labels, values, color='mediumpurple')
    ax3.set_title("Peak CCU total por género")
    ax3.set_ylabel("Jugadores concurrentes acumulados")
    ax3.set_xticklabels(labels, rotation=45)
    st.pyplot(fig3)
else:
    st.info("No hay datos suficientes para mostrar géneros.")

# Mostrar tabla
st.subheader("📋 Juegos filtrados")
st.dataframe(df_loot[['Name', 'Genres', 'Tags', 'Price', 'Peak CCU', 'Release Year', 'FreeToPlay']].sort_values('Peak CCU', ascending=False), use_container_width=True)
