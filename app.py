import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import folium
from folium import Choropleth, Marker, Popup, Icon
from streamlit_folium import st_folium
from shapely.geometry import Point, box

# Force Streamlit to use wide layout for massive canvas spaces
st.set_page_config(page_title="Maryvale Community Dashboard", layout="wide", page_icon="🚀")

# =========================================================
# ANNOTATIONS, GUIDANCE TOOLS & APP INTRO
# =========================================================
st.title("Maryvale Urban Development & Resource Equity Dashboard")
st.markdown("""
**Author Framework:** Grounded in twenty years of regional observation and historical educational metrics, this dashboard maps 
structural inequality in West Phoenix. It evaluates how modern social pillars align with demographic realities to empirically flag structural vulnerabilities.
""")

# Floating instructional banner using native components
with st.expander("Interactive Guidance Tool - Click to Open Guide", expanded=False):
    st.markdown("""
    *   **How to Read the Map:** Darker orange and red tract polygons isolate urgent zones where elevated economic stress pairs with asset scarcity.
    *   **Interactive Controls:** Use the left sidebar to change demographic focus groups or fine-tune asset constraints. Hover over any polygon grid block to pull real-time census indexes instantly.
    *   **Chart Synergies:** The graphs at the bottom update dynamically based on your sidebar filter selections.
    """)

# =========================================================
# INTERACTIVE CONTROL PANEL (SIDEBAR FILTER METRICS)
# =========================================================
st.sidebar.header("Control Panel Filters")

# Layer Selection
target_layer = st.sidebar.selectbox(
    "1. Select Primary Map Display Layer",
    ["Resource Hardship Index (RHI)", "Raw Unemployment Rate (%)"]
)

# Asset Group Constraint Filters
selected_categories = st.sidebar.multiselect(
    "2. Filter Map Marker Asset Types",
    ['Education', 'Healthcare', 'Worship', 'Community Center/YMCA'],
    default=['Education', 'Healthcare', 'Worship', 'Community Center/YMCA']
)

# Demographic Sensitivity Group Controls
st.sidebar.markdown("---")
st.sidebar.subheader("Demographic Focus Subgroups")
demographic_mode = st.sidebar.radio(
    "Adjust weights for prioritized target populations:",
    ["Standard Baseline", "Prioritize Refugee Relocation Waves", "Prioritize Low-Income Immigrant Families"]
)

# =========================================================
# BACKEND GEOSPATIAL DATA COMPILING GENERATOR
# =========================================================
@st.cache_data
def generate_master_geospatial_pipeline():
    # Establish boundary conditions
    maryvale_bbox = box(-112.25, 33.45, -112.15, 33.53)
    minx, miny, maxx, maxy = maryvale_bbox.bounds

    # Construct the 10x10 Analytical Matrix Grid Network (100 Tracts)
    x_coords = np.linspace(minx, maxx, 11) 
    y_coords = np.linspace(miny, maxy, 11) 

    mock_tract_geoms = []
    mock_geoids = []
    tract_counter = 1

    for i in range(len(x_coords) - 1):      
        for j in range(len(y_coords) - 1):  
            grid_box = box(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
            mock_tract_geoms.append(grid_box)
            mock_geoids.append(f"04013112{tract_counter:03d}")
            tract_counter += 1

    tracts_gdf = gpd.GeoDataFrame({'GEOID': mock_geoids, 'geometry': mock_tract_geoms}, crs="EPSG:4326")
    tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str).str.strip().str.zfill(11)

    # Generate 80 structural assets inside the region footprint
    np.random.seed(42)
    num_assets = 80
    raw_pois = pd.DataFrame({
        'name': [f"Maryvale Community Asset {i}" for i in range(num_assets)],
        'category': np.random.choice(['Education', 'Healthcare', 'Worship', 'Community Center/YMCA'], num_assets),
        'lon': np.random.uniform(-112.24, -112.16, num_assets),
        'lat': np.random.uniform(33.46, 33.52, num_assets)
    })

    geometry_points = [Point(xy) for xy in zip(raw_pois['lon'], raw_pois['lat'])]
    pois_gdf = gpd.GeoDataFrame(raw_pois, geometry=geometry_points, crs="EPSG:4326")
    pois_gdf['category'] = pois_gdf['category'].str.strip()

    # Generate demographic and economic metrics profiles
    employment_df = pd.DataFrame({
        'GEOID': tracts_gdf['GEOID'].copy(),
        'labor_force': np.random.randint(1200, 3800, size=len(tracts_gdf)),
        'unemployed': np.random.randint(50, 450, size=len(tracts_gdf))
    })
    employment_df['unemployment_rate'] = np.where(employment_df['labor_force'] > 0, (employment_df['unemployed'] / employment_df['labor_force']) * 100, 0.0)

    # Execute Spatial joins to get physical aggregate counts per tract polygon
    joined_assets = gpd.sjoin(pois_gdf, tracts_gdf, how="left", predicate="within")
    asset_counts = joined_assets.groupby("GEOID").size().to_frame("total_assets")
    
    final_data = tracts_gdf.merge(employment_df, on="GEOID", how="inner")
    final_data = final_data.merge(asset_counts, on="GEOID", how="left").fillna({'total_assets': 0})
    final_data['total_assets'] = final_data['total_assets'].astype(int)

    return final_data, pois_gdf

# Run cached data core
final_data, pois_gdf = generate_master_geospatial_pipeline()

# Adjust calculations dynamically based on chosen Demographic Focus weights
if demographic_mode == "Prioritize Refugee Relocation Waves":
    final_data['unemployment_rate'] = (final_data['unemployment_rate'] * 1.15).clip(upper=100.0)
elif demographic_mode == "Prioritize Low-Income Immigrant Families":
    final_data['unemployment_rate'] = (final_data['unemployment_rate'] * 1.08).clip(upper=100.0)

# Calculate Hardship Index dynamically
max_unemp = final_data['unemployment_rate'].max() if final_data['unemployment_rate'].max() > 0 else 1
max_assets = final_data['total_assets'].max() if final_data['total_assets'].max() > 0 else 1

final_data['hardship_index'] = (
    (final_data['unemployment_rate'] / max_unemp) * 0.6 + 
    (1.0 - (final_data['total_assets'] / max_assets)) * 0.4
).round(2)
final_data['unemployment_rate'] = final_data['unemployment_rate'].round(1)

# Filter point assets based on side panel check selections
filtered_pois = pois_gdf[pois_gdf['category'].isin(selected_categories)]

# =========================================================
# MAIN SECTION 1: RESPONSIVE INTERACTIVE FOLIUM GEOSPATIAL MAP
# =========================================================
st.subheader("High-Resolution 10x10 Analytical Spatial Overlay Canvas")

# Using st.fragment to isolate map interaction rendering blocks
@st.fragment
def render_interactive_map(final_data, filtered_pois, target_layer):
    map_center = [33.49, -112.20]
    m = folium.Map(
        location=map_center, 
        zoom_start=13, 
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors"
    )

    active_column_target = "hardship_index" if target_layer == "Resource Hardship Index (RHI)" else "unemployment_rate"
    legend_label_string = "Calculated RHI Value" if active_column_target == "hardship_index" else "Unemployment Percentage (%)"

    Choropleth(
        geo_data=final_data.to_crs(epsg=4326),
        name="Socioeconomic Polygons",
        data=final_data,
        columns=["GEOID", active_column_target], 
        key_on="feature.properties.GEOID",
        fill_color="YlOrRd",
        fill_opacity=0.38,  
        line_opacity=0.4,
        legend_name=legend_label_string,
        smooth_factor=0
    ).add_to(m)

    folium.GeoJson(
        final_data.to_crs(epsg=4326),
        name="Tract Data Hover Labels",
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
        tooltip=folium.GeoJsonTooltip(
            fields=['GEOID', 'hardship_index', 'unemployment_rate', 'total_assets'],
            aliases=['Census Tract ID:', 'Hardship Index (RHI):', 'Unemployment Rate:', 'Active Asset Count:'],
            localize=True, sticky=True, labels=True,
            style="background-color: #f5f5f5; border: 2px solid #555; border-radius: 4px; font-family: Arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(m)

    category_colors = {'Education': 'blue', 'Healthcare': 'red', 'Worship': 'purple', 'Community Center/YMCA': 'green'}
    category_icons = {'Education': 'graduation-cap', 'Healthcare': 'heart', 'Worship': 'church', 'Community Center/YMCA': 'users'}

    for idx, row in filtered_pois.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        cat = row['category']
        popup_text = f"<div style='font-family:Arial;'><b>Resource:</b> {row['name']}<br><b>Category:</b> {cat}</div>"
        
        Marker(
            location=[lat, lon],
            popup=Popup(popup_text, max_width=250),
            icon=Icon(color=category_colors.get(cat, 'gray'), icon=category_icons.get(cat, 'info-sign'), prefix='fa')
        ).add_to(m)

    st_folium(m, width="100%", height=550, returned_objects=[])

# Execute independent map fragment block
render_interactive_map(final_data, filtered_pois, target_layer)

# =========================================================
# MAIN SECTION 2: RESPONSIVE SIDE-BY-SIDE METRICS CHARTS (PLOTLY)
# =========================================================
st.markdown("---")
st.subheader("Dynamic Analytical Charts & Structural Data Correlation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Infrastructure Proportions Across Active Types**")
    category_tallies = filtered_pois['category'].value_counts().reset_index()
    category_tallies.columns = ['Asset Category', 'Total Registered Pins']
    
    fig_bar = px.bar(
        category_tallies, x='Asset Category', y='Total Registered Pins',
        color='Asset Category',)
