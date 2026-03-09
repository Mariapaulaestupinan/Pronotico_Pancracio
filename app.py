import streamlit as st
import pandas as pd
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

import pronostico

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Pronóstico de Demanda",
    page_icon="logo_Pancracio.png",
    layout="wide"
)

# ── Carpeta raíz en Drive ─────────────────────────────────────────────────────
DRIVE_FOLDER_ID = "1nQYLs0eJmuDrXr7jeyGCuZS2wXRUnu3s"
WORK_DIR        = pronostico.LOCAL_DIR   # /tmp/pronostico

# ── Archivos según su ubicación en Drive ─────────────────────────────────────
ARCHIVOS_RAIZ = [
    "historico_ventas.csv",
]

ARCHIVOS_MODELOS = [
    "historico_pareto.csv",
    "mae_por_serie.json",
    "pct_ceros.json",
    "productos_a.csv",
    "umbral_bin.json",
    "stack_reg.pkl",
    "xgb_bajo.pkl",
    "xgb_medio.pkl",
    "clf4_cal.pkl",
    "rf_pico.pkl",
    "clf_bin_inter.pkl",
    "xgb_cantidad_inter.pkl",
    "modelos_serie_inter.pkl",
]

ACTUALIZAR_RAIZ = [
    "historico_ventas.csv",
]
ACTUALIZAR_MODELOS = [
    "historico_pareto.csv",
    "mae_por_serie.json",
    "productos_a.csv",
    "stack_reg.pkl",
    "xgb_bajo.pkl",
    "xgb_medio.pkl",
    "clf4_cal.pkl",
    "rf_pico.pkl",
    "clf_bin_inter.pkl",
    "xgb_cantidad_inter.pkl",
    "modelos_serie_inter.pkl",
]


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE — funciones de conexión
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["google_credentials"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def listar_archivos(service, folder_id):
    """Devuelve dict {nombre: file_id} de los archivos en una carpeta."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


def buscar_subcarpeta(service, nombre, parent_id):
    """Busca una subcarpeta por nombre dentro de parent_id. Devuelve su ID o None."""
    results = service.files().list(
        q=(f"'{parent_id}' in parents and trashed=false "
           f"and mimeType='application/vnd.google-apps.folder' "
           f"and name='{nombre}'"),
        fields="files(id, name)"
    ).execute()
    archivos = results.get("files", [])
    return archivos[0]["id"] if archivos else None


def descargar_archivo(service, file_id, dest_path):
    """Descarga un archivo de Drive a dest_path."""
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def subir_archivo(service, local_path, nombre, archivos_existentes, folder_id):
    """Sube o actualiza un archivo en una carpeta de Drive."""
    media = MediaFileUpload(local_path, resumable=True)
    if nombre in archivos_existentes:
        service.files().update(
            fileId=archivos_existentes[nombre],
            media_body=media
        ).execute()
    else:
        service.files().create(
            body={"name": nombre, "parents": [folder_id]},
            media_body=media
        ).execute()


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA INICIAL — se ejecuta una vez al arrancar la app
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def descargar_modelos():
    """
    Descarga todos los archivos necesarios desde Drive al directorio local.
    - Raíz del Drive   → ARCHIVOS_RAIZ
    - Subcarpeta modelos/ → ARCHIVOS_MODELOS
    """
    service   = get_drive_service()
    faltantes = []

    arch_raiz = listar_archivos(service, DRIVE_FOLDER_ID)
    for nombre in ARCHIVOS_RAIZ:
        dest = os.path.join(WORK_DIR, nombre)
        if nombre in arch_raiz:
            descargar_archivo(service, arch_raiz[nombre], dest)
        else:
            faltantes.append(nombre)

    modelos_id = buscar_subcarpeta(service, "modelos", DRIVE_FOLDER_ID)
    if modelos_id is None:
        faltantes.extend(ARCHIVOS_MODELOS)
        return faltantes

    arch_modelos = listar_archivos(service, modelos_id)
    for nombre in ARCHIVOS_MODELOS:
        dest = os.path.join(WORK_DIR, nombre)
        if nombre in arch_modelos:
            descargar_archivo(service, arch_modelos[nombre], dest)
        else:
            faltantes.append(nombre)

    return faltantes


# ══════════════════════════════════════════════════════════════════════════════
# SUBIDA TRAS ACTUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def subir_actualizados(service):
    """Sube al Drive los archivos que fueron modificados por cargar_nuevos_datos."""
    arch_raiz  = listar_archivos(service, DRIVE_FOLDER_ID)
    modelos_id = buscar_subcarpeta(service, "modelos", DRIVE_FOLDER_ID)
    arch_modelos = listar_archivos(service, modelos_id) if modelos_id else {}

    for nombre in ACTUALIZAR_RAIZ:
        ruta = os.path.join(WORK_DIR, nombre)
        if os.path.exists(ruta):
            subir_archivo(service, ruta, nombre, arch_raiz, DRIVE_FOLDER_ID)

    if modelos_id:
        for nombre in ACTUALIZAR_MODELOS:
            ruta = os.path.join(WORK_DIR, nombre)
            if os.path.exists(ruta):
                subir_archivo(service, ruta, nombre, arch_modelos, modelos_id)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.image("logo_Pancracio.png", width=80)
st.title("Pronóstico de Demanda")
st.markdown("---")

with st.spinner("Cargando modelos desde Google Drive..."):
    faltantes = descargar_modelos()
    if faltantes:
        st.warning(f"Archivos no encontrados en Drive: {', '.join(faltantes)}")
    else:
        st.success("✅ Modelos cargados correctamente.")

st.markdown("---")

st.subheader("📂 Subir archivo de ventas")
archivo = st.file_uploader("Sube el Excel con las ventas nuevas", type=["xlsx", "xls"])

if archivo:
    st.success(f"Archivo cargado: **{archivo.name}**")

    if st.button("▶️ Procesar y generar pronóstico", type="primary"):
        with st.spinner("Procesando... esto puede tardar unos minutos."):
            try:
                tmp_input = os.path.join(WORK_DIR, archivo.name)
                with open(tmp_input, "wb") as f:
                    f.write(archivo.read())

                df_forecast = pronostico.cargar_nuevos_datos(tmp_input, verbose=False)

                st.markdown("---")
                st.subheader("Resumen de confiabilidad")
                vc = df_forecast['Confiabilidad'].value_counts()
                col1, col2, col3 = st.columns(3)
                niveles = {
                    'Confiable':              ('🟢', col1),
                    'Parcialmente confiable': ('🟡', col2),
                    'No confiable':           ('🔴', col3),
                }
                for nivel, (emoji, col) in niveles.items():
                    col.metric(label=f"{emoji} {nivel}", value=int(vc.get(nivel, 0)))

                st.subheader("Pronóstico semanal")

                def color_fila(row):
                    c = row.get('Confiabilidad', '')
                    if c == 'Confiable':              return ['background-color: #C6EFCE'] * len(row)
                    if c == 'Parcialmente confiable': return ['background-color: #FFEB9C'] * len(row)
                    if c == 'No confiable':           return ['background-color: #FFC7CE'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_forecast.style.apply(color_fila, axis=1),
                    use_container_width=True,
                    height=450
                )

                excels = [f for f in os.listdir(WORK_DIR)
                          if f.startswith("pronostico_") and f.endswith(".xlsx")]
                if excels:
                    excel_path = os.path.join(WORK_DIR, sorted(excels)[-1])
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar pronóstico Excel",
                            data=f.read(),
                            file_name=os.path.basename(excel_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                with st.spinner("Guardando cambios en Google Drive..."):
                    service = get_drive_service()
                    subir_actualizados(service)

                st.success("✅ Pronóstico generado y modelos actualizados en Drive.")

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


