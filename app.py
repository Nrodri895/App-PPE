import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Título e introducción
st.title("🛡️ Sistema inteligente de uso de PPE")
st.image("logo.jpg", width=150)
st.markdown("""
Este sistema utiliza visión por computadora para detectar si una persona cumple con el uso correcto de los elementos de protección personal (PPE): casco, chaleco y botas.

---
""")

# Instrucciones
st.subheader("📋 Instrucciones")
st.markdown("""
1. Carga una imagen desde tu dispositivo o toma una usando la cámara.
2. El sistema detectará las personas en la imagen usando un modelo base.
3. Luego analizará si cada persona cumple con los requisitos de PPE (casco, chaleco y botas).
4. Se mostrará un mensaje de cumplimiento o alerta para cada persona detectada.
""")

# Cargar imagen
imagen = None
opcion = st.radio("Selecciona una opción:", ["📁 Cargar imagen", "📸 Usar cámara"])

if opcion == "📁 Cargar imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        imagen = Image.open(archivo)
elif opcion == "📸 Usar cámara":
    imagen = st.camera_input("Toma una foto")

if imagen is not None:
    st.image(imagen, caption="📷 Imagen cargada", use_container_width =True)

    # Convertir imagen a formato compatible con OpenCV
    img_array = np.array(imagen)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Cargar modelos
    modelo_personas = YOLO("yolov8n.pt")
    modelo_ppe = YOLO("best.pt")

    # Detectar personas
    resultados_personas = modelo_personas(img_array)[0]
    personas = [box for box in resultados_personas.boxes if int(box.cls[0]) == 0]  # clase 0 = persona

    if len(personas) == 0:
        st.warning("⚠️ No se detectaron personas en la imagen.")
    else:
        st.success(f"👥 Se detectaron {len(personas)} persona(s). Procesando PPE...")

        for i, persona in enumerate(personas, start=1):
            x1, y1, x2, y2 = map(int, persona.xyxy[0])
            recorte = img_array[y1:y2, x1:x2]

            # Detección de PPE en la región recortada
            resultado_ppe = modelo_ppe(recorte)[0]
            clases_detectadas = [modelo_ppe.names[int(cls)] for cls in resultado_ppe.boxes.cls]

            tiene_casco = "casco" in clases_detectadas
            tiene_chaleco = "chaleco" in clases_detectadas
            tiene_botas = "botas" in clases_detectadas

            st.subheader(f"👤 Persona {i}")
            for box in resultado_ppe.boxes:
                cls = modelo_ppe.names[int(box.cls[0])]
                x1_, y1_, x2_, y2_ = map(int, box.xyxy[0])
                cv2.rectangle(recorte, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
                cv2.putText(recorte, cls, (x1_, y1_ - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.image(recorte, caption="📦 Detección de objetos en la persona", use_container_width =True)

            if tiene_casco and tiene_chaleco and tiene_botas:
                st.success("✅ Cumple con los requisitos para el ingreso a la fábrica 🏭")
            else:
                st.error("🚨 ALERTA: No cumple con los requisitos del PPE, no puede ingresar a la fábrica")

# Pie de página
st.markdown("---")
st.markdown("**Nicolás Rodríguez Mateus, UNAB 2025 - Derechos reservados**")
