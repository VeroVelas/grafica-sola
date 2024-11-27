import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Cargar los datos
data = pd.read_csv('consumo_mascotas_sin_gasto.csv')
data['Fecha'] = pd.to_datetime(data['Fecha'])
data.set_index('Fecha', inplace=True)

# ---- Gráfica de Pastel: Tipo de Alimento ---- #
# Sumar el consumo total por tipo de alimento
tipo_alimento_totales = data.groupby('Tipo de Alimento')['Cantidad Consumida (gr)'].sum()

# Crear gráfica de pastel
plt.figure(figsize=(8, 8))
plt.pie(tipo_alimento_totales, labels=tipo_alimento_totales.index, autopct='%1.1f%%',
        startangle=90, colors=['#FFD700', '#90EE90', '#FF6347'])
plt.title('Distribución del Consumo por Tipo de Alimento', fontsize=16, fontweight='bold')
plt.legend(tipo_alimento_totales.index, title='Tipos de Alimento', loc='upper right', bbox_to_anchor=(1.3, 0.9))
plt.tight_layout()
plt.savefig('grafica_pastel_tipo_alimento.png')  # Guardar gráfica
plt.show()

# ---- Preparación para Gráfica de Consumo Diario ---- #
# Consolidar duplicados y rellenar días faltantes
data = data.groupby(data.index).sum()  # Consolidar duplicados
data = data.asfreq('D', fill_value=0)  # Asegurar continuidad temporal
daily_consumption = data['Cantidad Consumida (gr)']

# Suavizar la serie para la tendencia
smoothed_consumption = gaussian_filter1d(daily_consumption, sigma=3)

# Calcular aumento del consumo
initial_value = daily_consumption.iloc[0]
final_value = daily_consumption.iloc[-1]
percentage_increase = ((final_value - initial_value) / initial_value) * 100

# Crear gráfica personalizada para consumo diario
plt.figure(figsize=(12, 6))
fig, ax = plt.subplots(figsize=(12, 6))

# Fondo degradado
cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#D7E3FC", "#AFCBFF"])  # Tonos de azul claro
for i in range(100):
    ax.axhspan(i, i+1, color=cmap(i/100), zorder=0)

# Graficar el consumo diario y la tendencia
ax.plot(daily_consumption.index, smoothed_consumption, color='#023E8A', linewidth=3, label='Tendencia Suavizada', zorder=2)  # Azul oscuro
ax.scatter(daily_consumption.index[::30], smoothed_consumption[::30], color='#FFD700', edgecolor='#023E8A', s=100, zorder=3)  # Puntos amarillos

# Personalización de los ejes
ax.set_xticks(daily_consumption.index[::30])  # Mostrar un mes cada 30 días
ax.set_xticklabels([date.strftime('%b') for date in daily_consumption.index[::30]], fontsize=12, color='#023E8A')  # Azul oscuro
ax.set_yticks(np.linspace(0, daily_consumption.max(), 5))  # 5 etiquetas en el eje y
ax.set_yticklabels([f'{int(value):,} gr' for value in np.linspace(0, daily_consumption.max(), 5)],
                   fontsize=12, color='#023E8A')  # Azul oscuro

# Título, leyenda y texto del aumento
ax.set_title('Consumo Diario de Alimentos', fontsize=18, fontweight='bold', color='#023E8A', pad=20)  # Azul oscuro
ax.legend(loc='upper left', fontsize=12, frameon=False, labelcolor='#023E8A')
ax.text(0.02, 0.9, f"Aumento del consumo: {percentage_increase:.2f}%", transform=ax.transAxes,
        fontsize=12, color='#023E8A', fontweight='bold')

# Ajustar límites para una mejor visualización
ax.set_xlim(daily_consumption.index.min(), daily_consumption.index.max())
ax.set_ylim(0, daily_consumption.max() * 1.1)

# Ocultar bordes del gráfico
for spine in ax.spines.values():
    spine.set_visible(False)

# Guardar gráfica
plt.tight_layout()
plt.savefig('grafica_mejorada_consumo_diario.png', facecolor=fig.get_facecolor())
plt.show()

# ---- Modelos de Predicción ---- #
# Modelo SARIMA
sarima_model = SARIMAX(daily_consumption, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.get_forecast(steps=30).predicted_mean

# Modelo Holt-Winters
holt_model = ExponentialSmoothing(daily_consumption, trend='add', seasonal='add', seasonal_periods=7).fit()
holt_forecast = holt_model.forecast(steps=30)

# ---- Gráfica de Predicciones ---- #
plt.figure(figsize=(14, 8))
plt.plot(daily_consumption.index, daily_consumption, label='Datos Reales', color='blue', linewidth=1.5)
plt.plot(sarima_forecast.index, sarima_forecast, label='Predicción SARIMA', color='red', linestyle='--')
plt.plot(holt_forecast.index, holt_forecast, label='Predicción Holt-Winters', color='green', linestyle='--')
plt.title('Predicción del Consumo Diario', fontsize=16)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Cantidad Consumida (gr)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('predicciones_consumo_diario.png')  # Guardar gráfica
plt.show()

#---------------------------------------codigo para back-------------------------------------#
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

app = FastAPI()

# Carpeta para guardar gráficas generadas
GRAPH_PATH = "./generated_graphs"
os.makedirs(GRAPH_PATH, exist_ok=True)

@app.post("/generate-graphs/")
async def generate_graphs(file: UploadFile):
    # Validar el archivo subido
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    
    # Leer el archivo CSV
    try:
        data = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo: {e}")

    # Validar que tenga las columnas esperadas
    required_columns = {"Fecha", "Tipo de Alimento", "Cantidad Consumida (gr)"}
    if not required_columns.issubset(data.columns):
        raise HTTPException(status_code=400, detail=f"El archivo debe contener las columnas: {required_columns}")

    # Procesar datos
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    data.set_index('Fecha', inplace=True)

    # ---- Gráfica de Pastel ---- #
    tipo_alimento_totales = data.groupby('Tipo de Alimento')['Cantidad Consumida (gr)'].sum()
    pie_path = os.path.join(GRAPH_PATH, "grafica_pastel.png")
    plt.figure(figsize=(8, 8))
    plt.pie(tipo_alimento_totales, labels=tipo_alimento_totales.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribución del Consumo por Tipo de Alimento')
    plt.savefig(pie_path)
    plt.close()

    # ---- Gráfica de Consumo Diario ---- #
    data = data.asfreq('D', fill_value=0)
    daily_consumption = data['Cantidad Consumida (gr)']
    smoothed_consumption = gaussian_filter1d(daily_consumption, sigma=3)
    line_path = os.path.join(GRAPH_PATH, "grafica_linea.png")
    plt.figure(figsize=(12, 6))
    plt.plot(daily_consumption.index, smoothed_consumption, label="Tendencia Suavizada", color="blue")
    plt.title("Consumo Diario de Alimentos")
    plt.legend()
    plt.savefig(line_path)
    plt.close()

    # ---- Predicciones con SARIMA y Holt-Winters ---- #
    sarima_model = SARIMAX(daily_consumption, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.get_forecast(steps=30).predicted_mean

    holt_model = ExponentialSmoothing(daily_consumption, trend='add', seasonal='add', seasonal_periods=7).fit()
    holt_forecast = holt_model.forecast(steps=30)

    pred_path = os.path.join(GRAPH_PATH, "grafica_predicciones.png")
    plt.figure(figsize=(14, 8))
    plt.plot(daily_consumption.index, daily_consumption, label="Datos Reales", color="blue")
    plt.plot(sarima_forecast.index, sarima_forecast, label="Predicción SARIMA", color="red", linestyle="--")
    plt.plot(holt_forecast.index, holt_forecast, label="Predicción Holt-Winters", color="green", linestyle="--")
    plt.title("Predicciones del Consumo Diario")
    plt.legend()
    plt.savefig(pred_path)
    plt.close()

    # Devolver rutas de las gráficas
    return {
        "pie_chart": pie_path,
        "line_chart": line_path,
        "prediction_chart": pred_path
    }

@app.get("/download-graph/")
async def download_graph(filename: str):
    filepath = os.path.join(GRAPH_PATH, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Gráfica no encontrada.")
    return FileResponse(filepath)
