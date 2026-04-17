# =============================================================================
# sarimax_predictive_model.py
#
# Sistema de análisis predictivo de series temporales de incidencias/eventos.
# Construye variables de infraestructura (RAM, CPU, máquinas), ajusta un modelo
# SARIMAX con transformación raíz cuadrada, calcula intervalos de confianza
# mediante bootstrap MBB y genera un conjunto completo de gráficos de
# diagnóstico y predicción a largo plazo.
#
# Flujo principal:
#   1. Carga de datos desde la fuente (ver Sección de Datos)
#   2. Integración y limpieza del DataFrame maestro
#   3. Construcción de proxy de infraestructura (si RAM real no disponible)
#   4. Ajuste SARIMAX + bootstrap IC 95%
#   5. Generación de 8 gráficos de diagnóstico (G1–G8)
# =============================================================================

import traceback
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


# =============================================================================
# SECCIÓN DE DATOS — ADAPTAR A TU FUENTE
#
# Este módulo está diseñado para trabajar con cualquier fuente de datos que
# pueda construir un DataFrame maestro mensual con la siguiente estructura
# mínima:
#
#   DataFrame maestro (índice: DatetimeIndex mensual, freq='MS')
#   ┌─────────────────────────────┬────────────────────────────────────────┐
#   │ Columna                     │ Descripción                            │
#   ├─────────────────────────────┼────────────────────────────────────────┤
#   │ INCIDENCIAS  (requerida)    │ Variable objetivo: conteo mensual de   │
#   │                             │ eventos/incidencias/tickets            │
#   │ CLIENTES_ACTIVOS (opcional) │ Nº de clientes/usuarios activos/mes    │
#   │ RAM_TOTAL_MB (opcional)     │ RAM real asignada en MB (NETBOX u otro)│
#   │ CPU_TOTAL_MB (opcional)     │ vCPUs totales asignadas                │
#   │ MAQUINAS_REALES (opcional)  │ Nº de máquinas/instancias activas      │
#   └─────────────────────────────┴────────────────────────────────────────┘
#
# OPCIONES DE CARGA (elige la que aplique a tu caso):
#
# ── Opción A: CSV / Excel ─────────────────────────────────────────────────
#
#   df_maestro = pd.read_csv("datos.csv", parse_dates=["FECHA"])
#   df_maestro = df_maestro.set_index("FECHA").sort_index()
#
# ── Opción B: Base de datos SQL (cualquier motor) ─────────────────────────
#
#   import sqlalchemy
#   engine = sqlalchemy.create_engine("dialect+driver://user:pass@host/db")
#   df_maestro = pd.read_sql("SELECT * FROM mi_tabla", engine, index_col="FECHA",
#                             parse_dates=["FECHA"])
#
#   Para Oracle específicamente:
#     engine = sqlalchemy.create_engine("oracle+oracledb://user:pass@host:port/?service_name=svc")
#
#   Para PostgreSQL:
#     engine = sqlalchemy.create_engine("postgresql+psycopg2://user:pass@host/db")
#
# ── Opción C: API (World Bank, UNESCO, etc.) ──────────────────────────────
#
#   import wbgapi as wb
#   df_raw = wb.data.DataFrame("SE.PRM.ENRR", time=range(2000, 2024))
#   # ... transformar al formato maestro requerido
#
# ── Opción D: Datos sintéticos para pruebas ───────────────────────────────
#
#   fechas = pd.date_range("2016-01-01", periods=96, freq="MS")
#   np.random.seed(42)
#   tendencia = np.linspace(100, 300, 96)
#   estacionalidad = 20 * np.sin(2 * np.pi * np.arange(96) / 12)
#   ruido = np.random.normal(0, 10, 96)
#   df_maestro = pd.DataFrame({
#       "INCIDENCIAS":      (tendencia + estacionalidad + ruido).clip(0),
#       "CLIENTES_ACTIVOS": np.linspace(20, 60, 96) + np.random.normal(0, 2, 96),
#       "RAM_TOTAL_MB":     np.linspace(10000, 50000, 96),
#   }, index=fechas)
#
# NOTAS IMPORTANTES:
#   · El índice DEBE ser un DatetimeIndex mensual (freq='MS' o 'M').
#   · Las columnas opcionales permiten que el modelo SARIMAX sea multivariante.
#     Si no están disponibles, el modelo funciona en modo univariante.
#   · Los valores NaN en columnas opcionales se interpolan automáticamente.
#   · Se excluye automáticamente el mes en curso (datos incompletos).
# =============================================================================

# ── CARGAR AQUÍ TU DATAFRAME MAESTRO ─────────────────────────────────────────
# Ejemplo con datos sintéticos (sustituir por tu fuente real):
fechas = pd.date_range("2016-01-01", periods=96, freq="MS")
np.random.seed(42)
tendencia      = np.linspace(100, 300, 96)
estacionalidad = 20 * np.sin(2 * np.pi * np.arange(96) / 12)
ruido          = np.random.normal(0, 10, 96)
df_maestro_input = pd.DataFrame({
    "INCIDENCIAS":      (tendencia + estacionalidad + ruido).clip(0),
    "CLIENTES_ACTIVOS": np.linspace(20, 60, 96) + np.random.normal(0, 2, 96),
    "RAM_TOTAL_MB":     np.linspace(10_000, 50_000, 96),
}, index=fechas)
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# MÓDULO 0 — INTEGRACIÓN RAM / CPU CON EXPANSIÓN TEMPORAL
#
# Las fuentes de RAM y CPU pueden devolver una fila por proyecto con sus fechas
# de inicio y fin, no una fila por mes. Este módulo "expande" esas filas:
# para cada proyecto genera una entrada por cada mes en que estaba activo y
# luego agrega (suma) todos los proyectos activos en ese mes.
#
# Resultado: un DataFrame mensual con RAM_TOTAL_MB, CPU_TOTAL_MB y
# MAQUINAS_REALES alineado con el índice temporal del DataFrame maestro.
# =============================================================================

def _expandir_proyectos_por_mes(df_raw: pd.DataFrame,
                                 cols_suma: list[str]) -> pd.DataFrame | None:
    required = {'FECHA_INICIAL', 'FECHA_FINAL'} | set(cols_suma)
    if df_raw.empty or not required.issubset(df_raw.columns):
        return None

    df = df_raw.copy()
    df['FECHA_INICIAL'] = pd.to_datetime(df['FECHA_INICIAL'], errors='coerce')
    df['FECHA_FINAL']   = pd.to_datetime(df['FECHA_FINAL'],   errors='coerce')

    antes = len(df)
    df = df.dropna(subset=['FECHA_INICIAL', 'FECHA_FINAL'])
    excluidos = antes - len(df)
    if excluidos:
        print(f"    · Proyectos excluidos (fecha inicio o fin nula): {excluidos}")
    if df.empty:
        return None

    df['MES_INICIO'] = df['FECHA_INICIAL'].dt.to_period('M')
    df['MES_FIN']    = df['FECHA_FINAL'].dt.to_period('M')
    for col in cols_suma:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    filas = []
    for _, row in df.iterrows():
        periodos = pd.period_range(row['MES_INICIO'], row['MES_FIN'], freq='M')
        for p in periodos:
            fila = {'PERIODO': p.strftime('%Y-%m')}
            for col in cols_suma:
                fila[col] = row[col]
            filas.append(fila)

    if not filas:
        return None

    df_exp = pd.DataFrame(filas)
    df_agg = df_exp.groupby('PERIODO')[cols_suma].sum().reset_index()
    return df_agg.sort_values('PERIODO').reset_index(drop=True)


def integrar_ram_cpu(df_maestro: pd.DataFrame,
                     resultados_df: dict) -> pd.DataFrame:
    df = df_maestro.copy()

    raw_ram = resultados_df.get('RAM', pd.DataFrame())
    df_ram_agg = _expandir_proyectos_por_mes(raw_ram, ['MAQUINAS_REALES', 'RAM_TOTAL_MB'])

    if df_ram_agg is not None and not df_ram_agg.empty:
        for col in [c for c in df_ram_agg.columns if c != 'PERIODO']:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(df_ram_agg, on='PERIODO', how='left')
        df['RAM_TOTAL_MB']    = df['RAM_TOTAL_MB'].fillna(0)
        df['MAQUINAS_REALES'] = df['MAQUINAS_REALES'].fillna(0)
        n = (df['RAM_TOTAL_MB'] > 0).sum()
        print(f"  ✓ RAM_TOTAL_MB integrada. Meses: {n} | Max: {df['RAM_TOTAL_MB'].max():,.0f} MB")
    else:
        print("  ⚠ Consulta 'RAM' sin datos válidos.")
        if 'RAM_TOTAL_MB' not in df.columns:
            df['RAM_TOTAL_MB'] = 0.0
        if 'MAQUINAS_REALES' not in df.columns:
            df['MAQUINAS_REALES'] = 0.0

    raw_cpu = resultados_df.get('CPU', pd.DataFrame())
    df_cpu_agg = _expandir_proyectos_por_mes(raw_cpu, ['MAQUINAS_REALES', 'CPU_TOTAL_MB'])

    if df_cpu_agg is not None and not df_cpu_agg.empty:
        df_cpu_solo = df_cpu_agg[['PERIODO', 'CPU_TOTAL_MB']].copy()
        if 'CPU_TOTAL_MB' in df.columns:
            df = df.drop(columns=['CPU_TOTAL_MB'])
        df = df.merge(df_cpu_solo, on='PERIODO', how='left')
        df['CPU_TOTAL_MB'] = df['CPU_TOTAL_MB'].fillna(0)
        n = (df['CPU_TOTAL_MB'] > 0).sum()
        print(f"  ✓ CPU_TOTAL_MB integrada. Meses: {n} | Max: {df['CPU_TOTAL_MB'].max():,.0f} vCPUs")
    else:
        print("  ⚠ Consulta 'CPU' sin datos válidos.")
        if 'CPU_TOTAL_MB' not in df.columns:
            df['CPU_TOTAL_MB'] = 0.0

    return df


# =============================================================================
# MÓDULO 1 — PROXY DE INFRAESTRUCTURA
#
# El modelo SARIMAX necesita una variable exógena que represente el tamaño de
# la infraestructura desplegada. La fuente preferida es RAM_TOTAL_MB (datos
# reales). Si esa columna no tiene suficiente cobertura (< 5% de meses con
# valor > 0), se intenta RAM_MB_ASIGNADA como alternativa.
#
# Si ninguna fuente de RAM tiene datos suficientes, se construye un PROXY
# sintético mediante regresión OLS sobre las variables disponibles
# (CLIENTES_ACTIVOS, CLIENTES, MAQUINAS_ACTIVAS), escaladas a MB de RAM
# estimada (factor base: 512 MB por cliente).
#
# Devuelve el DataFrame enriquecido con PROXY_INFRAESTRUCTURA y un flag
# booleano que indica si se usó proxy (True) o datos reales (False).
# =============================================================================
def construir_proxy_infraestructura(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    df = df.copy()

    if 'RAM_TOTAL_MB' in df.columns and (df['RAM_TOTAL_MB'].fillna(0) > 0).mean() >= 0.05:
        print("  ✓ RAM_TOTAL_MB (real) con datos suficientes. No se usa proxy.")
        df['PROXY_INFRAESTRUCTURA'] = df['RAM_TOTAL_MB'].astype(float)
        return df, False

    ram_col = 'RAM_MB_ASIGNADA'
    if ram_col in df.columns and (df[ram_col].fillna(0) > 0).mean() >= 0.05:
        print("  ✓ RAM_MB_ASIGNADA con datos suficientes. No se usa proxy.")
        df['PROXY_INFRAESTRUCTURA'] = df[ram_col].astype(float)
        return df, False

    print("  ⚠ RAM sin datos suficientes. Construyendo PROXY_INFRAESTRUCTURA…")
    predictores = [
        col for col in ['CLIENTES_ACTIVOS', 'CLIENTES', 'MAQUINAS_ACTIVAS']
        if col in df.columns and (df[col].fillna(0) > 0).sum() >= 12
    ]

    if not predictores:
        print("  ⚠ Sin predictores disponibles. PROXY = 0.")
        df['PROXY_INFRAESTRUCTURA'] = 0.0
        return df, True

    cli_col = predictores[0]
    X       = df[predictores].fillna(0).astype(float).values
    y_ref   = df[cli_col].fillna(0).astype(float).values * 512.0

    try:
        X_bias = np.column_stack([X, np.ones(len(X))])
        beta, _, _, _ = np.linalg.lstsq(X_bias, y_ref, rcond=None)
        proxy = np.clip(X_bias @ beta, 0, None)
    except Exception:
        proxy = np.clip(y_ref, 0, None)

    df['PROXY_INFRAESTRUCTURA'] = proxy
    corr = np.corrcoef(df[cli_col].fillna(0), proxy)[0, 1]
    print(f"  ✓ PROXY creado. Correlación con '{cli_col}': {corr:.3f}")
    return df, True


# =============================================================================
# MÓDULO 2 — BOOTSTRAP MBB VECTORIZADO (Moving Block Bootstrap)
#
# Calcula los intervalos de confianza al 95% para la predicción SARIMAX
# usando un método híbrido: los errores estándar analíticos del modelo
# se combinan con remuestreo de residuos históricos mediante MBB.
#
# ¿Por qué MBB y no bootstrap clásico?
#   Los residuos de series temporales tienen autocorrelación; el bootstrap
#   clásico (muestreo individual) la destruye. El MBB remuestrea bloques
#   contiguos de longitud fija (block_size=6 meses), preservando la
#   estructura de dependencia temporal.
#
# Proceso:
#   1. Estandarizar los residuos históricos del modelo.
#   2. Construir 1000 simulaciones tomando bloques aleatorios de residuos
#      y añadiéndolos sobre la predicción media (transformada √).
#   3. Calcular percentiles 2.5% y 97.5% de las simulaciones → IC inferior/superior.
#   4. Deshacer la transformación √ elevando al cuadrado → escala original.
#
# Nota: el se_mean analítico se limita (cap) a 3σ histórico para evitar que
# intervalos excesivamente anchos en el horizonte largo distorsionen el IC.
# =============================================================================
def _bootstrap_ic_vectorizado(
    modelo_fit,
    pred_obj,
    periodos: int,
    n_iter: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    block_size: int = 6,
    ic_cap_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(random_state)

    se_mean_raw     = pred_obj.se_mean.values.astype(np.float32)
    pred_mean_trans = pred_obj.predicted_mean.values.astype(np.float32)

    residuos_raw = modelo_fit.resid.values
    residuos_raw = residuos_raw[np.isfinite(residuos_raw)]
    sigma_hist   = float(residuos_raw.std())
    se_mean_cap  = np.minimum(se_mean_raw, ic_cap_sigma * sigma_hist).astype(np.float32)

    if se_mean_raw.max() > se_mean_cap.max() * 1.01:
        print(f"  IC cap: se_mean max {se_mean_raw.max():.3f} → "
              f"{se_mean_cap.max():.3f}  (cap={ic_cap_sigma}×σ={ic_cap_sigma*sigma_hist:.3f})")

    residuos_std = ((residuos_raw - residuos_raw.mean())
                    / (residuos_raw.std() + 1e-9)).astype(np.float32)
    n_res = len(residuos_std)

    n_bloques      = int(np.ceil(periodos / block_size))
    starts_validos = np.arange(0, n_res - block_size + 1)
    starts_matrix  = rng.choice(starts_validos, size=(n_iter, n_bloques), replace=True)
    offsets        = np.arange(block_size, dtype=np.int32)
    idx_matrix     = (starts_matrix[:, :, None] + offsets[None, None, :])
    idx_matrix     = idx_matrix.reshape(n_iter, -1)[:, :periodos]
    ruido_matrix   = residuos_std[idx_matrix]

    simulaciones = pred_mean_trans[None, :] + ruido_matrix * se_mean_cap[None, :]

    lower_trans = np.minimum(
        np.percentile(simulaciones, (alpha / 2) * 100, axis=0),
        pred_mean_trans,
    )
    upper_trans = np.maximum(
        np.percentile(simulaciones, (1 - alpha / 2) * 100, axis=0),
        pred_mean_trans,
    )

    ic_lower = np.power(lower_trans.astype(np.float64), 2).clip(min=0)
    ic_upper = np.power(upper_trans.astype(np.float64), 2)

    print(f"  IC a 12m: [{ic_lower[11]:.0f}, {ic_upper[11]:.0f}]")
    print(f"  IC a {periodos}m: [{ic_lower[-1]:.0f}, {ic_upper[-1]:.0f}]")
    return ic_lower, ic_upper


# =============================================================================
# MÓDULO 3 — PREDICCIÓN SARIMAX MULTIDIMENSIONAL
#
# Núcleo del análisis predictivo. Ajusta un modelo SARIMAX sobre la serie
# mensual de incidencias e incorpora variables exógenas (RAM, CPU, clientes,
# máquinas) para mejorar la precisión.
#
# Pasos internos:
#   1. Llama al Módulo 1 para obtener/construir la variable de infraestructura.
#   2. Filtra las variables exógenas candidatas: solo se admiten las que tienen
#      al menos 12 meses con valor > 0.
#   3. Selecciona las 3 con mayor correlación absoluta con INCIDENCIAS.
#   4. Aplica transformación √ a la serie objetivo para estabilizar la varianza
#      (series de conteo crecientes tienen heterocedasticidad).
#   5. Ajusta SARIMAX(1,1,1)(1,1,0)[12] con L-BFGS, hasta 200 iteraciones.
#   6. Proyecta las variables exógenas futuras mediante tendencia lineal sobre
#      los últimos 24 meses de histórico.
#   7. Obtiene la predicción puntual y delega el IC al Módulo 2 (MBB).
#   8. Deshace la transformación √ en la predicción final (eleva al cuadrado).
#
# Parámetros fijos del modelo:
#   order=(1,1,1)  seasonal_order=(1,1,0,12)
#   — 1 diferenciación regular + 1 estacional para eliminar tendencia y
#     estacionalidad multiplicativa anual.
# =============================================================================
def predecir_sarimax_multidimensional(
    df: pd.DataFrame,
    periodos: int = 58,
    n_bootstrap: int = 1000,
) -> tuple[pd.DataFrame | None, object, pd.DataFrame, pd.DataFrame, bool]:

    try:
        df, proxy_activo = construir_proxy_infraestructura(df)
        cols_candidatas = [
            'RAM_TOTAL_MB', 'CPU_TOTAL_MB', 'MAQUINAS_REALES',
            'CLIENTES_ACTIVOS', 'CLIENTES', 'MAQUINAS_ACTIVAS',
        ]
        if proxy_activo:
            cols_candidatas.append('PROXY_INFRAESTRUCTURA')

        exog_validas = []
        for col in cols_candidatas:
            if col not in df.columns:
                continue
            n_nonzero = (df[col].fillna(0) > 0).sum()
            if n_nonzero >= 12:
                exog_validas.append(col)
                print(f"  ✓ Exógena '{col}' activa ({n_nonzero} meses)")
            else:
                print(f"  ⚠ Exógena '{col}' descartada ({n_nonzero} meses)")

        if not exog_validas:
            print("  ⚠ Sin exógenas válidas. Modelo univariante.")
            exog_hist = None
        else:
            y_inc = df['INCIDENCIAS'].astype(float)
            corrs = {}
            for col in exog_validas:
                try:
                    corrs[col] = abs(y_inc.corr(df[col].astype(float)))
                except Exception:
                    corrs[col] = 0.0
            exog_validas = sorted(corrs, key=corrs.get, reverse=True)[:3]
            print(f"  ✓ Exógenas seleccionadas por correlación (top 3):")
            for col in exog_validas:
                print(f"    · {col}: r={corrs[col]:.3f}")

            exog_hist = (
                df[exog_validas].copy().astype(float)
                .replace(0, np.nan)
                .interpolate(method='linear')
                .bfill()
                .fillna(0)
            )

        y_trans = np.sqrt(df['INCIDENCIAS'].astype(float).clip(lower=0))

        order          = (1, 1, 1)
        seasonal_order = (1, 1, 0, 12)
        print(f"  [SARIMAX] Parámetros: order={order}  seasonal={seasonal_order}")

        modelo = SARIMAX(
            y_trans, exog=exog_hist,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        modelo_fit = modelo.fit(disp=False, method='lbfgs', maxiter=200)
        print(f"  ✓ SARIMAX ajustado. AIC={modelo_fit.aic:.2f}")

        fechas_pred = pd.date_range(
            start=df.index[-1] + pd.offsets.MonthBegin(1),
            periods=periodos, freq='MS',
        )

        if exog_validas:
            exog_f = {}
            tail_exog = min(24, len(exog_hist))
            x_tail   = np.arange(tail_exog)
            x_future = np.arange(tail_exog, tail_exog + periodos)
            for col in exog_validas:
                serie = exog_hist[col]
                valores_tail = serie.tail(tail_exog).values.astype(float)
                try:
                    m, b = np.polyfit(x_tail, valores_tail, 1)
                    proyeccion = np.clip(np.polyval([m, b], x_future), 0, None)
                    metodo = f'tendencia lineal ({m:+.2f}/mes)'
                except Exception:
                    proyeccion = np.full(periodos, valores_tail[-1])
                    metodo = 'estático'
                exog_f[col] = proyeccion
                print(f"  → Exógena futura '{col}': {metodo}")
            exog_forecast = pd.DataFrame(exog_f, index=fechas_pred)
        else:
            exog_forecast = None

        pred_obj        = modelo_fit.get_forecast(steps=periodos, exog=exog_forecast)
        pred_mean_trans = pred_obj.predicted_mean.values
        pred_mean       = np.power(pred_mean_trans, 2)

        print(f"  IC híbrido analítico-MBB ({n_bootstrap} iteraciones, float32)…")
        ic_lower, ic_upper = _bootstrap_ic_vectorizado(
            modelo_fit=modelo_fit,
            pred_obj=pred_obj,
            periodos=periodos,
            n_iter=n_bootstrap,
            alpha=0.05,
            random_state=42,
            block_size=6,
        )
        print("  ✓ IC híbrido completado.")

        df_pred = pd.DataFrame({
            'FECHA':            fechas_pred,
            'PRED_INCIDENCIAS': pred_mean,
            'INC_IC_INF':       ic_lower,
            'INC_IC_SUP':       ic_upper,
        })
        df_pred['PRED_PRESUPUESTO'] = 0.0

        x_axis   = np.arange(len(y_trans))
        tail_len = min(24, len(y_trans))

        if not proxy_activo and exog_validas and 'RAM_TOTAL_MB' in exog_validas:
            df_pred['PRED_RAM'] = exog_forecast['RAM_TOTAL_MB'].values
            df_pred['PRED_MAQUINAS'] = (
                exog_forecast['MAQUINAS_REALES'].values
                if 'MAQUINAS_REALES' in exog_validas
                else (exog_forecast['MAQUINAS_ACTIVAS'].values
                      if 'MAQUINAS_ACTIVAS' in exog_validas
                      else np.zeros(periodos))
            )
        elif proxy_activo and exog_validas and 'PROXY_INFRAESTRUCTURA' in exog_validas:
            df_pred['PRED_RAM'] = exog_forecast['PROXY_INFRAESTRUCTURA'].values
            df_pred['PRED_MAQUINAS'] = (
                exog_forecast['MAQUINAS_ACTIVAS'].values
                if 'MAQUINAS_ACTIVAS' in exog_validas
                else np.zeros(periodos)
            )
        elif exog_validas and 'MAQUINAS_ACTIVAS' in exog_validas:
            df_pred['PRED_MAQUINAS'] = exog_forecast['MAQUINAS_ACTIVAS'].values
            df_pred['PRED_RAM']      = np.zeros(periodos)
        elif 'MAQUINAS_ACTIVAS' in df.columns:
            c = np.polyfit(
                x_axis[-tail_len:],
                df['MAQUINAS_ACTIVAS'].tail(tail_len).values.astype(float), 1,
            )
            df_pred['PRED_MAQUINAS'] = np.polyval(
                c, np.arange(len(y_trans), len(y_trans) + periodos)
            ).clip(min=0)
            df_pred['PRED_RAM'] = np.zeros(periodos)
        else:
            df_pred['PRED_MAQUINAS'] = np.zeros(periodos)
            df_pred['PRED_RAM']      = np.zeros(periodos)

        return df_pred, modelo_fit, exog_hist if exog_hist is not None else pd.DataFrame(), df, proxy_activo

    except Exception as e:
        print(f"!!! Error Crítico en predecir_sarimax_multidimensional: {e}")
        traceback.print_exc()
        return None, None, None, df, False


# =============================================================================
# MÓDULO 5 — GRÁFICOS
#
#   G1 — generar_grafico_individual()
#        Serie histórica + ajuste del modelo + predicción + IC 95%.
#
#   G2 — generar_grafico_estudio_pendientes()
#        Incidencias históricas | RAM histórica (GB). Pendientes de crecimiento.
#
#   G3 — generar_grafico_descomposicion()
#        Descomposición STL: tendencia, estacionalidad y residuo.
#
#   G4 — generar_grafico_correlacion()
#        Correlación entre incidencias y clientes activos + ratio tickets/cliente.
#
#   G5 — generar_grafico_unificado()
#        Normalización Min-Max (0→1) de incidencias y RAM en un mismo eje.
#
#   G6 — generar_grafico_inventario_infra_individual()
#        Tres paneles: Máquinas | CPU (vCPUs) | RAM (GB).
#
#   G7 — generar_grafico_inventario_normalizado()
#        Tres métricas de infraestructura normalizadas en un solo eje.
#
#   G8 — generar_grafico_diagnostico()
#        Panel 2×4: residuos, histograma, Q-Q, fitted vs real, ACF, PACF
#        y tarjeta de métricas (R², MAE, RMSE, MAPE, AIC, Ljung-Box, JB, ADF).
# =============================================================================


def generar_grafico_individual(
    df_historico: pd.DataFrame,
    df_prediccion: pd.DataFrame,
    df_ajuste: pd.DataFrame,
) -> None:
    try:
        fig, ax = plt.subplots(figsize=(15, 7))
        fig.patch.set_facecolor('#f0f2f5')
        ax.set_facecolor('#f0f2f5')

        ax.plot(df_historico.index, df_historico['INCIDENCIAS'],
                label='Histórico Real', color='#2c3e50',
                linewidth=2, marker='o', markersize=4, zorder=3)
        ax.plot(df_ajuste['FECHA_DT'], df_ajuste['AJUSTE'],
                label='Ajuste del Modelo', color='#f39c12',
                alpha=0.85, linewidth=1.5, zorder=2)
        ax.plot(df_prediccion['FECHA'], df_prediccion['PRED_INCIDENCIAS'],
                label='Predicción Futura', color='#e74c3c',
                linestyle='--', linewidth=2.2, zorder=3)
        ax.fill_between(
            df_prediccion['FECHA'],
            df_prediccion['INC_IC_INF'], df_prediccion['INC_IC_SUP'],
            color='#e74c3c', alpha=0.15, label='IC 95%',
        )

        corte = df_historico.index[-1]
        ax.axvline(corte, color='#7f8c8d', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(corte, ax.get_ylim()[1] * 0.97, '  Inicio predicción',
                color='#7f8c8d', fontsize=9, va='top')
        ultimo_val = df_historico['INCIDENCIAS'].iloc[-1]
        ax.axhline(ultimo_val, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                   label=f'Nivel actual ({ultimo_val:.0f})')
        ax.set_title(
            'Análisis Predictivo SARIMAX\n'
            'IC 95% Híbrido Analítico · √-transform · MBB residuos',
            fontsize=14, fontweight='bold', pad=12,
        )
        ax.set_xlim(df_historico.index[0], df_prediccion['FECHA'].max())
        ax.set_ylabel('Incidencias (unidades)', fontsize=11)
        ax.legend(loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()
        fig.savefig("G1_Prediccion_SARIMAX.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G1] Gráfico guardado → G1_Prediccion_SARIMAX.png")
    except Exception as e:
        print(f"  ERROR [G1]: {e}")
        traceback.print_exc()


def generar_grafico_estudio_pendientes(
    df_maestro: pd.DataFrame,
    df_prediccion: pd.DataFrame,
    proxy_activo: bool = False,
) -> None:
    try:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.patch.set_facecolor('#f0f2f5')
        for ax in axes:
            ax.set_facecolor('#f0f2f5')
            ax.grid(True, alpha=0.25, linestyle='--')

        ax1, ax3 = axes

        ax1.plot(df_maestro.index, df_maestro['INCIDENCIAS'],
                 color='#2c3e50', linewidth=1.8, label='Real')
        ax1.set_ylabel('Tickets / mes', fontsize=11)
        ax1.set_title('Incidencias', fontsize=12, loc='left', fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.85, fontsize=9)

        ram_hist_col = None
        for cand in ('RAM_TOTAL_MB', 'RAM_MB_ASIGNADA', 'PROXY_INFRAESTRUCTURA'):
            if cand in df_maestro.columns and \
                    (df_maestro[cand].fillna(0).astype(float) > 0).sum() >= 6:
                ram_hist_col = cand
                break

        if ram_hist_col is not None:
            hist_gb  = df_maestro[ram_hist_col].astype(float) / 1024
            hist_lbl = ('RAM Real (GB)' if ram_hist_col == 'RAM_TOTAL_MB'
                        else 'RAM MB Asig. (GB)' if ram_hist_col == 'RAM_MB_ASIGNADA'
                        else 'Proxy Infraest. (GB)')
            ax3.fill_between(df_maestro.index, hist_gb, alpha=0.15, color='#8e44ad')
            ax3.plot(df_maestro.index, hist_gb, color='#8e44ad',
                     linewidth=1.8, label=hist_lbl)
            m_hist, _ = np.polyfit(np.arange(len(hist_gb)), hist_gb.values, 1)
            ax3.set_title(
                f'Infraestructura · RAM (GB)  tendencia: {m_hist:+.3f} GB/mes',
                fontsize=12, loc='left', fontweight='bold',
            )
        else:
            ax3.set_title('Infraestructura  ·  Sin datos disponibles',
                          fontsize=12, loc='left', fontweight='bold')

        ax3.set_ylabel('RAM (GB)', fontsize=11)
        ax3.set_xlabel('Tiempo', fontsize=11)
        ax3.legend(loc='upper left', framealpha=0.85, fontsize=9)

        for ax in axes:
            ax.set_xlim(df_maestro.index[0], df_maestro.index[-1])

        plt.suptitle('Estudio de Pendientes Históricas de Crecimiento',
                     fontsize=15, fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig("G2_Pendientes.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G2] Gráfico guardado → G2_Pendientes.png")
    except Exception as e:
        print(f"  ERROR [G2]: {e}")
        traceback.print_exc()


def generar_grafico_descomposicion(df_historico: pd.DataFrame) -> None:
    try:
        serie = df_historico['INCIDENCIAS'].astype(float)
        if len(serie) < 24:
            print(f"  ⚠ STL requiere ≥24 meses, hay {len(serie)}. Saltando.")
            return
        res = STL(serie, period=12, seasonal=13, robust=True).fit()
        fig = res.plot()
        fig.set_size_inches(15, 10)
        fig.patch.set_facecolor('#f0f2f5')
        plt.suptitle(
            "Descomposición STL — Tendencia · Estacionalidad · Residuo",
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig("G3_Descomposicion_STL.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print("  ✓ [G3] Gráfico STL guardado → G3_Descomposicion_STL.png")
    except Exception as e:
        print(f"  ⚠ Error en descomposición STL: {e}")


def generar_grafico_correlacion(df_maestro: pd.DataFrame) -> None:
    try:
        col_inc = next((c for c in df_maestro.columns if 'INCIDENCIA' in c.upper()), None)
        col_cli = next(
            (c for c in df_maestro.columns
             if 'ACTIVO' in c.upper() or 'CLIENTE' in c.upper()), None
        )
        if col_inc is None or col_cli is None:
            print(f"  ⚠ [G4] Columnas necesarias no encontradas.")
            return

        inc   = df_maestro[col_inc].astype(float)
        cli   = df_maestro[col_cli].astype(float).replace(0, np.nan)
        ratio = (inc / cli).fillna(0)
        corr  = inc.corr(df_maestro[col_cli].astype(float))

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True,
            gridspec_kw={'height_ratios': [3, 1.5]},
        )
        fig.patch.set_facecolor('#f0f2f5')
        for ax in (ax_top, ax_bot):
            ax.set_facecolor('#f0f2f5')

        color_inc = '#2c3e50'
        color_cli = '#e67e22'

        ln1 = ax_top.plot(inc.index, inc.values, color=color_inc, linewidth=2,
                          marker='o', markersize=3, label='Incidencias')
        ax_top.set_ylabel('Incidencias (tickets/mes)', color=color_inc, fontsize=11)
        ax_top.tick_params(axis='y', labelcolor=color_inc)

        ax_top_r = ax_top.twinx()
        ln2 = ax_top_r.plot(df_maestro.index, df_maestro[col_cli].astype(float),
                            color=color_cli, linewidth=2, linestyle='--',
                            label='Clientes activos')
        ax_top_r.set_ylabel('Clientes activos', color=color_cli, fontsize=11)
        ax_top_r.tick_params(axis='y', labelcolor=color_cli)

        mm6 = inc.rolling(6, min_periods=1).mean()
        ln3 = ax_top.plot(inc.index, mm6.values, color='#e74c3c', linewidth=1.5,
                          alpha=0.7, linestyle=':', label='Media móvil (incidencias)')

        lines  = ln1 + ln2 + ln3
        labels = [l.get_label() for l in lines]
        ax_top.legend(lines, labels, loc='upper left', framealpha=0.85, fontsize=9)
        ax_top.grid(True, alpha=0.25, linestyle='--')
        ax_top.set_title(
            f'Correlación Incidencias vs Clientes  (r = {corr:.3f})',
            fontsize=13, fontweight='bold',
        )

        ax_bot.bar(ratio.index, ratio.values, width=20,
                   color='#3498db', alpha=0.65, label='Tickets por cliente')
        ratio_mm = ratio.rolling(6, min_periods=1).mean()
        ax_bot.plot(ratio.index, ratio_mm.values, color='#1a5276',
                    linewidth=1.8, label='Media móvil')
        ax_bot.set_ylabel('Tickets / cliente', fontsize=10)
        ax_bot.set_xlabel('Tiempo', fontsize=11)
        ax_bot.legend(fontsize=9, framealpha=0.85)
        ax_bot.grid(True, alpha=0.25, linestyle='--')
        ax_bot.set_title('Ratio de Carga Operativa por Cliente', fontsize=11, loc='left')

        plt.suptitle('Análisis de Correlación Incidencias vs Clientes',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig("G4_Correlacion.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G4] Gráfico guardado → G4_Correlacion.png")
    except Exception as e:
        print(f"  ERROR [G4]: {e}")
        traceback.print_exc()


def generar_grafico_unificado(
    df_maestro: pd.DataFrame,
    df_prediccion: pd.DataFrame,
    proxy_activo: bool = False,
) -> None:
    try:
        inc_hist = df_maestro['INCIDENCIAS'].astype(float)

        ram_col = None
        for cand in ('RAM_TOTAL_MB', 'RAM_MB_ASIGNADA', 'PROXY_INFRAESTRUCTURA'):
            if cand in df_maestro.columns and (df_maestro[cand].fillna(0) > 0).sum() >= 6:
                ram_col = cand
                break

        ram_hist = (
            df_maestro[ram_col].astype(float) / 1024
            if ram_col
            else pd.Series(np.nan, index=df_maestro.index)
        )

        def norm_hist(serie: pd.Series) -> pd.Series:
            s = serie.dropna()
            if s.empty:
                return serie
            mn, mx = s.min(), s.max()
            if mx == mn:
                return serie * 0.0
            return (serie - mn) / (mx - mn)

        inc_n = norm_hist(inc_hist)
        ram_n = norm_hist(ram_hist) if ram_col else None

        c_inc = '#2c3e50'
        c_ram = '#8e44ad'

        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_facecolor('#f0f2f5')
        ax.set_facecolor('#f0f2f5')
        ax.grid(True, alpha=0.25, linestyle='--')

        ax.plot(df_maestro.index, inc_n, color=c_inc, lw=2.2, label='Incidencias')
        ax.fill_between(df_maestro.index, inc_n, alpha=0.08, color=c_inc)

        if ram_col and ram_n is not None:
            lbl_ram = {
                'RAM_TOTAL_MB':          'RAM Real (GB)',
                'RAM_MB_ASIGNADA':       'RAM MB Asig. (GB)',
                'PROXY_INFRAESTRUCTURA': 'Proxy Infraest. (GB)',
            }.get(ram_col, ram_col)
            ax.plot(df_maestro.index, ram_n,
                    color=c_ram, lw=2.2, label=f'Infraestructura / {lbl_ram}')
            ax.fill_between(df_maestro.index, ram_n, alpha=0.08, color=c_ram)

        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(df_maestro.index[0], df_maestro.index[-1])
        ax.set_ylabel('Valor normalizado (0 – 1)', fontsize=11)
        ax.set_xlabel('Tiempo', fontsize=11)
        ax.set_title(
            'Alineación Histórica: Incidencias · RAM/Infraestructura\n'
            'Normalización Min-Max (0 → 1)',
            fontsize=13, fontweight='bold',
        )
        ax.legend(loc='upper left', framealpha=0.85, fontsize=10)
        fig.tight_layout()
        fig.savefig("G5_Unificado_Normalizado.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G5] Gráfico guardado → G5_Unificado_Normalizado.png")
    except Exception as e:
        print(f"  ERROR [G5]: {e}")
        traceback.print_exc()


def generar_grafico_inventario_infra_individual(df_maestro: pd.DataFrame) -> None:
    try:
        c_maq = '#e67e22'
        c_cpu = '#2980b9'
        c_ram = '#8e44ad'

        col_maq = next(
            (c for c in ('MAQUINAS_REALES', 'MAQUINAS_ACTIVAS')
             if c in df_maestro.columns and (df_maestro[c].fillna(0) > 0).any()),
            None,
        )
        col_cpu = ('CPU_TOTAL_MB'
                   if 'CPU_TOTAL_MB' in df_maestro.columns
                   and (df_maestro['CPU_TOTAL_MB'].fillna(0) > 0).any()
                   else None)
        col_ram = next(
            (c for c in ('RAM_TOTAL_MB', 'RAM_MB_ASIGNADA')
             if c in df_maestro.columns and (df_maestro[c].fillna(0) > 0).any()),
            None,
        )

        fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
        fig.patch.set_facecolor('#f0f2f5')
        for ax in axes:
            ax.set_facecolor('#f0f2f5')
            ax.grid(True, alpha=0.25, linestyle='--')

        ax1, ax2, ax3 = axes
        idx = df_maestro.index

        def _panel(ax, col, color, marker, ylabel, titulo):
            if col and col in df_maestro.columns and (df_maestro[col].fillna(0) > 0).any():
                serie = df_maestro[col].astype(float)
                ax.fill_between(idx, serie, alpha=0.15, color=color)
                ax.plot(idx, serie, color=color, lw=2.0,
                        marker=marker, markersize=3, label=col)
                mm6 = serie.rolling(6, min_periods=1).mean()
                ax.plot(idx, mm6, color=color, lw=1.2, ls=':', alpha=0.75, label='Media móvil 6m')
                ax.set_ylabel(ylabel, fontsize=10, color=color)
                ax.tick_params(axis='y', labelcolor=color)
                ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
                ax.set_title(titulo, fontsize=12, loc='left', fontweight='bold')
            else:
                nombre = col if col else '(columna no encontrada)'
                ax.text(0.5, 0.5, f'Sin datos de {nombre}',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, color='#95a5a6',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
                ax.set_title(f'{titulo}  ·  Sin datos', fontsize=12, loc='left', fontweight='bold')

        _panel(ax1, col_maq, c_maq, 'o', 'Nº de máquinas',  'Evolución de Máquinas')
        _panel(ax2, col_cpu, c_cpu, 's', 'vCPUs totales',   'Evolución de CPU Total (vCPUs)')

        if col_ram and col_ram in df_maestro.columns and \
                (df_maestro[col_ram].fillna(0) > 0).any():
            serie_gb = df_maestro[col_ram].astype(float) / 1024
            ax3.fill_between(idx, serie_gb, alpha=0.15, color=c_ram)
            ax3.plot(idx, serie_gb, color=c_ram, lw=2.0,
                     marker='^', markersize=3, label=f'{col_ram} (GB)')
            mm6_ram = serie_gb.rolling(6, min_periods=1).mean()
            ax3.plot(idx, mm6_ram, color=c_ram, lw=1.2, ls=':',
                     alpha=0.75, label='Media móvil 6m')
            ax3.set_ylabel('RAM (GB)', fontsize=10, color=c_ram)
            ax3.tick_params(axis='y', labelcolor=c_ram)
            ax3.legend(loc='upper left', fontsize=9, framealpha=0.85)
            ax3.set_title('Evolución de RAM Total (GB)', fontsize=12, loc='left', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Sin datos de RAM',
                     transform=ax3.transAxes, ha='center', va='center',
                     fontsize=12, color='#95a5a6',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            ax3.set_title('Evolución de RAM  ·  Sin datos', fontsize=12, loc='left', fontweight='bold')

        ax3.set_xlabel('Tiempo', fontsize=11)
        for ax in axes:
            ax.set_xlim(df_maestro.index[0], df_maestro.index[-1])
        plt.suptitle('Análisis de Inventario de Infraestructura',
                     fontsize=15, fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig("G6_Inventario_Individual.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G6] Gráfico guardado → G6_Inventario_Individual.png")
    except Exception as e:
        print(f"  ERROR [G6]: {e}")
        traceback.print_exc()


def generar_grafico_inventario_normalizado(df_maestro: pd.DataFrame) -> None:
    try:
        c_maq = '#e67e22'
        c_cpu = '#2980b9'
        c_ram = '#8e44ad'

        col_maq = next(
            (c for c in ('MAQUINAS_REALES', 'MAQUINAS_ACTIVAS')
             if c in df_maestro.columns and (df_maestro[c].fillna(0) > 0).any()),
            None,
        )
        col_cpu = ('CPU_TOTAL_MB'
                   if 'CPU_TOTAL_MB' in df_maestro.columns
                   and (df_maestro['CPU_TOTAL_MB'].fillna(0) > 0).any()
                   else None)
        col_ram = next(
            (c for c in ('RAM_TOTAL_MB', 'RAM_MB_ASIGNADA')
             if c in df_maestro.columns and (df_maestro[c].fillna(0) > 0).any()),
            None,
        )

        disponibles = [c for c in (col_maq, col_cpu, col_ram) if c is not None]
        if not disponibles:
            print("  ⚠ [G7] Ninguna columna de infraestructura con datos. Saltando.")
            return

        idx = df_maestro.index

        def _norm(serie: pd.Series) -> pd.Series:
            s = serie.dropna()
            mn, mx = s.min(), s.max()
            if mx == mn:
                return serie * 0.0
            return (serie - mn) / (mx - mn)

        fig, ax = plt.subplots(figsize=(15, 7))
        fig.patch.set_facecolor('#f0f2f5')
        ax.set_facecolor('#f0f2f5')
        ax.grid(True, alpha=0.25, linestyle='--')

        if col_maq:
            s  = df_maestro[col_maq].astype(float)
            sn = _norm(s)
            ax.plot(idx, sn, color=c_maq, lw=2.0, label=f'Máquinas ({col_maq})')
            ax.fill_between(idx, sn, alpha=0.08, color=c_maq)

        if col_cpu:
            s  = df_maestro[col_cpu].astype(float)
            sn = _norm(s)
            ax.plot(idx, sn, color=c_cpu, lw=2.0, label='CPU Total (vCPUs)')
            ax.fill_between(idx, sn, alpha=0.08, color=c_cpu)

        if col_ram:
            s_raw = df_maestro[col_ram].astype(float)
            sn    = _norm(s_raw)
            ax.plot(idx, sn, color=c_ram, lw=2.2, label=f'RAM Total (GB)')
            ax.fill_between(idx, sn, alpha=0.10, color=c_ram)
            ax_r = ax.twinx()
            ax_r.plot(idx, s_raw / 1024, color=c_ram, lw=0, alpha=0)
            ax_r.set_ylabel('RAM (GB)', fontsize=10, color=c_ram)
            ax_r.tick_params(axis='y', labelcolor=c_ram)
            mn_r, mx_r = s_raw.min() / 1024, s_raw.max() / 1024
            ax_r.set_ylim(mn_r - (mx_r - mn_r) * 0.05, mx_r + (mx_r - mn_r) * 0.15)

        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(df_maestro.index[0], df_maestro.index[-1])
        ax.set_ylabel('Valor normalizado (0 – 1)', fontsize=11)
        ax.set_xlabel('Tiempo', fontsize=11)
        ax.set_title(
            'Inventario de Infraestructura — Máquinas · CPU · RAM\n'
            'Normalización Min-Max (0 → 1)',
            fontsize=13, fontweight='bold',
        )
        ax.legend(loc='upper left', framealpha=0.85, fontsize=10)
        fig.tight_layout()
        fig.savefig("G7_Inventario_Normalizado.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G7] Gráfico guardado → G7_Inventario_Normalizado.png")
    except Exception as e:
        print(f"  ERROR [G7]: {e}")
        traceback.print_exc()


def generar_grafico_diagnostico(df_maestro: pd.DataFrame, modelo_fit) -> None:
    try:
        y_true   = df_maestro['INCIDENCIAS'].astype(float)
        y_pred   = np.power(modelo_fit.fittedvalues, 2).clip(lower=0)
        residuos = y_true - y_pred

        r2   = 1 - (np.sum(residuos**2) / np.sum((y_true - np.mean(y_true))**2))
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        aic  = modelo_fit.aic

        mask = y_true > 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        lags_max = min(10, len(modelo_fit.resid) - 1)
        if lags_max > 0:
            lb_test = acorr_ljungbox(modelo_fit.resid, lags=[lags_max], return_df=True)
            p_lb    = lb_test['lb_pvalue'].values[0]
        else:
            p_lb = float('nan')

        resid_clean = modelo_fit.resid.dropna().values
        jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid_clean)

        serie_adf = df_maestro['INCIDENCIAS'].astype(float).dropna()
        adf_result = adfuller(serie_adf, autolag='AIC')
        adf_stat, adf_p = adf_result[0], adf_result[1]

        fig = plt.figure(figsize=(26, 11))
        fig.patch.set_facecolor('#f0f2f5')
        gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.36,
                              width_ratios=[1, 1, 1, 1.15])

        axes_plot = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 2]),
        ]
        ax_metrics = fig.add_subplot(gs[:, 3])

        for ax in axes_plot + [ax_metrics]:
            ax.set_facecolor('#f0f2f5')

        resid_raw = modelo_fit.resid.dropna()
        idx_plot  = df_maestro.index[-len(resid_raw):]

        ax = axes_plot[0]
        ax.plot(idx_plot, resid_raw.values, color='#2980b9', lw=1.2, alpha=0.8)
        ax.axhline(0, color='#e74c3c', ls='--', lw=1.0)
        ax.fill_between(idx_plot, resid_raw.values, 0,
                        where=resid_raw.values > 0, alpha=0.15, color='#27ae60')
        ax.fill_between(idx_plot, resid_raw.values, 0,
                        where=resid_raw.values < 0, alpha=0.15, color='#e74c3c')
        ax.set_title('Residuos vs Tiempo', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuo (√-espacio)')
        ax.grid(True, alpha=0.25, linestyle='--')

        ax = axes_plot[1]
        ax.hist(resid_raw.values, bins=25, color='#2980b9', edgecolor='white',
                alpha=0.8, density=True)
        xr = np.linspace(resid_raw.min(), resid_raw.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, resid_raw.mean(), resid_raw.std()),
                color='#e74c3c', lw=2.0, label='Normal teórica')
        ax.set_title('Distribución de Residuos', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')

        ax = axes_plot[2]
        (osm, osr), (slope, intercept, _) = stats.probplot(resid_raw.values, dist='norm')
        ax.scatter(osm, osr, color='#2980b9', s=15, alpha=0.7, zorder=3)
        ax.plot(osm, np.array(osm) * slope + intercept,
                color='#e74c3c', lw=2.0, label='Línea normal')
        ax.set_title('Q-Q Plot (Normalidad Residuos)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cuantiles teóricos')
        ax.set_ylabel('Cuantiles observados')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')

        ax = axes_plot[3]
        y_true_sqrt = np.sqrt(y_true.clip(lower=0))
        y_fit_sqrt  = modelo_fit.fittedvalues.values
        ax.scatter(y_true_sqrt, y_fit_sqrt, color='#2c3e50', s=18, alpha=0.6, zorder=3)
        lim_max = max(y_true_sqrt.max(), y_fit_sqrt.max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], color='#e74c3c', lw=1.5,
                ls='--', label='Línea ideal (y=x)')
        ax.set_title('Ajuste: Real vs Predicho (√)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Real (√incidencias)')
        ax.set_ylabel('Ajustado (√incidencias)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')

        plot_acf(resid_raw, ax=axes_plot[4], lags=24, alpha=0.05, color='#2980b9',
                 vlines_kwargs={'colors': '#2980b9'})
        axes_plot[4].set_title('ACF — Autocorrelación Residuos', fontsize=11, fontweight='bold')
        axes_plot[4].set_xlabel('Lag (meses)')
        axes_plot[4].grid(True, alpha=0.2, linestyle='--')

        plot_pacf(resid_raw, ax=axes_plot[5], lags=24, alpha=0.05, color='#8e44ad',
                  vlines_kwargs={'colors': '#8e44ad'}, method='ywm')
        axes_plot[5].set_title('PACF — Autocorrelación Parcial', fontsize=11, fontweight='bold')
        axes_plot[5].set_xlabel('Lag (meses)')
        axes_plot[5].grid(True, alpha=0.2, linestyle='--')

        ax_metrics.axis('off')
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)

        print("\n" + "=" * 65)
        print("  DIAGNÓSTICO MATEMÁTICO EXTENDIDO")
        print("=" * 65)
        print(f"  1. R²            (Precisión):          {r2:.4f}")
        print(f"  2. MAE           (Error medio):        {mae:.2f}")
        print(f"  3. RMSE          (Error cuadrático):   {rmse:.2f}")
        print(f"  4. MAPE          (Error porcentual):   {mape:.2f}%")
        print(f"  5. AIC           (Complejidad):        {aic:.2f}")
        lb_estado = "OK" if p_lb > 0.05 else "⚠ Autocorrelación detectada"
        print(f"  6. Ljung-Box p:  {p_lb:.4f}  ({lb_estado})")
        jb_estado = "OK" if jb_p > 0.05 else "Residuos no-normales (IC aproximado)"
        print(f"  7. Jarque-Bera:  stat={jb_stat:.3f}, p={jb_p:.4f} → {jb_estado}")
        adf_estado = "Estacionaria" if adf_p <= 0.05 else "⚠ No estacionaria"
        print(f"  8. ADF Test:     stat={adf_stat:.3f}, p={adf_p:.4f} → {adf_estado}")
        print("=" * 65)

        c_ok   = '#27ae60'
        c_warn = '#e74c3c'
        c_neu  = '#2c3e50'
        c_hint = '#7f8c8d'
        c_card = '#e8ecf0'
        c_title = '#1a252f'

        ax_metrics.text(0.5, 0.975, 'Diagnóstico del Modelo',
                        transform=ax_metrics.transAxes,
                        ha='center', va='top', fontsize=12, fontweight='bold', color=c_title)
        ax_metrics.text(0.5, 0.945, 'SARIMAX  ·  √-transform  ·  MBB IC',
                        transform=ax_metrics.transAxes,
                        ha='center', va='top', fontsize=8, color=c_hint, style='italic')
        ax_metrics.add_line(Line2D([0.03, 0.97], [0.927, 0.927],
                                   transform=ax_metrics.transAxes,
                                   color='#bdc3c7', linewidth=1.0))

        grupos = [
            ('Calidad de Ajuste', '#2980b9', [
                ('R²  (Precisión)',    f'{r2:.4f}',    r2 >= 0.85,  'Umbral ≥ 0.85'),
                ('MAE  Error medio',  f'{mae:.1f}',   None,        ''),
                ('RMSE  Cuadrático',  f'{rmse:.1f}',  None,        ''),
                ('MAPE  Error %',     f'{mape:.2f}%', mape < 15,   'Umbral < 15%'),
                ('AIC  Complejidad',  f'{aic:.2f}',   None,        ''),
            ]),
            ('Residuos', '#8e44ad', [
                ('Ljung-Box  p',   f'{p_lb:.4f}',  p_lb > 0.05,  '> 0.05 sin autocorr.'),
                ('Jarque-Bera  p', f'{jb_p:.4f}',  jb_p > 0.05,  '> 0.05 normalidad'),
                ('skew / kurt',    f'{jb_skew:.3f} / {jb_kurt:.3f}', None, ''),
            ]),
            ('Estacionariedad', '#e67e22', [
                ('ADF  p',    f'{adf_p:.4f}',   adf_p <= 0.05, '≤ 0.05 estacionaria'),
                ('ADF  stat', f'{adf_stat:.3f}', None,          ''),
            ]),
        ]

        unidad_base = 0.038
        y = 0.915

        for titulo_g, color_g, items in grupos:
            n = len(items)
            altura_grupo = (1.2 + n * 1.55 + 0.3) * unidad_base

            rect = FancyBboxPatch(
                (0.02, y - altura_grupo), 0.96, altura_grupo,
                boxstyle='round,pad=0.01',
                facecolor=c_card, edgecolor='#d0d3d4', linewidth=0.7,
                transform=ax_metrics.transAxes, zorder=1,
            )
            ax_metrics.add_patch(rect)

            bar = FancyBboxPatch(
                (0.02, y - altura_grupo), 0.022, altura_grupo,
                boxstyle='round,pad=0.005',
                facecolor=color_g, edgecolor='none',
                transform=ax_metrics.transAxes, zorder=2,
            )
            ax_metrics.add_patch(bar)

            y_titulo = y - 1.0 * unidad_base
            ax_metrics.text(0.07, y_titulo, titulo_g,
                            transform=ax_metrics.transAxes,
                            va='center', fontsize=8.5, color=color_g, fontweight='bold',
                            zorder=3)

            y_item = y_titulo - 1.3 * unidad_base
            for label, valor, ok, hint in items:
                if ok is True:
                    color_v, semaf = c_ok, '✅ '
                elif ok is False:
                    color_v, semaf = c_warn, '⚠️ '
                else:
                    color_v, semaf = c_neu, '    '

                ax_metrics.text(0.07, y_item, f'{semaf}{label}',
                                transform=ax_metrics.transAxes,
                                va='center', fontsize=8, color=color_v, fontweight='bold',
                                zorder=3)
                ax_metrics.text(0.97, y_item, valor,
                                transform=ax_metrics.transAxes,
                                ha='right', va='center', fontsize=8.5, color=color_v,
                                fontweight='bold', zorder=3)
                if hint:
                    ax_metrics.text(0.10, y_item - 0.55 * unidad_base, hint,
                                    transform=ax_metrics.transAxes,
                                    va='center', fontsize=6.8, color=c_hint, style='italic',
                                    zorder=3)
                    y_item -= 1.55 * unidad_base
                else:
                    y_item -= 1.3 * unidad_base

            y = y - altura_grupo - 0.018

        plt.suptitle('Diagnóstico Matemático Extendido — SARIMAX',
                     fontsize=15, fontweight='bold', y=1.01)
        fig.savefig("G8_Diagnostico_Modelo.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print("  ✓ [G8] Gráfico guardado → G8_Diagnostico_Modelo.png")

    except Exception as e:
        print(f"  ERROR [G8]: {e}")
        traceback.print_exc()


# =============================================================================
# EJECUCIÓN PRINCIPAL
#
# Orquesta el pipeline completo en el siguiente orden:
#
#   1. Carga y preparación del DataFrame maestro.
#   2. Limpieza: forward-fill de RAM, relleno de NaN con 0, filtro de meses
#      completos (excluye el mes en curso).
#   3. Gráficos preliminares: STL (G3) y correlación (G4), antes del modelo.
#   4. Ajuste SARIMAX + bootstrap IC (Módulo 3) → objeto prediccion.
#   5. Generación de gráficos finales: G1, G2, G5, G6, G7, G8
# =============================================================================
try:
    print("Iniciando pipeline SARIMAX...")

    df_maestro = df_maestro_input.copy()

    if 'RAM_MB_ASIGNADA' in df_maestro.columns:
        df_maestro['RAM_MB_ASIGNADA'] = (
            df_maestro['RAM_MB_ASIGNADA']
            .replace(0, np.nan).ffill().fillna(0)
        )

    df_maestro = df_maestro.fillna(0)
    df_maestro = df_maestro.sort_index()
    df_maestro = df_maestro[
        df_maestro.index < pd.Timestamp.now().normalize().replace(day=1)
    ]

    col_inc = next((c for c in df_maestro.columns if 'INCIDENCIA' in c.upper()), None)
    col_cli = next((c for c in df_maestro.columns
                    if 'ACTIVO' in c.upper() or 'CLIENTE' in c.upper()), None)
    if col_inc and col_cli:
        corr = df_maestro[col_inc].corr(df_maestro[col_cli].astype(float))
        print(f"\n📊 Correlación Clientes-Incidencias: {corr:.3f}")

    generar_grafico_descomposicion(df_maestro)   # G3
    generar_grafico_correlacion(df_maestro)      # G4

    print("\n--- Iniciando Análisis Predictivo SARIMAX ---")
    prediccion, modelo_fit, exog_final, df_maestro, proxy_activo = \
        predecir_sarimax_multidimensional(df_maestro, periodos=58, n_bootstrap=1000)

    if prediccion is not None:
        df_ajuste_grafico = pd.DataFrame({
            'FECHA_DT': df_maestro.index,
            'AJUSTE':   np.power(modelo_fit.fittedvalues, 2).clip(lower=0),
        })

        generar_grafico_individual(df_maestro, prediccion, df_ajuste_grafico)          # G1
        generar_grafico_estudio_pendientes(df_maestro, prediccion,
                                           proxy_activo=proxy_activo)                  # G2
        generar_grafico_unificado(df_maestro, prediccion, proxy_activo=proxy_activo)   # G5
        generar_grafico_inventario_infra_individual(df_maestro)                        # G6
        generar_grafico_inventario_normalizado(df_maestro)                             # G7
        generar_grafico_diagnostico(df_maestro, modelo_fit)                            # G8

    print("\n✅ Pipeline completado. Gráficos generados:")
    print("   · G1_Prediccion_SARIMAX.png")
    print("   · G2_Pendientes.png")
    print("   · G3_Descomposicion_STL.png")
    print("   · G4_Correlacion.png")
    print("   · G5_Unificado_Normalizado.png")
    print("   · G6_Inventario_Individual.png")
    print("   · G7_Inventario_Normalizado.png")
    print("   · G8_Diagnostico_Modelo.png")

except Exception as e:
    print(f"\nERROR general: {e}")
    traceback.print_exc()
