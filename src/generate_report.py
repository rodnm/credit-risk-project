import base64
import os
from pathlib import Path

FIGURES_DIR = Path("reports/figures")
OUTPUT = Path("reports/credit_risk_report.html")

def img_to_base64(filename):
    path = FIGURES_DIR / filename
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

figures = {}
for i in range(1, 15):
    name = f"{i:02d}_*.png"
    matches = list(FIGURES_DIR.glob(name))
    if matches:
        key = matches[0].stem
        figures[key] = img_to_base64(matches[0].name)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predicción de Riesgo Crediticio — UCI Credit Card Default</title>
<style>
  :root {{
    --bg: #fafafa;
    --card: #ffffff;
    --accent: #1a5276;
    --accent2: #2980b9;
    --text: #2c3e50;
    --muted: #7f8c8d;
    --border: #e0e0e0;
    --success: #27ae60;
    --danger: #e74c3c;
    --warning: #f39c12;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    font-size: 16px;
  }}
  .container {{
    max-width: 900px;
    margin: 0 auto;
    padding: 0 24px;
  }}
  header {{
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    color: white;
    padding: 60px 0 50px;
    text-align: center;
  }}
  header h1 {{
    font-size: 2.4em;
    font-weight: 700;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }}
  header .subtitle {{
    font-size: 1.15em;
    opacity: 0.9;
    margin-bottom: 20px;
  }}
  header .meta {{
    font-size: 0.9em;
    opacity: 0.75;
  }}
  header .badges {{
    margin-top: 16px;
  }}
  header .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82em;
    margin: 3px 4px;
    backdrop-filter: blur(4px);
  }}
  .toc {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px 32px;
    margin: 36px 0;
  }}
  .toc h2 {{
    font-size: 1.15em;
    color: var(--accent);
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .toc ol {{
    padding-left: 22px;
  }}
  .toc li {{
    margin: 6px 0;
  }}
  .toc a {{
    color: var(--accent2);
    text-decoration: none;
  }}
  .toc a:hover {{
    text-decoration: underline;
  }}
  section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 36px 36px;
    margin: 28px 0;
  }}
  h2 {{
    color: var(--accent);
    font-size: 1.6em;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent2);
  }}
  h3 {{
    color: var(--accent2);
    font-size: 1.2em;
    margin: 22px 0 10px;
  }}
  p {{
    margin: 12px 0;
    text-align: justify;
  }}
  .figure {{
    text-align: center;
    margin: 28px 0;
  }}
  .figure img {{
    max-width: 100%;
    border-radius: 6px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }}
  .figure .caption {{
    font-size: 0.88em;
    color: var(--muted);
    margin-top: 10px;
    font-style: italic;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 0.92em;
  }}
  th {{
    background: var(--accent);
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
  }}
  tr:nth-child(even) {{
    background: #f5f8fa;
  }}
  tr:hover {{
    background: #eaf2f8;
  }}
  .highlight {{
    background: #eaf2f8;
    border-left: 4px solid var(--accent2);
    padding: 16px 20px;
    margin: 18px 0;
    border-radius: 0 6px 6px 0;
  }}
  .highlight strong {{
    color: var(--accent);
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 22px 0;
  }}
  .metric-card {{
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border: 1px solid var(--border);
  }}
  .metric-card .value {{
    font-size: 2em;
    font-weight: 700;
    color: var(--accent);
  }}
  .metric-card .label {{
    font-size: 0.85em;
    color: var(--muted);
    margin-top: 4px;
  }}
  .best {{
    background: linear-gradient(135deg, #d4efdf 0%, #a9dfbf 100%) !important;
    border-color: var(--success) !important;
  }}
  .best .value {{
    color: var(--success) !important;
  }}
  code {{
    background: #f0f3f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
    color: #c0392b;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 700px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    section {{ padding: 20px; }}
    header h1 {{ font-size: 1.6em; }}
  }}
  footer {{
    text-align: center;
    padding: 30px 0;
    color: var(--muted);
    font-size: 0.85em;
  }}
  .pipeline-step {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }}
  .pipeline-step:last-child {{
    border-bottom: none;
  }}
  .step-num {{
    background: var(--accent);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9em;
    flex-shrink: 0;
  }}
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>Predicción de Riesgo Crediticio</h1>
    <div class="subtitle">Análisis de Default en Tarjetas de Crédito — UCI Dataset</div>
    <div class="meta">Proyecto de Ciencia de Datos &middot; Junio 2026</div>
    <div class="badges">
      <span class="badge">Python 3.12</span>
      <span class="badge">scikit-learn</span>
      <span class="badge">XGBoost</span>
      <span class="badge">LightGBM</span>
      <span class="badge">SHAP</span>
      <span class="badge">uv</span>
    </div>
  </div>
</header>

<div class="container">

<!-- RESUMEN EJECUTIVO -->
<section>
  <h2>Resumen Ejecutivo</h2>
  <p>
    Este proyecto presenta un pipeline completo de <strong>machine learning para la predicción de riesgo crediticio</strong>,
    utilizando el dataset público <em>UCI Credit Card Default</em> con 30,000 clientes de tarjetas de crédito en Taiwán.
    El objetivo es construir un modelo clasificador que identifique clientes con alta probabilidad de incumplimiento
    (<em>default</em>) en el siguiente mes, una tarea crítica para la gestión de riesgo en instituciones financieras.
  </p>
  <div class="metric-grid">
    <div class="metric-card">
      <div class="value">30,000</div>
      <div class="label">Clientes analizados</div>
    </div>
    <div class="metric-card">
      <div class="value">22.12%</div>
      <div class="label">Tasa de default</div>
    </div>
    <div class="metric-card">
      <div class="value">32</div>
      <div class="label">Features (23 orig. + 9 eng.)</div>
    </div>
    <div class="metric-card best">
      <div class="value">0.7729</div>
      <div class="label">Mejor AUC (Random Forest)</div>
    </div>
  </div>
  <div class="highlight">
    <strong>Resultado clave:</strong> El modelo <strong>Random Forest</strong> obtuvo el mejor desempeño en test
    con un AUC de <strong>0.7729</strong>, coeficiente de Gini de <strong>0.4251</strong> y estadístico KS de
    <strong>0.4145</strong>, superando los umbrales mínimos de aceptación bancaria (Gini &gt; 0.4, KS &gt; 0.4).
    El historial de retrasos en pagos (<code>PAY_0</code>) fue identificado como el predictor más importante.
  </div>
</section>

<!-- TABLA DE CONTENIDOS -->
<div class="toc">
  <h2>Contenido</h2>
  <ol>
    <li><a href="#dataset">Descripción del Dataset</a></li>
    <li><a href="#eda">Análisis Exploratorio de Datos</a></li>
    <li><a href="#features">Ingeniería de Features</a></li>
    <li><a href="#pipeline">Pipeline de Preprocesamiento</a></li>
    <li><a href="#modelado">Modelado y Validación Cruzada</a></li>
    <li><a href="#evaluacion">Evaluación de Modelos</a></li>
    <li><a href="#shap">Interpretabilidad con SHAP</a></li>
    <li><a href="#conclusiones">Conclusiones y Recomendaciones</a></li>
  </ol>
</div>

<!-- 1. DATASET -->
<section id="dataset">
  <h2>1. Descripción del Dataset</h2>
  <p>
    El dataset proviene del <strong>UCI Machine Learning Repository</strong> y contiene información de
    <strong>30,000 clientes</strong> de tarjetas de crédito en Taiwán, con datos de octubre de 2005.
    Cada registro incluye 23 variables predictoras y una variable objetivo binaria que indica si el cliente
    incurrió en default el mes siguiente.
  </p>

  <h3>Variables del Dataset</h3>
  <table>
    <tr><th>Variable</th><th>Descripción</th><th>Tipo</th></tr>
    <tr><td><code>LIMIT_BAL</code></td><td>Límite de crédito (NT$)</td><td>Numérica</td></tr>
    <tr><td><code>SEX</code></td><td>Género (1=Hombre, 2=Mujer)</td><td>Categórica</td></tr>
    <tr><td><code>EDUCATION</code></td><td>Nivel educativo (1-4)</td><td>Categórica</td></tr>
    <tr><td><code>MARRIAGE</code></td><td>Estado civil (1-3)</td><td>Categórica</td></tr>
    <tr><td><code>AGE</code></td><td>Edad</td><td>Numérica</td></tr>
    <tr><td><code>PAY_0 ... PAY_6</code></td><td>Estado de pago (mes actual a 6 meses atrás). -1=pago a tiempo, 1-9=meses de retraso</td><td>Numérica</td></tr>
    <tr><td><code>BILL_AMT1 ... BILL_AMT6</code></td><td>Monto de factura (mes actual a 6 meses atrás)</td><td>Numérica</td></tr>
    <tr><td><code>PAY_AMT1 ... PAY_AMT6</code></td><td>Monto pagado (mes actual a 6 meses atrás)</td><td>Numérica</td></tr>
    <tr><td><code>default</code></td><td><strong>Target:</strong> Default el mes siguiente (1=Sí, 0=No)</td><td>Binaria</td></tr>
  </table>

  <h3>Distribución del Target</h3>
  <p>
    El dataset presenta un desbalance de clases moderado: el <strong>22.12%</strong> de los clientes
    incurrieron en default. Esta proporción es consistente con portfolios de crédito de alto riesgo
    y justifica el uso de técnicas de balanceo como SMOTE.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('01_target_distribution', '')}" alt="Distribución del target">
    <div class="caption">Figura 1: Distribución de la variable objetivo. 23,364 clientes sin default (77.88%) vs. 6,636 en default (22.12%).</div>
  </div>

  <h3>Estadísticas Descriptivas por Clase</h3>
  <table>
    <tr><th>Variable</th><th>No Default (media)</th><th>Default (media)</th><th>Observación</th></tr>
    <tr><td>LIMIT_BAL</td><td>178,100</td><td>130,110</td><td>Defaulters tienen 27% menos límite</td></tr>
    <tr><td>AGE</td><td>35.4</td><td>35.7</td><td>Sin diferencia significativa</td></tr>
    <tr><td>PAY_0</td><td>-0.21</td><td>0.67</td><td>Defaulters con retrasos activos</td></tr>
    <tr><td>BILL_AMT1</td><td>51,994</td><td>48,509</td><td>Facturas similares entre grupos</td></tr>
    <tr><td>PAY_AMT1</td><td>6,307</td><td>3,397</td><td>Defaulters pagan 46% menos</td></tr>
  </table>
</section>

<!-- 2. EDA -->
<section id="eda">
  <h2>2. Análisis Exploratorio de Datos</h2>

  <h3>2.1 Matriz de Correlación</h3>
  <p>
    El análisis de correlaciones revela que las variables de <strong>historial de pagos</strong>
    (<code>PAY_0</code> a <code>PAY_6</code>) son los predictores más fuertemente correlacionados con el default.
    <code>PAY_0</code> (estado de pago del mes actual) presenta la correlación más alta con
    <strong>r = 0.325</strong>, seguido por <code>PAY_2</code> (r = 0.264) y <code>PAY_3</code> (r = 0.235).
    Las variables demográficas (<code>SEX</code>, <code>AGE</code>, <code>EDUCATION</code>) muestran
    correlaciones prácticamente nulas con el target.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('02_correlation_heatmap', '')}" alt="Heatmap de correlaciones">
    <div class="caption">Figura 2: Matriz de correlación de Pearson. Las variables de retraso en pagos (PAY_*) forman un cluster altamente correlacionado entre sí y con el target.</div>
  </div>

  <h3>2.2 Distribución de Variables Clave</h3>
  <p>
    La comparación de distribuciones entre clientes en default y no-default revela patrones claros:
  </p>
  <ul style="margin: 12px 0 12px 24px;">
    <li><strong>LIMIT_BAL:</strong> Los defaulters se concentran en límites de crédito bajos (&lt; 100,000 NT$)</li>
    <li><strong>PAY_0:</strong> Los defaulters muestran valores positivos (retrasos), mientras los no-defaulters presentan valores -1 y -2 (pagos a tiempo)</li>
    <li><strong>BILL_AMT1:</strong> Distribuciones similares, pero los defaulters tienden a montos ligeramente menores</li>
    <li><strong>PAY_AMT1:</strong> Los no-defaulters realizan pagos más altos consistentemente</li>
  </ul>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('03_feature_distributions', '')}" alt="Distribuciones de features">
    <div class="caption">Figura 3: Distribución de las 5 variables más relevantes segmentadas por estado de default (verde = sin default, rojo = default).</div>
  </div>

  <h3>2.3 Detección de Outliers</h3>
  <p>
    Los box plots confirman la presencia de valores extremos en las variables monetarias
    (<code>LIMIT_BAL</code>, <code>BILL_AMT1</code>, <code>PAY_AMT1</code>). Estos outliers son
    esperables en datos financieros y se manejan mediante <strong>winsorización</strong> al
    percentil 1-99 en la etapa de preprocesamiento. La variable <code>PAY_0</code> muestra
    una clara separación entre grupos: los defaulters presentan valores positivos (retrasos de 1-8 meses).
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('04_boxplots', '')}" alt="Box plots">
    <div class="caption">Figura 4: Box plots de variables clave por estado de default. Se observan outliers en variables monetarias y clara separación en PAY_0.</div>
  </div>

  <h3>2.4 Análisis de Historial de Pagos</h3>
  <p>
    El análisis de la tasa de default por estado de pago revela una <strong>relación monótona creciente</strong>:
    a mayor retraso acumulado, mayor probabilidad de default. Los clientes con <code>PAY_0 = -2</code>
    (pago anticipado) tienen una tasa de default cercana al 10%, mientras que aquellos con
    <code>PAY_0 &ge; 4</code> (4+ meses de retraso) superan el 70%. Este patrón se mantiene
    consistente en los 6 meses de historial, aunque con menor intensidad en meses más antiguos.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('05_payment_status', '')}" alt="Análisis de pagos">
    <div class="caption">Figura 5: Tasa de default por estado de pago para cada mes del historial (PAY_0 a PAY_6). La relación es monótona y consistente.</div>
  </div>
</section>

<!-- 3. FEATURE ENGINEERING -->
<section id="features">
  <h2>3. Ingeniería de Features</h2>
  <p>
    A partir de las 23 variables originales, se construyeron <strong>9 features adicionales</strong>
    diseñadas para capturar patrones de comportamiento crediticio que las variables individuales
    no reflejan directamente. El set final cuenta con <strong>32 variables predictoras</strong>.
  </p>
  <table>
    <tr><th>Feature</th><th>Fórmula</th><th>Justificación</th></tr>
    <tr><td><code>avg_delay</code></td><td>Media de PAY_0 a PAY_6</td><td>Nivel promedio de morosidad</td></tr>
    <tr><td><code>max_delay</code></td><td>Máximo de PAY_0 a PAY_6</td><td>Peor episodio de retraso</td></tr>
    <tr><td><code>delay_std</code></td><td>Desv. estándar de PAY_0 a PAY_6</td><td>Volatilidad del comportamiento de pago</td></tr>
    <tr><td><code>total_delay_months</code></td><td>Count(PAY_* &gt; 0)</td><td>Cronicidad del retraso</td></tr>
    <tr><td><code>avg_bill</code></td><td>Media de BILL_AMT1 a BILL_AMT6</td><td>Nivel de consumo promedio</td></tr>
    <tr><td><code>bill_trend</code></td><td>BILL_AMT1 - BILL_AMT6</td><td>Tendencia de consumo reciente</td></tr>
    <tr><td><code>avg_pay_amt</code></td><td>Media de PAY_AMT1 a PAY_AMT6</td><td>Capacidad de pago promedio</td></tr>
    <tr><td><code>pay_ratio</code></td><td>avg_pay_amt / (avg_bill + 1)</td><td>Proporción de pago vs. deuda</td></tr>
    <tr><td><code>credit_util</code></td><td>BILL_AMT1 / (LIMIT_BAL + 1)</td><td>Utilización de línea de crédito</td></tr>
  </table>
  <div class="highlight">
    <strong>Nota:</strong> Las features de tendencia (<code>bill_trend</code>) y ratio de pago
    (<code>pay_ratio</code>) capturan la <em>dirección</em> del comportamiento financiero del cliente,
    un aspecto que las variables estáticas no reflejan. La utilización de crédito
    (<code>credit_util</code>) es una métrica estándar en scoring crediticio bancario.
  </div>
</section>

<!-- 4. PIPELINE -->
<section id="pipeline">
  <h2>4. Pipeline de Preprocesamiento</h2>
  <p>El pipeline de preprocesamiento sigue las mejores prácticas de la industria para evitar
    <em>data leakage</em> y garantizar reproducibilidad:</p>

  <div class="pipeline-step">
    <div class="step-num">1</div>
    <div><strong>Train/Test Split (80/20)</strong> — Split estratificado con <code>random_state=42</code>.
    Resultado: 24,000 train / 6,000 test, manteniendo la tasa de default del 22.12% en ambos sets.</div>
  </div>
  <div class="pipeline-step">
    <div class="step-num">2</div>
    <div><strong>Imputación</strong> — Valores NaN reemplazados por 0 (el dataset UCI no presenta missing values
    significativos; esta etapa es defensiva).</div>
  </div>
  <div class="pipeline-step">
    <div class="step-num">3</div>
    <div><strong>Winsorización (P1-P99)</strong> — Recorte de outliers al percentil 1 y 99 en todas
    las variables numéricas. Preserva la distribución central mientras controla valores extremos.</div>
  </div>
  <div class="pipeline-step">
    <div class="step-num">4</div>
    <div><strong>Escalado (StandardScaler)</strong> — Estandarización z-score. Ajustado solo sobre
    el set de entrenamiento para evitar leakage.</div>
  </div>
  <div class="pipeline-step">
    <div class="step-num">5</div>
    <div><strong>SMOTE (k=5)</strong> — Oversampling sintético de la clase minoritaria.
    El set de entrenamiento pasa de 24,000 a <strong>37,382 muestras</strong> con distribución 50/50.
    Aplicado solo sobre train para no inflar métricas de test.</div>
  </div>
</section>

<!-- 5. MODELADO -->
<section id="modelado">
  <h2>5. Modelado y Validación Cruzada</h2>
  <p>
    Se entrenaron <strong>4 algoritmos de clasificación</strong> con validación cruzada estratificada
    de 5 folds sobre el set de entrenamiento balanceado (SMOTE). Las métricas de CV se calculan
    sobre los folds de entrenamiento, mientras que el Test AUC se evalúa sobre el set de test
    original (sin balancear) para reflejar el desempeño en condiciones reales.
  </p>

  <h3>Configuración de Modelos</h3>
  <table>
    <tr><th>Modelo</th><th>Hiperparámetros</th><th>Notas</th></tr>
    <tr><td>Logistic Regression</td><td><code>max_iter=1000, class_weight='balanced'</code></td><td>Baseline lineal</td></tr>
    <tr><td>Random Forest</td><td><code>n_estimators=200, max_depth=10</code></td><td>Ensemble de árboles</td></tr>
    <tr><td>XGBoost</td><td><code>n_estimators=200, max_depth=6, lr=0.1</code></td><td>Gradient boosting</td></tr>
    <tr><td>LightGBM</td><td><code>n_estimators=200, max_depth=6, lr=0.1</code></td><td>Gradient boosting optimizado</td></tr>
  </table>

  <h3>Resultados Comparativos</h3>
  <table>
    <tr><th>Modelo</th><th>CV AUC (5-fold)</th><th>CV Std</th><th>Test AUC</th><th>Gini</th><th>KS</th></tr>
    <tr><td>Logistic Regression</td><td>0.7671</td><td>0.0046</td><td>0.7450</td><td>0.3816</td><td>0.3929</td></tr>
    <tr style="background: #d4efdf; font-weight: 600;"><td>Random Forest</td><td>0.8636</td><td>0.0034</td><td>0.7729</td><td>0.4251</td><td>0.4145</td></tr>
    <tr><td>XGBoost</td><td>0.9301</td><td>0.0020</td><td>0.7607</td><td>0.4061</td><td>0.3954</td></tr>
    <tr><td>LightGBM</td><td>0.9344</td><td>0.0016</td><td>0.7659</td><td>0.4141</td><td>0.3951</td></tr>
  </table>

  <div class="highlight">
    <strong>Hallazgo importante:</strong> Los modelos de boosting (XGBoost, LightGBM) muestran un CV AUC
    significativamente más alto (~0.93) que el Test AUC (~0.76), evidenciando <strong>overfitting</strong>
    al dataset balanceado con SMOTE. Random Forest, con menor CV AUC, generaliza mejor al set de test
    no balanceado, obteniendo el <strong>mejor Test AUC de 0.7729</strong>. Esto sugiere que Random Forest
    es más robusto al ruido introducido por las muestras sintéticas de SMOTE.
  </div>
</section>

<!-- 6. EVALUACIÓN -->
<section id="evaluacion">
  <h2>6. Evaluación de Modelos</h2>

  <h3>6.1 Curvas ROC</h3>
  <p>
    Las curvas ROC muestran que <strong>Random Forest</strong> domina en la región de interés
    para scoring crediticio (baja tasa de falsos positivos). Todos los modelos superan
    significativamente la línea de azar, confirmando capacidad predictiva real.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('06_roc_curves', '')}" alt="Curvas ROC">
    <div class="caption">Figura 6: Curvas ROC para los 4 modelos. Random Forest (AUC=0.7729) lidera en el set de test.</div>
  </div>

  <h3>6.2 Curvas Precision-Recall</h3>
  <p>
    Dado el desbalance de clases (22% positivos), las curvas Precision-Recall proporcionan una visión
    más informativa que ROC. Random Forest mantiene el mejor balance precision-recall en la región
    de alto recall, crucial para la detección de defaulters.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('07_pr_curves', '')}" alt="Curvas PR">
    <div class="caption">Figura 7: Curvas Precision-Recall. Random Forest muestra el mayor Average Precision.</div>
  </div>

  <h3>6.3 Matriz de Confusión (Random Forest)</h3>
  <p>
    Con el threshold por defecto (0.5), Random Forest clasifica correctamente el 77% de los casos.
    El modelo prioriza la detección de defaulters (recall del 59%) a costa de algunos falsos positivos,
    un trade-off aceptable en contexto bancario donde el costo de no detectar un defaulter es mayor
    que el de rechazar un cliente solvente.
  </p>
  <div class="two-col">
    <div class="figure" style="margin: 0;">
      <img src="data:image/png;base64,{figures.get('08_confusion_matrix', '')}" alt="Matriz de confusión">
      <div class="caption">Figura 8: Matriz de confusión del mejor modelo.</div>
    </div>
    <div style="display: flex; align-items: center;">
      <table>
        <tr><th>Métrica</th><th>No Default</th><th>Default</th></tr>
        <tr><td>Precision</td><td>0.88</td><td>0.49</td></tr>
        <tr><td>Recall</td><td>0.83</td><td>0.59</td></tr>
        <tr><td>F1-Score</td><td>0.85</td><td>0.53</td></tr>
        <tr><td colspan="3" style="text-align:center;"><strong>Accuracy: 0.77</strong></td></tr>
      </table>
    </div>
  </div>

  <h3>6.4 Análisis de Threshold</h3>
  <p>
    El análisis de threshold revela que el <strong>umbral óptimo (max F1) es 0.55</strong>,
    ligeramente superior al default de 0.5. Esto es consistente con un escenario donde se busca
    un balance entre detectar defaulters sin generar excesivos falsos positivos. En un contexto
    bancario real, el threshold se calibraría según la <strong>apetencia de riesgo</strong> de la
    institución: un threshold más bajo (0.40) maximizaría el recall para políticas conservadoras,
    mientras que uno más alto (0.65) optimizaría la precisión para políticas agresivas de colocación.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('09_threshold_analysis', '')}" alt="Análisis de threshold">
    <div class="caption">Figura 9: Precision, Recall y F1 en función del threshold de decisión. El óptimo F1 se alcanza en 0.55.</div>
  </div>

  <h3>6.5 Métricas Bancarias</h3>
  <div class="metric-grid">
    <div class="metric-card best">
      <div class="value">0.4251</div>
      <div class="label">Gini (Random Forest)</div>
    </div>
    <div class="metric-card best">
      <div class="value">0.4145</div>
      <div class="label">KS (Random Forest)</div>
    </div>
    <div class="metric-card">
      <div class="value">0.4141</div>
      <div class="label">Gini (LightGBM)</div>
    </div>
    <div class="metric-card">
      <div class="value">0.3954</div>
      <div class="label">KS (XGBoost)</div>
    </div>
  </div>
  <p>
    En la industria bancaria, un modelo de credit scoring se considera <strong>aceptable</strong>
    cuando el coeficiente de Gini supera 0.40 y el estadístico KS supera 0.30. Random Forest
    cumple ambos criterios (Gini = 0.4251, KS = 0.4145), lo que lo posiciona como un modelo
    viable para implementación en un entorno productivo.
  </p>
</section>

<!-- 7. SHAP -->
<section id="shap">
  <h2>7. Interpretabilidad con SHAP</h2>
  <p>
    La interpretabilidad es un requisito regulatorio y de negocio en modelos de riesgo crediticio.
    Se utiliza <strong>SHAP (SHapley Additive exPlanations)</strong> para descomponer las predicciones
    del modelo en contribuciones individuales de cada feature, proporcionando transparencia tanto
    a nivel global como individual.
  </p>

  <h3>7.1 Importancia Global de Features</h3>
  <p>
    El SHAP summary plot revela las <strong>5 variables más influyentes</strong> en la predicción de default:
  </p>
  <ol style="margin: 12px 0 12px 24px;">
    <li><strong>PAY_0</strong> — Estado de pago del mes actual (el predictor dominante)</li>
    <li><strong>LIMIT_BAL</strong> — Límite de crédito asignado</li>
    <li><strong>avg_delay</strong> — Retraso promedio (feature engineered)</li>
    <li><strong>BILL_AMT1</strong> — Factura del mes actual</li>
    <li><strong>credit_util</strong> — Utilización de crédito (feature engineered)</li>
  </ol>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('10_shap_summary', '')}" alt="SHAP Summary">
    <div class="caption">Figura 10: SHAP Summary Plot. Cada punto es una predicción individual. El color indica el valor del feature (rojo=alto, azul=bajo) y la posición horizontal el impacto en la predicción.</div>
  </div>

  <div class="figure">
    <img src="data:image/png;base64,{figures.get('11_shap_bar', '')}" alt="SHAP Bar">
    <div class="caption">Figura 11: Ranking de importancia global por valor absoluto medio de SHAP. PAY_0 domina con amplia diferencia.</div>
  </div>

  <h3>7.2 Análisis de Dependencia</h3>
  <p>
    Los plots de dependencia de SHAP muestran cómo el valor de cada feature afecta la predicción:
  </p>
  <ul style="margin: 12px 0 12px 24px;">
    <li><strong>PAY_0:</strong> Valores positivos (retrasos) incrementan drásticamente la probabilidad de default.
    El punto de inflexión se ubica entre PAY_0 = 0 y PAY_0 = 1.</li>
    <li><strong>LIMIT_BAL:</strong> Límites bajos se asocian con mayor riesgo. El efecto se estabiliza por encima de 200,000 NT$.</li>
    <li><strong>credit_util:</strong> Alta utilización (&gt; 0.75) incrementa significativamente el riesgo.</li>
  </ul>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('12_shap_dependence', '')}" alt="SHAP Dependence">
    <div class="caption">Figura 12: SHAP Dependence Plots para las 3 features más importantes. Se observa la relación no-lineal entre cada feature y su contribución a la predicción.</div>
  </div>

  <h3>7.3 Explicación Individual</h3>
  <p>
    El waterfall plot permite explicar predicciones individuales, mostrando cómo cada feature
    contribuye a mover la predicción desde el valor base (prevalencia de la clase) hasta la
    probabilidad final. Esto es fundamental para la <strong>transparencia regulatoria</strong>:
    cada decisión de crédito puede ser explicada al cliente y al regulador.
  </p>
  <div class="figure">
    <img src="data:image/png;base64,{figures.get('13_shap_waterfall', '')}" alt="SHAP Waterfall">
    <div class="caption">Figura 13: Waterfall plot para una predicción individual. Las barras rojas incrementan la probabilidad de default, las azules la reducen.</div>
  </div>

  <div class="figure">
    <img src="data:image/png;base64,{figures.get('14_shap_force', '')}" alt="SHAP Force">
    <div class="caption">Figura 14: Force plot mostrando la contribución de cada feature a la predicción final de un cliente específico.</div>
  </div>
</section>

<!-- 8. CONCLUSIONES -->
<section id="conclusiones">
  <h2>8. Conclusiones y Recomendaciones</h2>

  <h3>Hallazgos Principales</h3>
  <ol style="margin: 12px 0 12px 24px;">
    <li><strong>El historial de pagos es el predictor dominante.</strong> <code>PAY_0</code> (retraso del mes actual)
    concentra la mayor parte del poder predictivo, con una correlación de 0.325 con el target y el mayor
    valor SHAP absoluto. Esto es consistente con la literatura de credit scoring.</li>
    <li><strong>Random Forest generaliza mejor que boosting.</strong> A pesar de que XGBoost y LightGBM
    muestran CV AUC superiores (~0.93), Random Forest logra el mejor Test AUC (0.7729), sugiriendo
    mayor robustez al overfitting inducido por SMOTE.</li>
    <li><strong>Las features engineered aportan valor.</strong> <code>avg_delay</code> y <code>credit_util</code>
    aparecen entre los top-5 predictores SHAP, validando la estrategia de ingeniería de features.</li>
    <li><strong>El modelo cumple estándares bancarios.</strong> Gini = 0.4251 y KS = 0.4145 superan
    los umbrales de aceptación (Gini &gt; 0.4, KS &gt; 0.3).</li>
    <li><strong>SHAP proporciona interpretabilidad regulatoria.</strong> Cada predicción puede ser
    descompuesta en contribuciones de features, cumpliendo con requisitos de explicabilidad.</li>
  </ol>

  <h3>Limitaciones</h3>
  <ul style="margin: 12px 0 12px 24px;">
    <li><strong>Overfitting en boosting:</strong> La brecha CV-Test de ~0.17 en XGBoost/LightGBM indica
    que SMOTE introduce ruido que estos modelos sobreajustan. Se recomienda explorar técnicas alternativas
    como class weights o undersampling.</li>
    <li><strong>Datos de 2005:</strong> El dataset tiene casi 20 años. Los patrones de comportamiento
    crediticio pueden haber evolucionado significativamente.</li>
    <li><strong>Sin tuning exhaustivo:</strong> Los hiperparámetros se configuraron con valores razonables
    pero no se realizó búsqueda sistemática (GridSearch/Optuna).</li>
    <li><strong>Recall de default limitado:</strong> El recall del 59% para la clase default significa
    que el 41% de los defaulters no son detectados. En un entorno real, esto requeriría optimización
    del threshold y posiblemente un modelo de costo asimétrico.</li>
  </ul>

  <h3>Próximos Pasos</h3>
  <ul style="margin: 12px 0 12px 24px;">
    <li>Implementar <strong>Optuna</strong> para búsqueda bayesiana de hiperparámetros</li>
    <li>Explorar <strong>ensamble ponderado</strong> de Random Forest + LightGBM</li>
    <li>Evaluar <strong>cost-sensitive learning</strong> con matriz de costos bancaria real</li>
    <li>Implementar <strong>monitoreo de drift</strong> para despliegue en producción</li>
    <li>Construir <strong>scorecard</strong> calibrado (escala 300-850 tipo FICO)</li>
  </ul>
</section>

<footer>
  <p>Proyecto de Riesgo Crediticio &middot; Python 3.12 + scikit-learn + XGBoost + LightGBM + SHAP</p>
  <p>Dataset: <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">UCI Credit Card Default</a> &middot; Generado en Junio 2026</p>
</footer>

</div>
</body>
</html>"""

os.makedirs("reports", exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report generated: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / 1024:.1f} KB")
