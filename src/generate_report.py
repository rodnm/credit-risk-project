"""Render the original report layout with corrected, measured evaluation results."""
import base64
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports/figures"
OUTPUT = ROOT / "reports/credit_risk_report.html"
METRICS = ROOT / "models/evaluation_metrics.json"
SHAP_METADATA = ROOT / "models/interpretability_metadata.json"

def generate_report():
    if not METRICS.exists():
        raise FileNotFoundError("Ejecute 02_Modeling.ipynb para generar evaluation_metrics.json.")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != 1 or metrics.get("threshold_source") != "training_out_of_fold":
        raise ValueError("Se requieren métricas del pipeline corregido y umbral OOF.")
    if not SHAP_METADATA.exists():
        raise FileNotFoundError("Ejecute 03_Interpretability.ipynb antes de generar el informe.")
    shap_metadata = json.loads(SHAP_METADATA.read_text(encoding="utf-8"))
    if (shap_metadata["best_model"] != metrics["best_model"] or
            shap_metadata["dataset_sha256"] != metrics["dataset_sha256"]):
        raise ValueError("Las figuras SHAP no corresponden a esta evaluación.")
    winner = metrics["best_model"]
    winner_html = escape(winner)
    winner_result = metrics["models"][winner]
    best = metrics["best_test_metrics"]
    threshold = metrics["threshold"]
    tn, fp = best["confusion_matrix"][0]
    fn, tp = best["confusion_matrix"][1]
    negative_precision = tn / (tn + fn)
    negative_recall = tn / (tn + fp)
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall)
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    model_rows = ""
    for name, result in metrics["models"].items():
        style = ' style="background: #d4efdf; font-weight: 600;"' if name == winner else ""
        model_rows += (
            f"<tr{style}><td>{escape(name)}</td>"
            f"<td>{result['cv_mean']:.4f}</td><td>{result['cv_std']:.4f}</td>"
            f"<td>{result['test_auc']:.4f}</td><td>{result['gini']:.4f}</td>"
            f"<td>{result['ks']:.4f}</td></tr>\n"
        )
    figures = {}
    for number in range(1, 15):
        matches = list(FIGURES_DIR.glob(f"{number:02d}_*.png"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Se esperaba una figura {number:02d} en {FIGURES_DIR}")
        figures[matches[0].stem] = base64.b64encode(matches[0].read_bytes()).decode("ascii")

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
    <div class="meta">Proyecto de Ciencia de Datos &middot; Evaluación actualizada</div>
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
    Este proyecto presenta un pipeline de <strong>machine learning para predicción de riesgo crediticio</strong>
    con el dataset público <em>UCI Credit Card Default</em>: 30,000 clientes y un objetivo binario
    de incumplimiento en el mes siguiente. La evaluación corregida separa el test antes de ajustar
    estadísticas y aplica SMOTE únicamente al entrenamiento de cada fold.
  </p>
  <div class="metric-grid">
    <div class="metric-card"><div class="value">30,000</div><div class="label">Clientes analizados</div></div>
    <div class="metric-card"><div class="value">22.12%</div><div class="label">Tasa de default</div></div>
    <div class="metric-card"><div class="value">32</div><div class="label">Variables predictoras</div></div>
    <div class="metric-card best"><div class="value">{winner_result['test_auc']:.4f}</div>
      <div class="label">Test AUC ({winner_html})</div></div>
  </div>
  <div class="highlight">
    <strong>Resultado clave:</strong> {winner_html} fue seleccionado por AUC media de validación cruzada
    ({winner_result['cv_mean']:.4f} ± {winner_result['cv_std']:.4f}); obtuvo AUC {winner_result['test_auc']:.4f}
    en test. La brecha CV−test es {winner_result['cv_mean'] - winner_result['test_auc']:.4f}.
    El umbral {threshold:.2f} se eligió mediante predicciones fuera de fold del entrenamiento.
  </div>
  <p>El test actual ya se había usado para elegir modelos en la versión anterior. La comparación
  corregida es informativa, pero requiere datos nuevos para una confirmación independiente.</p>
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
    incurrieron en default. Esta proporción motiva evaluar técnicas para tratar el desbalance, entre ellas SMOTE.
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
  <p>Se separan primero 24,000 clientes para entrenamiento y 6,000 para test. Cada candidato
  utiliza un <code>imblearn.pipeline.Pipeline</code> con las mismas transformaciones:</p>
  <div class="pipeline-step"><div class="step-num">1</div>
    <div><strong>Split 80/20 estratificado</strong> — <code>random_state=42</code>;
    el test conserva clientes originales y no participa en ajustes o selección.</div></div>
  <div class="pipeline-step"><div class="step-num">2</div>
    <div><strong>Imputación</strong> — Los valores faltantes se sustituyen por cero; el paso se ajusta
    únicamente con el entrenamiento de cada fold.</div></div>
  <div class="pipeline-step"><div class="step-num">3</div>
    <div><strong>Winsorización (P1–P99)</strong> — Los límites se aprenden en el entrenamiento del fold
    y se aplican sin recalcularlos a validación y test.</div></div>
  <div class="pipeline-step"><div class="step-num">4</div>
    <div><strong>StandardScaler</strong> — Aprende medias y desviaciones del entrenamiento del fold.</div></div>
  <div class="pipeline-step"><div class="step-num">5</div>
    <div><strong>SMOTE (k=5)</strong> — Genera ejemplos sintéticos solo al ajustar con el entrenamiento
    del fold. Validación y test conservan sus filas originales; el pipeline omite SMOTE al predecir.</div></div>
  <p>Tras elegir el modelo y el umbral con datos de entrenamiento, el pipeline ganador se vuelve
  a ajustar con los 24,000 clientes y se evalúa una vez sobre test.</p>
</section>

<!-- 5. MODELADO -->
<section id="modelado">
  <h2>5. Modelado y Validación Cruzada</h2>
  <p>Se comparan cuatro clasificadores con <strong>validación estratificada de 5 folds</strong>
  (<code>shuffle=True</code>, semilla 42). Cada fold reajusta imputación, winsorización,
  escalado y SMOTE con sus propios clientes de entrenamiento. El AUC se mide en los
  clientes originales de validación. La selección usa la media de esos cinco AUC.</p>
  <h3>Configuración de Modelos</h3>
  <table>
    <tr><th>Modelo</th><th>Hiperparámetros</th><th>Notas</th></tr>
    <tr><td>Logistic Regression</td><td><code>max_iter=1000, class_weight='balanced'</code></td><td>Baseline lineal</td></tr>
    <tr><td>Random Forest</td><td><code>n_estimators=200, max_depth=10</code></td><td>Ensemble de árboles</td></tr>
    <tr><td>XGBoost</td><td><code>n_estimators=200, max_depth=6, lr=0.1</code></td><td>Gradient boosting</td></tr>
    <tr><td>LightGBM</td><td><code>n_estimators=200, max_depth=6, lr=0.1</code></td><td>Gradient boosting</td></tr>
  </table>
  <h3>Resultados Comparativos</h3>
  <table>
    <tr><th>Modelo</th><th>CV AUC (5 folds)</th><th>CV Std</th><th>Test AUC</th><th>Gini normalizado</th><th>KS</th></tr>
    {model_rows}
  </table>
  <p>CV Std describe la variación entre folds; no es un intervalo de confianza.
  Los AUC de test de los demás modelos son comparaciones descriptivas y no cambiaron la selección.</p>
  <div class="highlight">
    <strong>Por qué cambió la brecha:</strong> En la versión anterior se aplicaba SMOTE a todo train
    antes de crear los folds. Una muestra original y derivados sintéticos podían quedar en lados
    opuestos de una partición, y la validación incluía clientes artificiales.
    El CV histórico de {winner_html} era 0.8636 frente a 0.7729 en test; con el protocolo
    corregido es {winner_result['cv_mean']:.4f} frente a {winner_result['test_auc']:.4f}.
    La winsorización y el escalado también cambiaron de alcance, por lo que no es posible
    atribuir toda la diferencia numérica solo a SMOTE.
  </div>
  <p><a href="https://imbalanced-learn.org/stable/common_pitfalls.html">
  Documentación de imbalanced-learn sobre fuga por remuestreo previo a CV</a>.</p>
</section>

<!-- 6. EVALUACIÓN -->
<section id="evaluacion">
  <h2>6. Evaluación de Modelos</h2>
  <h3>6.1 Curvas ROC</h3>
  <p>Las curvas muestran el desempeño descriptivo de los cuatro pipelines en el test original.
  El ganador ya se había seleccionado por CV.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['06_roc_curves']}" alt="Curvas ROC">
    <div class="caption">Figura 6: curvas ROC en test; {winner_html} obtuvo AUC {winner_result['test_auc']:.4f}.</div></div>
  <h3>6.2 Curvas Precision-Recall</h3>
  <p>Estas curvas complementan el AUC ROC para el objetivo con 22.12% de casos positivos.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['07_pr_curves']}" alt="Curvas Precision-Recall">
    <div class="caption">Figura 7: precisión y recall en clientes originales de test.</div></div>
  <h3>6.3 Matriz de Confusión ({winner_html})</h3>
  <p>Con el umbral {threshold:.2f}, fijado en entrenamiento mediante predicciones fuera de fold,
  se observan {tp} verdaderos positivos, {fn} falsos negativos, {fp} falsos positivos
  y {tn} verdaderos negativos en test.</p>
  <div class="two-col">
    <div class="figure" style="margin: 0;"><img src="data:image/png;base64,{figures['08_confusion_matrix']}" alt="Matriz de confusión">
      <div class="caption">Figura 8: matriz de confusión con el umbral seleccionado en train.</div></div>
    <div style="display: flex; align-items: center;"><table>
      <tr><th>Métrica</th><th>No Default</th><th>Default</th></tr>
      <tr><td>Precision</td><td>{negative_precision:.4f}</td><td>{best['precision']:.4f}</td></tr>
      <tr><td>Recall</td><td>{negative_recall:.4f}</td><td>{best['recall']:.4f}</td></tr>
      <tr><td>F1-Score</td><td>{negative_f1:.4f}</td><td>{best['f1']:.4f}</td></tr>
      <tr><td colspan="3" style="text-align:center;"><strong>Accuracy: {accuracy:.4f}</strong></td></tr>
    </table></div>
  </div>
  <h3>6.4 Análisis de Umbral</h3>
  <p>Se probaron umbrales de 0.10 a 0.85 en incrementos de 0.05. El máximo F1
  de las predicciones <strong>fuera de fold del entrenamiento</strong> se alcanzó en {threshold:.2f}.
  Su elección no utilizó las etiquetas de test.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['09_threshold_analysis']}" alt="Análisis de umbral">
    <div class="caption">Figura 9: precisión, recall y F1 fuera de fold usados para elegir el umbral.</div></div>
  <h3>6.5 Gini y KS</h3>
  <div class="metric-grid">
    <div class="metric-card best"><div class="value">{winner_result['gini']:.4f}</div><div class="label">Gini normalizado ({winner_html})</div></div>
    <div class="metric-card best"><div class="value">{winner_result['ks']:.4f}</div><div class="label">KS ({winner_html})</div></div>
    <div class="metric-card"><div class="value">{winner_result['cv_mean']:.4f}</div><div class="label">AUC medio CV</div></div>
    <div class="metric-card"><div class="value">{winner_result['test_auc']:.4f}</div><div class="label">AUC test</div></div>
  </div>
  <p>Gini normalizado = 2 × AUC − 1. Estas métricas por sí solas no acreditan
  aptitud para decisiones bancarias o despliegue.</p>
</section>

<!-- 7. SHAP -->
<section id="shap">
  <h2>7. Interpretabilidad con SHAP</h2>
  <p>SHAP explica el <strong>estimador ganador por CV ({winner_html})</strong> después de aplicar
  los transformadores ya ajustados del pipeline. No vuelve a ajustar el preprocesamiento
  y no aplica SMOTE al test. Las figuras usan una muestra reproducible de
  {shap_metadata['sample_size']} clientes originales de test. Las contribuciones describen
  el comportamiento del modelo y no efectos causales.</p>
  <h3>7.1 Importancia Global de Variables</h3>
  <p><code>PAY_0</code> mantiene la mayor importancia media. Entre las variables derivadas
  destacan <code>max_delay</code>, <code>total_delay_months</code> y <code>avg_delay</code>.
  Las cifras no prueban que añadirlas mejorara el rendimiento, pues no se hizo una
  comparación sin esas variables.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['10_shap_summary']}" alt="SHAP Summary">
    <div class="caption">Figura 10: impacto de variables sobre la predicción en la muestra de test.</div></div>
  <div class="figure"><img src="data:image/png;base64,{figures['11_shap_bar']}" alt="SHAP Bar">
    <div class="caption">Figura 11: importancia global por valor absoluto medio de SHAP.</div></div>
  <h3>7.2 Análisis de Dependencia</h3>
  <p>Los gráficos muestran la variación de la contribución de <code>PAY_0</code>,
  <code>LIMIT_BAL</code> y <code>credit_util</code> según su valor.
  El eje de variables refleja la escala transformada usada por el clasificador.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['12_shap_dependence']}" alt="SHAP Dependence">
    <div class="caption">Figura 12: dependencia de tres variables del modelo seleccionado.</div></div>
  <h3>7.3 Explicación Individual</h3>
  <p>Los gráficos de cascada y fuerza descomponen la predicción de un cliente
  de la muestra. Para este Random Forest, las contribuciones se expresan en la
  escala de salida del estimador mostrada en el gráfico.</p>
  <div class="figure"><img src="data:image/png;base64,{figures['13_shap_waterfall']}" alt="SHAP Waterfall">
    <div class="caption">Figura 13: contribuciones a una predicción individual.</div></div>
  <div class="figure"><img src="data:image/png;base64,{figures['14_shap_force']}" alt="SHAP Force">
    <div class="caption">Figura 14: representación alternativa de la misma predicción.</div></div>
</section>

<!-- 8. CONCLUSIONES -->
<section id="conclusiones">
  <h2>8. Conclusiones y Recomendaciones</h2>
  <h3>Hallazgos Principales</h3>
  <ol style="margin: 12px 0 12px 24px;">
    <li><strong>Se corrigió una fuga de información en CV.</strong> El SMOTE histórico
    se aplicaba antes de crear los folds; ahora solo actúa en el entrenamiento de cada fold.</li>
    <li><strong>{winner_html} fue elegido por CV.</strong> Su AUC medio es
    {winner_result['cv_mean']:.4f} ± {winner_result['cv_std']:.4f}, y en test obtuvo
    {winner_result['test_auc']:.4f}. La brecha es {winner_result['cv_mean'] - winner_result['test_auc']:.4f}.</li>
    <li><strong>Se corrigieron otras fuentes de fuga.</strong> Los percentiles de winsorización
    se aprenden después del split; el escalado se ajusta dentro de CV.</li>
    <li><strong>El historial de pagos domina la explicación del modelo.</strong>
    Los gráficos SHAP corregidos corresponden al ganador y usan su preprocesamiento ajustado.</li>
  </ol>
  <h3>Limitaciones</h3>
  <ul style="margin: 12px 0 12px 24px;">
    <li><strong>Test históricamente reutilizado:</strong> La versión anterior seleccionó
    decisiones mirando el test. Se necesitan datos nuevos para confirmar el rendimiento de forma independiente.</li>
    <li><strong>Atribución de la brecha:</strong> SMOTE, winsorización y escalado se corrigieron
    juntos. Esta ejecución no mide la contribución aislada de cada cambio.</li>
    <li><strong>Datos de 2005:</strong> Su antigüedad limita la extrapolación a clientes actuales.</li>
    <li><strong>Modelo sin validación operativa:</strong> No se evaluaron calibración, estabilidad
    temporal, equidad, costos de decisión ni requisitos aplicables a una entidad concreta.</li>
  </ul>
  <h3>Próximos Pasos</h3>
  <ul style="margin: 12px 0 12px 24px;">
    <li>Confirmar el pipeline y el umbral fijados con datos nuevos no usados en decisiones previas.</li>
    <li>Evaluar calibración de probabilidades y costos de falsos positivos y negativos.</li>
    <li>Comprobar estabilidad por periodo y segmento antes de cualquier uso operativo.</li>
  </ul>
</section>

<footer>
  <p>Proyecto de Riesgo Crediticio &middot; Python 3.12 + scikit-learn + XGBoost + LightGBM + SHAP</p>
  <p>Dataset: <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">UCI Credit Card Default</a>
  &middot; Métricas: <code>models/evaluation_metrics.json</code></p>
</footer>
</div>
</body>
</html>"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Report generated: {OUTPUT}")

if __name__ == "__main__":
    generate_report()
