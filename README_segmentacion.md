# Modelo no supervisado: segmentación de clientes

Dataset `bank_customer_segmentation.csv` (8 950 clientes de tarjeta de crédito, 17 variables de comportamiento).

## Cómo ejecutar
```bash
python -m src.clustering                                              # elige k, challenge, perfilado y registro
python -m src.segmentar data/bank_customer_segmentation.csv outputs/segmentos.csv
```

## Archivos
```
src/
├── features_segmentacion.py  # limpieza basada en el EDA + variables derivadas
├── clustering.py             # flujo de segmentación (editar solo CONFIG)
├── segmentar.py              # asigna segmento con models:/modelo_segmentacion@champion
└── utils/clustering.py       # preprocesamiento, k óptimo, challenge, estabilidad, perfilado, gráficos
```

## Decisiones metodológicas
- **Limpieza:** `CASH_ADVANCE_FREQUENCY` > 1 se recorta a 1 (error de datos); `MINIMUM_PAYMENTS` nulo con `PAYMENTS = 0` se imputa con 0; el resto de nulos, con la mediana de train.
- **Variables derivadas:** utilización del límite (`BALANCE / CREDIT_LIMIT`), ratio pago / pago mínimo y compra promedio por transacción.
- **`TENURE` fuera del clustering** (85 % vale 12), pero se usa para describir los segmentos.
- **Atípicos:** `log1p` en montos y conteos con cola larga, en lugar de eliminar o recortar: conserva el orden de los clientes y reduce la influencia de los extremos en las distancias.
- **Escalado + PCA:** K-Means usa distancias, así que todas las variables se estandarizan. PCA (85 % de varianza) elimina la redundancia de las variables muy correlacionadas.
- **Validación:** split 80/20; la silhouette se mide también en validation (clientes no vistos).
- **Elección de k:** mejor silhouette en validation entre k ≥ 3 con el cluster más chico ≥ 3 %, contrastado con el método del codo.
- **Challenge:** KMeans vs BisectingKMeans vs GaussianMixture con el mismo k; en empate se prefiere KMeans.
- **Estabilidad:** el campeón se reentrena con 5 semillas y se compara con el Adjusted Rand Index.
- **Registro:** preprocesamiento + modelo como un único PipelineModel en MLflow (alias `champion`, tags `algoritmo` y `k`).

## Resultados

**Modelo campeón:** K-Means con k = 6, PCA de 7 componentes (88.3 % de la varianza).

| Métrica | Valor |
|---|---|
| Silueta train | 0.401 |
| Silueta validation | 0.381 |
| Gap train − validation | 0.020 |
| ARI medio entre 5 semillas | 0.795 (mínimo 0.581) |
| Cluster más chico / más grande | 11.7 % / 22.6 % |

**Elección de k:** k = 6 es el pico local de silueta en validation y el punto donde el codo se aplana (WSSSE de 85 238 con k = 2 a 48 497 con k = 6; solo 2 100 menos con k = 7). Se descartó k = 2 (silueta 0.428) porque solo separa compradores de no compradores, y k = 9 y 10 por tener clusters de menos del 3 %.

**Challenge (k = 6):**

| Algoritmo | Silueta validation | Cluster más chico |
|---|---|---|
| **K-Means** | **0.381** | 11.7 % |
| Bisecting K-Means | 0.283 | 9.7 % |
| Gaussian Mixture | 0.202 | 2.7 % |

## Segmentos

| Cluster | % clientes | Nombre | Rasgos principales (índice vs promedio) | Acción comercial |
|---|---|---|---|---|
| 4 | 16.1 % | **Premium / grandes compradores** | Compras ×3.4 (de un solo pago ×3.9), límite ×1.6, paga el total ×2.3, casi sin avances | Fidelización, beneficios exclusivos, aumento de cupo |
| 1 | 18.6 % | **Compradores a cuotas cumplidos** | Frecuencia de compra a cuotas ×2.2, paga el total ×2.1, sin avances, saldo bajo | Promociones de cuotas sin interés; segmento de bajo riesgo |
| 3 | 15.7 % | **Compradores de ticket alto que financian** | Compra promedio ×2.0, utilización ×1.4, casi nunca paga el total (×0.1) | Ofertas de diferido; monitorear el nivel de endeudamiento |
| 0 | 15.2 % | **Revolventes con avances** | Avances en efectivo ×3.0, saldo ×2.2, utilización ×1.6, paga el total ×0.2 | Mayor riesgo: seguimiento de cobranza y límites a los avances |
| 2 | 22.2 % | **Usuarios de efectivo** | Solo usa avances (×2.1), compras ≈ 0, utilización ×1.6 | Ofrecer préstamo personal (más barato que el avance) e incentivar compras |
| 5 | 12.1 % | **Inactivos** | Saldo ×0.06, actualización de saldo ×0.44, utilización ×0.08 | Campañas de reactivación |

**Lectura de riesgo:** los segmentos *Revolventes con avances* y *Usuarios de efectivo* (37 % de los clientes) concentran la utilización alta del cupo y casi nunca pagan el total. *Premium* y *Compradores a cuotas cumplidos* (35 %) pagan el total más del doble que el promedio y son los más sanos.

**Limitaciones:**
- Una silueta de 0.38 indica separación moderada: el comportamiento de los clientes es un continuo, y hay clientes de frontera entre segmentos parecidos (lo confirma el ARI mínimo de 0.58).
- El `RATIO_PAGO_MINIMO` ×5.2 de los *Inactivos* se debe a que su pago mínimo es casi 0, no a que paguen mucho: los ratios se leen junto a sus componentes.
- Los números de cluster pueden cambiar al reentrenar; los nombres se asignan según el perfil, no según el número.