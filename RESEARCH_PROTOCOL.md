# Research Protocol

> Protocolo obligatorio para cada senal que entre al sistema.
> En quant, media ventaja competitiva es simplemente no engañarse.

---

## Regla fundamental

**Ninguna senal entra al composite score sin pasar por este protocolo completo.**

No importa que tan intuitiva sea, que tan bonita se vea en un grafico, o que tan bien funcione "eyeballing" unos cuantos ejemplos. Si no pasa el protocolo con metricas cuantificables, no existe para el sistema.

---

## Los 10 pasos del protocolo

### Paso 1 — Hipotesis economica

Antes de tocar datos, escribir en texto claro:

```
¿Por que esta senal deberia predecir retornos futuros?
¿Cual es el mecanismo economico o conductual?
¿Hay literatura academica que la respalde?
¿Por que el mercado no la habria arbitrado ya?
```

**Ejemplo bueno**:
> "Earnings surprise positivo predice retornos futuros porque el mercado sub-reacciona a nueva informacion de earnings. Documentado como PEAD (Post-Earnings Announcement Drift) por Bernard & Thomas (1989). Persiste porque el ajuste de expectativas de analistas es lento."

**Ejemplo malo**:
> "Probe muchas combinaciones de indicadores y esta funciona."

Si no hay hipotesis economica coherente, la senal probablemente es data mining. Matar antes de empezar.

---

### Paso 2 — Definicion exacta

Documentar con precision milimetrica:

```
Nombre:              earnings_surprise
Formula:             (EPS_actual - EPS_estimado_consenso) / abs(EPS_estimado_consenso)
Fuente de datos:     FMP /earnings-surprises/{ticker}
Frecuencia:          Trimestral (actualiza 4x al año por ticker)
Timestamp real:      fecha de reporte de earnings (earnings_date)
Universo aplicable:  Acciones con cobertura de analistas (>3 estimados)
Normalizacion:       Cross-sectional z-score dentro del universo por fecha
Rango output:        [-1, +1] via winsorization al percentil 5/95
```

**Regla**: otra persona debe poder reproducir la senal exactamente a partir de esta definicion.

---

### Paso 3 — Timestamp de disponibilidad

Para cada dato que alimenta la senal, registrar:

| Dato | Period date | Filing/availability date | Lag tipico |
|------|-----------|------------------------|-----------|
| Income statement Q1 | 2026-03-31 | 2026-04-28 | ~4 semanas |
| Earnings surprise | 2026-04-28 | 2026-04-28 | Mismo dia |
| Precio cierre | 2026-04-28 | 2026-04-28 | Mismo dia (EOD) |
| Analyst estimate | N/A | 2026-04-15 | Publicacion |

**Regla**: en backtesting, la senal de fecha T solo puede usar datos con `availability_date <= T`. Violar esto es look-ahead bias y invalida todo resultado.

---

### Paso 4 — Horizonte objetivo

Definir explicitamente:

```
Horizonte de evaluacion:  21 dias habiles (1 mes)
Horizonte alternativo:    63 dias habiles (3 meses)
Justificacion:            PEAD se disipa en ~60 dias segun literatura
```

**Regla**: evaluar la senal en el horizonte correcto. Una senal de valor (que opera en 6-12 meses) evaluada a 5 dias va a parecer que no funciona — no porque sea mala, sino porque esta en el horizonte equivocado.

---

### Paso 5 — Split temporal

```
Periodo total disponible:   2019-01-01 a 2026-03-24 (~7 años)

Train period:               2019-01-01 a 2023-12-31 (5 años)
  → para calibrar normalizacion, thresholds, entender la senal

Test period (OOS):          2024-01-01 a 2026-03-24 (2+ años)
  → para evaluar. NO se toca hasta que la senal esta finalizada.

Walk-forward (alternativa): ventanas rolling de 3 años train + 1 año test
```

**Regla**: el periodo de test OOS se usa UNA VEZ por senal. Si ajustas la senal y re-evaluas en OOS, ya no es out-of-sample. Si necesitas iterar, usa cross-validation temporal dentro del train period.

---

### Paso 6 — Metrica primaria: Information Coefficient (IC)

```python
IC = spearman_rank_correlation(signal_t, forward_return_{t, t+h})
```

Evaluacion cross-sectional: en cada fecha t, calcular correlacion de ranking entre senal y retorno futuro a traves de todas las acciones del universo. Promediar IC a traves de todas las fechas.

**Thresholds**:

| IC promedio | Interpretacion |
|------------|----------------|
| < 0.02 | Sin senal. Descartar. |
| 0.02 - 0.03 | Muy debil. Posiblemente util en composite grande. |
| 0.03 - 0.05 | Senal util. Candidata para composite. |
| 0.05 - 0.10 | Senal buena. Contribucion significativa. |
| > 0.10 | Senal excelente. Raro — verificar que no hay leakage. |

**Tambien reportar**:
- IC por año (¿estable o variable?)
- IC en bull vs bear markets
- IC en high vol vs low vol
- t-stat del IC (IC / std(IC_mensual) * sqrt(N_meses))

---

### Paso 7 — Analisis por deciles/quintiles

Dividir el universo en 5 grupos por score de la senal en cada fecha. Calcular retorno promedio de cada grupo a horizonte h.

```
Quintil 1 (peor score):    retorno promedio 21d = +0.2%
Quintil 2:                 retorno promedio 21d = +0.5%
Quintil 3:                 retorno promedio 21d = +0.7%
Quintil 4:                 retorno promedio 21d = +1.1%
Quintil 5 (mejor score):   retorno promedio 21d = +1.8%

Spread Q5 - Q1:            +1.6% mensual = +19.2% annualizado
Monotonia:                  5/5 quintiles ordenados ✓
```

**Lo que buscamos**:
- Spread Q5-Q1 positivo y significativo
- Relacion monotonica (cada quintil mejor que el anterior)
- Si solo Q5 funciona pero Q1-Q4 son similares, la senal es fragil

---

### Paso 8 — Factor attribution

¿Cuanto del IC es alpha real vs. exposicion a factores conocidos?

```
Regresion Fama-French-Carhart:
  r_i - r_f = α + β_mkt·(r_mkt - r_f) + β_smb·SMB + β_hml·HML + β_mom·MOM + ε

Medir:
  IC total de la senal:                    0.062
  IC despues de neutralizar market beta:   0.055
  IC despues de neutralizar mom:           0.041
  IC despues de neutralizar todos:         0.028 ← IC residual

  Si IC residual > 0.02 → la senal aporta algo genuinamente nuevo
  Si IC residual ≈ 0   → la senal es solo factor exposure disfrazado
```

**Factores a controlar**:

| Factor | Proxy | Por que controlarlo |
|--------|-------|---------------------|
| Market | retorno SPY | ¿Es solo beta? |
| Size | market cap | ¿Favorece small caps? |
| Value | P/B o P/E inverso | ¿Es solo value disfrazado? |
| Momentum | retorno 12-1m | ¿Es solo momentum? |
| Quality | ROE, leverage | ¿Es solo quality? |
| Low vol | volatilidad historica | ¿Es solo low-beta? |

---

### Paso 9 — Turnover y costos implicitos

```
Turnover mensual de la senal:
  = promedio de cambio en posiciones del top quintil mes a mes

  Turnover < 30% mensual → bajo costo, bueno
  Turnover 30-60% → aceptable con spreads bajos
  Turnover > 60% → cuidado, costos pueden destruir alpha

Costo estimado:
  costo_annual = turnover_annual * 2 * cost_per_trade
  donde cost_per_trade ≈ 10-15 bps (spread + slippage para acciones liquidas US)

Alpha neto:
  alpha_neto = alpha_bruto - costo_annual
  Si alpha_neto ≤ 0, la senal no es invertible aunque tenga IC positivo
```

---

### Paso 10 — Veredicto final

Formato estandarizado para cada senal evaluada:

```
═══════════════════════════════════════════════════
SIGNAL EVALUATION REPORT
═══════════════════════════════════════════════════

Nombre:               earnings_surprise
Hipotesis:            PEAD — sub-reaccion del mercado a earnings
Horizonte evaluado:   21 dias
Periodo OOS:          2024-01 a 2026-03

METRICAS:
  IC promedio (OOS):          0.062   ✓ (> 0.03)
  IC t-stat:                  2.84    ✓ (> 2.0)
  IC bull market:             0.058
  IC bear market:             0.071
  Spread Q5-Q1 (ann.):       +12.4%  ✓
  Monotonia deciles:          4/5     ✓
  IC residual (post-factors): 0.028   ✓ (> 0.02)
  Turnover mensual:           22%     ✓ (< 30%)
  Alpha neto estimado:        +7.1%   ✓ (> 0)

VEREDICTO:  ✅ PASA → entra al composite score

Peso sugerido en composite: proporcional a IC residual
Notas: senal mas fuerte post-earnings (primeras 2 semanas).
       Considerar decay function.
═══════════════════════════════════════════════════
```

Si la senal no pasa, el reporte documenta por que y se archiva. **Las senales muertas tambien se documentan** — para no reinventarlas despues.

---

## Reglas adicionales

### Multiple testing

Si evaluas 20 senales, algunas van a parecer buenas por azar. Ajustar por multiple testing:

- **Bonferroni conservador**: p-value ajustado = p * N_senales_testeadas
- **FDR (Benjamini-Hochberg)**: menos conservador, controla tasa de falsos descubrimientos
- **Regla practica**: si el t-stat del IC no es > 3.0 despues de testear 20+ senales, ser esceptico

### Regla anti-snooping

Llevar un registro cronologico de:
1. Fecha en que se definio la senal
2. Fecha en que se evaluo en OOS
3. Numero total de senales evaluadas hasta esa fecha
4. Cuantas pasaron vs cuantas se descartaron

Si el hit rate es > 50%, probablemente hay snooping inconsciente.

### Estabilidad minima

Una senal con IC = 0.08 promedio pero que oscila entre -0.05 y +0.20 por subperiodo es peligrosa. Preferir senales con IC estable (std del IC mensual baja) sobre senales con IC alto pero erratico.

### Regla de muerte

Una senal que pasa el protocolo puede morir despues si:
- IC cae a < 0.01 por 6+ meses consecutivos en produccion
- El mecanismo economico subyacente cambia (ej: regulacion nueva)
- Se publica ampliamente y se arbitra (capacity decay)

Monitorear IC de cada senal activa mensualmente.

---

## Resumen ejecutivo

```
Antes de tocar datos:    hipotesis economica clara
Definicion:              exacta, reproducible, con timestamps
Split temporal:          train/test estricto, OOS se toca UNA VEZ
IC:                      > 0.03 para pasar
Deciles:                 spread monotono Q5 > Q1
Factor attribution:      IC residual > 0.02 despues de controlar factores
Turnover:                alpha neto > 0 despues de costos
Estabilidad:             IC positivo en multiples subperiodos y regimenes
Multiple testing:        ajustar por numero de senales evaluadas

Si pasa todo:            ✅ entra al composite con peso proporcional a IC residual
Si falla cualquiera:     ❌ se documenta y se archiva. No se fuerza.
```
