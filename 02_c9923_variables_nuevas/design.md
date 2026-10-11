# c9923: c9920 + variables nuevas, con bayesiana propia (diseño)

Notebook: `c9923_VariablesNuevas.ipynb`. Pensado para correr de noche en una máquina virtual de Google Cloud de la cátedra, con los resultados en el bucket (`~/buckets/b1/exp/c9923/`).

## Qué es

El modelo de `c9920` (envío `c9920c11000`, 104,75 en el público) con variables nuevas y una optimización bayesiana propia, porque al cambiar las variables el óptimo de hiperparámetros puede moverse. Termina en semillas de agosto listas para el bot y para ensamblar.

## Variables

Todas se calculan con los valores crudos de cada mes y después pasan por el ranking con cero fijo por variable y por mes, como en `c9920`. "Mes anterior" es la observación anterior del cliente, igual que en los lags.

1. **Base:** las 128 de `c9920` (`media_2desvios`), leídas de `exp/c9920/1_seleccion/listas.json`, sin volver a seleccionar.
2. **11 atributos nuevos**, cada uno con lag1, delta_lag1, lag2 y delta_lag2 (55 columnas). Sumas: NA cuenta como 0 (NA si todos son NA). Ratios: NA si el denominador es 0 o NA.

| Variable | Cálculo |
|---|---|
| `mfondos_pesos_total` | mcaja_ahorro + mcuenta_corriente + mplazo_fijo_pesos |
| `ctc_transacciones_total` | ctarjeta_visa_transacciones + ctarjeta_master_transacciones |
| `mtc_saldo_total` | Visa_msaldototal + Master_msaldototal |
| `mmargen_activo_pasivo` | mactivos_margen − mpasivos_margen |
| `mtc_pagominimo_total` | Visa_mpagominimo + Master_mpagominimo |
| `rtc_saldo_limite` | mtc_saldo_total / (Visa_mlimitecompra + Master_mlimitecompra) |
| `rdigital_trx` | (chomebanking_transacciones + cmobile_app_trx) / (las dos + ccallcenter_transacciones + ccajas_transacciones + catm_trx + catm_trx_other) |
| `cproductos_perdidos` | productos activos el mes anterior e inactivos en este; NA sin mes anterior |
| `tperdio_payroll` | 1 si cpayroll_trx anterior > 0 y actual = 0; 0 si los dos existen y no pasa; NA sin mes anterior |
| `tinicio_comision` | 1 si mcomisiones_mantenimiento actual > 0 y anterior ≤ 0; 0 si los dos existen y no pasa; NA sin mes anterior |
| `mpatrimonio_neto` | (mcuentas_saldo + mplazo_fijo_pesos + mplazo_fijo_dolares + minversion1_pesos + minversion1_dolares + minversion2) − (mprestamos_personales + mprestamos_prendarios + mprestamos_hipotecarios + Visa_msaldototal + Master_msaldototal) |

Productos de `cproductos_perdidos` (activo si alguna fuente > 0, inactivo si todas son 0, NA si ninguna es > 0 y alguna es NA): tarjetas (ctarjeta_visa, ctarjeta_master), sueldo (cpayroll_trx, cpayroll2_trx), débitos automáticos (ccuenta_debitos_automaticos, ctarjeta_visa_debitos_automaticos, ctarjeta_master_debitos_automaticos), pagos (cpagodeservicios, cpagomiscuentas), préstamos (cprestamos_personales, cprestamos_prendarios, cprestamos_hipotecarios), inversiones (cplazo_fijo, cinversion1, cinversion2), seguros (cseguro_vida, cseguro_auto, cseguro_vivienda, cseguro_accidentes_personales) y caja de seguridad (ccaja_seguridad).

3. **`_avg3` y `_slope3`** para cada campo base que aparece en las 128 (como original, lag o delta) y para los 11 nuevos: promedio y pendiente de mínimos cuadrados (contra el mes) de los valores disponibles en la observación actual y las dos anteriores, con mínimo 2 valores.

Total: 128 + 55 + 2 × (N + 11) columnas, con N = campos base de la lista (el notebook lo imprime), y el mismo número de canaritos.

## Pasos

1. **Bayesiana:** Optuna TPE multivariado, 25 corridas (1 con los valores de `c9920`, 3.981 / 675 / 0,468, + 10 al azar + 14 guiadas) sobre `num_iterations` (64–4.096), `num_leaves` (16–2.048) y `min_sum_hessian_in_leaf` (1e-6–1). Entrena marzo a mayo, predice junio, semilla 101737, **maximiza el AUC de junio** (BAJA+1 y BAJA+2 positivos), con `metric = "auc"` en LightGBM. Se guardan también la *average precision* (solo BAJA+2) y la ganancia suavizada, como información. Los fijos son los de `c9920` (learning_rate 0,005, feature_fraction 0,5, bynode 0,2, min_data_in_leaf 0, max_bin 31) y canaritos 1 a 1.
2. **Referencia de junio:** los mejores valores con las semillas 101737, 303011 y 602909, comparada con la referencia de junio de `c9920` (mismas semillas) si está en el bucket.
3. **Agosto:** marzo a junio → agosto con los mejores valores de la bayesiana (sin volver a los de `c9920`), hasta 60 semillas: las 10 de `s0004` y después la lista de primos de `s0005`/`s0006`/`c9920`. Después de cada semilla se rehacen los CSV de cortes (9.000 a 15.000 de a 500) y las probabilidades del ensamble.
4. **Apagado:** al terminar, al encontrar `PARAR` o si una celda falla (en la corrida real), borra la máquina. Todo queda en el bucket.

## Cómo se corta

- `touch ~/buckets/b1/exp/c9923/PARAR`, solo cuando ya empezó agosto: antes de cada semilla el notebook lo busca; si está, no empieza otra, deja los cortes al día y apaga. Se borra solo al arrancar una corrida nueva.
- `touch ~/buckets/b1/exp/c9923/CORTAR_BAYESIANA`: termina la corrida de la bayesiana en curso y sigue con junio y agosto con lo mejor que haya. No se borra solo (al retomar no vuelve a la bayesiana); para empezar de cero hay que borrarlo.
- Un error pasajero del bucket al rehacer los cortes o al copiar el estudio queda en el log como aviso y no frena la corrida.
- Si Google borra la máquina spot: otra máquina, el mismo comando. El estudio de Optuna, cada referencia de junio y cada semilla de agosto se guardan en el bucket al terminar; saltea lo hecho.
- Cada predicción guardada lleva una huella del orden de los clientes (marzo a junio en la bayesiana y junio; agosto en cada semilla). Al retomar, una semilla de agosto guardada con otro orden se rehace, y un estudio guardado con otro orden de marzo a junio frena la corrida con un error, en vez de mezclar clientes.

## Pruebas antes de entregarlo

Fuera de Google Cloud, en un entorno aparte con el dataset completo y una lista de 128 variables armada al azar en lugar de la de `c9920` (que en ese momento estaba solo en el bucket):

- Modo prueba de punta a punta (5 % de los clientes), retomando después de matar el proceso en medio de la bayesiana y en medio de agosto (también sin el disco local, como en una máquina nueva), y con `PARAR` y `CORTAR_BAYESIANA`.
- Con los datos completos y la configuración real: datos, atributos nuevos, controles y ranking, más una medición de tiempos de entrenamiento.
