# DMEyF 2026: primera competencia - Mario Freites

Este repo tiene el código y los resultados para reproducir el envío final de la primera competencia: la predicción de BAJA+2 para 202108.

**Envío final:** `c9923_60sem_corte11000.csv`. Tiene los 11.000 clientes de mayor probabilidad promediada sobre 60 semillas. Obtuvo **106,205 en el leaderboard público**.

## El modelo

Son dos experimentos encadenados:

1. **c9920, canaritos asesinos.** Parte de 760 variables: las originales y sus lag1, delta_lag1, lag2 y delta_lag2, con ranking con cero fijo por mes. Elimina las que no le ganan a los canaritos, con el corte en media + 2 desvíos. Quedan 128 variables. También hace una optimización bayesiana sobre esa lista.
2. **c9923, variables nuevas.** Toma esas 128 variables y les suma 11 atributos nuevos, con sus lags y deltas, más promedios y pendientes de 3 meses: 299 variables en total. Hace una bayesiana propia de 25 corridas que maximiza el AUC de junio. Después entrena de marzo a junio con los mejores valores (3304 iteraciones, 1125 hojas, `min_sum_hessian_in_leaf` 0,595) y predice agosto con **60 semillas**.

Los dos usan zLightGBM con canaritos 1 a 1, learning_rate 0,005, feature_fraction 0,5 y max_bin 31. El detalle de cada uno está en su `design.md`.

## Validación

Junio es el mes de validación. c9920 elige la lista de variables con mayo y compara las listas en junio. Las dos bayesianas entrenan con marzo a mayo y miden el AUC sobre junio, con BAJA+1 y BAJA+2 como positivos. El modelo final entrena con marzo a junio y predice agosto. Julio no se usa para entrenar porque todavía no tiene la clase completa.

## Por qué el corte 11.000

c9923 genera un CSV por corte, de 9.000 a 15.000 de a 500. Se mandaron cuatro al bot:

| Corte | Ganancia en junio | Público |
|---|---|---|
| 10.500 | 394,7 | 105,408 |
| **11.000** | **400,9** | **106,205** |
| 11.500 | 399,7 | 103,620 |
| 12.000 | 398,6 | 103,400 |

En junio, la ganancia forma una meseta entre 11.000 y 13.500 (de 398 a 404). 11.000 es el comienzo de esa meseta y el mejor de los cuatro en el público. La ganancia de junio de todos los cortes está en `resultados/c9923/ganancia_junio_por_corte.csv`.

## Estructura

```
01_c9920_canaritos_asesinos/   notebook y diseño de c9920
02_c9923_variables_nuevas/     notebook y diseño de c9923
resultados/c9920/              listas de variables, mejores hiperparámetros y log de la corrida
resultados/c9923/              envío final, bayesiana, ganancia de junio por corte y log de la corrida
```

## Cómo reproducirlo

Los notebooks están pensados para las máquinas virtuales de Google Cloud de la cátedra, con el bucket montado en `~/buckets/b1`. Cada notebook descarga el dataset y compila zLightGBM solo. Los resultados quedan en `~/buckets/b1/exp/<experimento>/`, y al terminar el notebook borra la máquina.

**Máquina:** c9920 corrió con 8 hilos, y c9923 en una máquina de 96 GB de RAM y 12 vCPU. Entre los dos son unas 9,5 horas. Con menos hilos funciona igual, pero tarda más.

```bash
git clone https://github.com/mariofreites/dmeyf-primera-competencia.git ~/dmeyf-primera-competencia

# 1) c9920 (unas 3 horas con 8 hilos)
cd ~/dmeyf-primera-competencia/01_c9920_canaritos_asesinos
nohup jupyter nbconvert --to notebook --execute c9920_CanaritosAsesinos.ipynb \
  --output c9920_corrida.ipynb --ExecutePreprocessor.timeout=-1 > c9920_run.txt 2>&1 &

# 2) c9923 en otra máquina, cuando termine c9920 (unas 6,5 horas con 12 hilos)
cd ~/dmeyf-primera-competencia/02_c9923_variables_nuevas
nohup jupyter nbconvert --to notebook --execute c9923_VariablesNuevas.ipynb \
  --output c9923_corrida.ipynb --ExecutePreprocessor.timeout=-1 > c9923_run.txt 2>&1 &
```

c9923 lee de `exp/c9920/` la lista `media_2desvios` y los mejores hiperparámetros de c9920, que usa como primera corrida de su bayesiana. El envío queda en `~/buckets/b1/exp/c9923/5_cortes/c9923_60sem_corte11000.csv`.

**Atajo:** se puede evitar correr c9920, ya que c9923 puede leer los archivos guardados en `resultados/c9920`:

```bash
DMEYF_C9920_DIR=~/dmeyf-primera-competencia/resultados/c9920 nohup jupyter nbconvert ...   # el mismo comando de c9923
```

En ese caso se saltea la comparación de junio contra c9920, que no influye en el envío.

**Prueba corta:** con `MODO_PRUEBA=1` delante del comando, el notebook corre con el 5 % de los clientes en unos 15 minutos. Escribe en una subcarpeta `prueba/` y no apaga la máquina.

**Para seguir la corrida:** `tail -f ~/buckets/b1/exp/c9923/log.txt`. Si Google borra la máquina spot, alcanza con volver a lanzar el mismo comando en otra: el notebook retoma desde lo que ya estaba guardado en el bucket.

## Notas

- **Nombres.** Estos experimentos corrieron como z9920 y z9923 en el repo de trabajo original. Acá se les cambió la letra a `c` para no confundirlos con los experimentos de la cátedra. El cambio no toca ningún cálculo. Los logs y los CSV de `resultados/` son los archivos originales de las corridas, así que todavía dicen z9920 y z9923. Las referencias a s0004, s0005, s0006 y z9918 son experimentos anteriores de ese repo de trabajo, y z495 es un notebook de la cátedra. Los valores que hacen falta de esos experimentos ya están fijos en el código.
- **Reproducibilidad.** Las semillas están fijas, pero LightGBM con varios hilos puede dar diferencias mínimas entre máquinas, y eso puede mover algo la bayesiana. Para comparar, en `resultados/` están las corridas originales. Se usó Optuna 5.0.0.
