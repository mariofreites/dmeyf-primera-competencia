# c9920: canaritos asesinos (diseño)

Notebook: `c9920_CanaritosAsesinos.ipynb`. Pensado para correr en una máquina virtual de Google Cloud (plantillas spot de la cátedra), con los resultados en el bucket.

## Pregunta

¿Eliminar las variables que no le ganan a los canaritos mejora el modelo de `s0006`?

## Receta del profesor

Agregar un 20 % de canaritos, entrenar un LightGBM razonable, numerar las variables por importancia (1 = la más importante), calcular media y desvío de los números de los canaritos y eliminar toda variable con número mayor que media + desvío o media + 2 desvíos. No se corta en el primer canarito (demasiado agresivo).

## Diseño

1. **Datos:** igual que `s0006` (760 variables reales: originales, lag1, delta_lag1, lag2, delta_lag2, con ranking con cero fijo; préstamos personales incluidos).
2. **Selección:** LightGBM común con los parámetros de z495 (2.848 árboles, 71 hojas, `min_sum_hessian_in_leaf` 0,0067, learning_rate 0,005, `feature_fraction` 0,5, `min_data_in_leaf` 0, `max_bin` 31). 760 variables + 152 canaritos. Entrena con **mayo**: todas las BAJA + 10 % de CONTINUA. Importancia por gain; las variables no usadas no reciben número y mueren. 3 repeticiones con semillas distintas (canaritos y undersampling).
   - Mayo y no marzo + abril: lag2 está vacío en marzo y abril, y las 304 variables de lag2 morirían sin ser evaluadas. Junio queda libre para juzgar.
3. **Listas:** media + 1 desvío y media + 2 desvíos (los dos puntos del profesor). Una variable sobrevive si sobrevive en al menos 2 de 3 repeticiones. Más "sin eliminar".
4. **Comparación:** zLightGBM con los fijos y los mejores valores de `s0006` (1.155 / 126 / 0,094), canaritos 1 a 1, marzo a mayo → junio, 5 semillas por lista, las mismas en todas las listas.
5. **Regla (fijada antes de ver resultados):** gana la lista con mayor mejora media de ganancia de junio (máxima suavizada) frente a "sin eliminar", si es positiva y con t pareado ≥ 2. Si ninguna gana, no hay envío.
6. **Bayesiana de la ganadora:** como `s0006` (21 corridas, TPE, AUC de junio, semilla 101737), marzo a mayo → junio, techo de `min_sum_hessian_in_leaf` en 1.
7. **Referencia de junio:** ganadora optimizada contra "sin eliminar" con los valores de `s0006`, mismas 3 semillas.
8. **Agosto:** ganadora, marzo a junio → agosto, 20 semillas, un CSV por corte (9.000 a 15.000).

## Por qué así

- Lista y semillas separadas de la bayesiana: en `s0006` el AUC casi no cambió con los hiperparámetros (0,909 a 0,913). Si la lista fuera un hiperparámetro más, una lista podría ganar por suerte. Con los mismos valores y las mismas semillas, la única diferencia entre listas son las variables.
- Solo los dos puntos del profesor: menos comparaciones, menos ganadores por azar (lo de `ccomisiones_otras` en z9918), y z9918 no encontró columnas que dañen claramente.

## Google Cloud

- Resultados en `~/buckets/b1/exp/c9920/` (sobreviven si Google borra la máquina spot); el estudio de Optuna en disco local, copiado al bucket después de cada corrida.
- Retoma: cada selección, corrida, prueba de la bayesiana y semilla se guarda al terminar; *Run All* en una máquina nueva saltea lo hecho.
- Al terminar (o si una celda falla, en la corrida real) borra la máquina con `~/machina/direct/apagar_vm.sh` de la cátedra.
- `MODO_PRUEBA=1` (o `MODO_PRUEBA = True` a mano): 5 % de los clientes y pocos árboles, en otra carpeta, sin apagar. Probado de punta a punta fuera de Google Cloud, incluida la recuperación del estudio desde el bucket.
