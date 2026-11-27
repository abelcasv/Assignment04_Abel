# Assignment 4 — Protocolo experimental definitivo (Fase 3)

**Curso:** BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC
**Curso académico:** 2026–2027
**Fecha de versión original (Fase 3 / Gate G4):** 13 de noviembre de 2026
**Última actualización cronológica de estado:** 26 de noviembre de 2026
**Estado actual:** **Protocolo experimental ejecutado en ámbito local autorizado.**
- *Trazabilidad cronológica:* (1) El 13-11-2026 (Fase 3 / Gate G4) este documento se aprobó como diseño congelado previo a compilación y mediciones; (2) el 14-11-2026 (Fases 5–8) se validaron compilación/corrección y se ejecutaron los pilotos y la campaña base Dask de un worker (`p = 1`, 48 filas preservadas en `results/raw/raw_measurements.csv`); (3) el 23-11-2026, tras autorización expresa docente (`docs/decisions/DEC-01-instructor-permission.md`), se ejecutaron la campaña local multi-worker de strong scaling (`p ∈ {1, 2, 4, 8, 10}`, 90 filas anexadas) y el pase diagnóstico separado (`n = 1000`); (4) el 26-11-2026 se ejecutaron las comparaciones controladas de los baselines congelados A1 y A3 sobre las matrices comunes deterministas `data/dense/` (`results/raw/a1_common_input_raw.csv`, `results/raw/a3_common_input_raw.csv`). La ejecución multinodo física permanece **BLOCKED / NOT AVAILABLE** y el weak scaling permanece **DEFERRED (DEC-07)**.

---

## 0. Estado, jerarquía por dominios y alcance

Este documento constituye la **única fuente canónica y detallada de verdad (*single detailed source of truth*)** para el protocolo experimental de Assignment 4, concretando parámetros, configuraciones, particionado, contratos temporales, métricas, tolerancias numéricas y presupuestos de recursos. Cualquier versión o borrador previo (incluyendo `docs/Assignment04_Experimental_Protocol_Final.md`) queda plenamente superado por este archivo y no debe ser tratado como una fuente competidora de verdad.

### Modelo de autoridad por dominios

Para evitar ambigüedades en las que las reglas de ingeniería parezcan anular las definiciones experimentales, se aplica un **modelo de autoridad específico por dominios**:

1. **Documento oficial de la asignatura (`BD_Individual_Assignments_2026_2027.pdf`) y aclaraciones expresas de la profesora:** Máxima autoridad académica; gobierna los objetivos de aprendizaje, requisitos de evaluación y limitaciones administrativas (incluyendo la restricción de ejecución local con múltiples workers).
2. **`SPEC.md`:** Especificación técnica suplementaria redactada por el estudiante; gobierna la arquitectura del proyecto, contratos técnicos, interfaces de baselines y puertas de verificación, subordinada siempre al PDF oficial.
3. **`docs/experimental_protocol.md` (este documento):** Gobierna los parámetros experimentales detallados, tamaños de matriz, particionado por bloques, contratos temporales de medición, métricas de telemetría, tolerancias numéricas y presupuestos de recursos, dentro de los límites del PDF oficial.
4. **Instrucciones de fase del estudiante:** Autorizan únicamente las acciones operativas y entregables explícitamente asignados a cada fase, sin anular ni debilitar el PDF oficial.

**Regla ante contradicciones:** Si surge un conflicto genuino entre documentos, el agente debe detenerse e informar al estudiante con citas exactas de los pasajes contradictorios antes de continuar, sin seleccionar interpretaciones de conveniencia.

### Estado de las decisiones

| Elemento | Estado |
|---|---|
| Dask Distributed como framework principal | Decisión arquitectónica aceptada por el estudiante (ADR-001). |
| Multiplicación por bloques 2D y reducción en árbol | Arquitectura aprobada como diseño de referencia; su implementación se realizará en Phase 4. |
| Topología local `LocalCluster`, procesos y un hilo por worker | Topología de referencia aprobada para el entorno conocido. |
| `p=1` en una máquina | Configuración de diseño disponible; el runtime real se verificará en la fase de entorno (Gate G7). |
| `p>1` en una máquina local | **AUTORIZADA (2026-11-23, DEC-01)** *(históricamente `BLOCKED` en Phases 1–10)* mediante `docs/decisions/DEC-01-instructor-permission.md`; exclusivamente 1 máquina física (`physical_machine_count=1`). |
| Multinodo real | No hay infraestructura multinodo acreditada actualmente; **BLOCKED / NOT AVAILABLE** hasta identificarla y aprobarla. |
| Strong scaling obligatorio | **COMPLETADO en modo local multiproceso (1 máquina física, 2026-11-23)** bajo DEC-01 (`p ∈ {1, 2, 4, 8}` completos; `p=10` detenido por timeout `131.92 s > 120 s` en `n=1500` rep 5). Escalado multinodo permanece **BLOCKED / NOT AVAILABLE**. |
| Inputs, A1/A3 y sus snapshots | Congelados en Phase 2.1; inmutables; no regenerar ni modificar. |
| Weak scaling | Extensión opcional aplazada; **DEFERRED / BLOCKED (DEC-07)**; no forma parte de la campaña ejecutada. |
| Implementación, compilación y mediciones autorizadas | **COMPLETADAS en fases posteriores autorizadas** (Dask `p=1` el 14-11-2026; strong scaling local `p ∈ {1..10}` y diagnósticos el 23-11-2026; comparación controlada A1/A3 con inputs comunes `data/dense/` el 26-11-2026; *no estaban autorizadas aún en la aprobación inicial de diseño del 13-11-2026*). |

> [!IMPORTANT]
> **Cumplimiento estricto del requisito de strong scaling:**
> El documento oficial de la asignatura exige un estudio de *strong scaling*. La ejecución con `p=1` por sí sola **NO constituye un barrido de strong scaling**, pues únicamente mide la sobrecarga del framework con un worker individual.
> Si la profesora no autoriza `p>1` local y no se habilita infraestructura multinodo externa, el estudio obligatorio de strong scaling quedará formalmente registrado como **BLOCKED / NOT COMPLETED**. Bajo ninguna circunstancia se pretenderá que el requisito de strong scaling ha sido satisfecho ejecutando `p=1` y comparándolo con las implementaciones secuenciales o de OpenMP; esa comparación es un estudio contextual entre sistemas con límites temporales distintos, no escalabilidad de workers. Se registra formalmente el bloqueo administrativo (DEC-01) y la necesidad imperativa de aclaración docente o de una alternativa autorizada.

---

## 1. Pregunta de investigación e hipótesis

### 1.1 Pregunta central

¿Cómo cambia el tiempo de multiplicación end-to-end, el comportamiento de memoria y el movimiento de datos de una implementación de multiplicación densa por bloques 2D con Dask Distributed al aumentar los workers disponibles, y cómo se sitúan esos tiempos frente a los kernels secuenciales y OpenMP de A1/A3 cuando procesan las mismas matrices lógicas y el mismo tipo numérico?

El estudio separará tres efectos que no deben confundirse:

- rendimiento del kernel local;
- coste del framework y de la descomposición/distribución/recolección;
- efecto de aumentar workers dentro de una topología física concreta.

### 1.2 Hipótesis (predicciones, no resultados)

- **H1 — Sobrecarga en tamaños pequeños.** Con pocas tareas y poco trabajo por tarea, la planificación, serialización, transferencia y recolección de Dask pueden representar una parte importante del tiempo end-to-end y hacer que Dask sea más lento que los kernels nativos.
- **H2 — Escalabilidad condicionada por granularidad.** Incrementar workers puede reducir el tiempo end-to-end cuando hay suficientes tareas independientes y trabajo útil por tarea, pero no se presupone que la ganancia aparezca ni que sea proporcional.
- **H3 — Rendimiento no monótono.** Más workers pueden dejar de mejorar o empeorar el tiempo debido a planificación, comunicación, contención de memoria, desequilibrio de tareas y heterogeneidad de CPU.
- **H4 — Comparación de sistemas, no de framework aislado.** Dask/NumPy/BLAS y C/OpenMP usan backends y estrategias de cómputo distintas. La diferencia observada no permite atribuir causalmente toda la brecha al framework.

No se utilizará la palabra «significativo» en sentido estadístico salvo que se defina y aplique posteriormente una prueba estadística apropiada. Mediana e IQR describen centro y dispersión, pero no prueban significación.

---

## 2. Métodos, procedencia y política de inputs

### 2.1 Métodos

| Etiqueta | Origen | Algoritmo / backend | Función en el estudio |
|---|---|---|---|
| `seq_basic` | A1 | Triple bucle `i-j-k`, C, `double`, orden original | Baseline secuencial básico requerido. |
| `seq_ikj` | A3 | Bucle `i-k-j`, C, `double` | Baseline secuencial optimizado; referencia emparejada de OpenMP. |
| `omp_ikj` | A3 | Bucle `i-k-j`, OpenMP `schedule(static)` | Baseline paralelo principal. |
| `omp_tiled_b32` | A3 | Variante OpenMP tiled con bloque 32×32 | Método secundario; no sustituye a `omp_ikj`. |
| `dask_block_2d` | A4 | Bloques 2D; `numpy.matmul`; reducción por árbol binario | Implementación distribuida objeto del estudio. |

Para Dask se mantendrá un hilo por worker y se limitará a un hilo el backend numérico interno. Se registrarán las variables de entorno y el backend BLAS realmente detectado. No se asumirá que `OMP_NUM_THREADS=1` por sí solo controla todos los backends; se registrarán también, cuando correspondan, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `NUMEXPR_NUM_THREADS`. En Phase 6 se verificó dinámicamente mediante `threadpoolctl` que el worker cargó `libscipy_openblas64` (v0.3.34.106.0) con `num_threads=1` y `nthreads=1` (con variables de entorno heredadas como 1 antes de lanzar el worker; sin biblioteca OpenMP cargada en el worker).

Para la comparación principal se propone `OMP_NUM_THREADS=10` en `omp_ikj` y `omp_tiled_b32`, con `OMP_DYNAMIC=FALSE` si el runtime lo admite; el número real de hilos se verificará y registrará. `seq_ikj` permanece secuencial. La comparación con OpenMP no se presentará como equivalente en recursos a Dask: un proceso con diez hilos y diez procesos Dask son topologías distintas.

### 2.2 Procedencia congelada

- A1: commit entregado `90681e85949b6d65bdbcfef702b6306b0003f971`.
- A3: commit entregado `23cffbc82c155b6c3de24b3c149b509b63fc870b`.
- Snapshots locales: `baselines/a1_sequential/` y `baselines/a3_openmp/`.
- Los snapshots son inmutables. No se cambiarán algoritmos, flags ni fuentes.
- Los ejecutables futuros irán exclusivamente a `build/bin/`, nunca dentro de los snapshots.

Hashes reportados en Phase 2.1:

| Archivo congelado | SHA-256 |
|---|---|
| A1 `benchmark.c` | `4893fc553e97a0ec596c3c1cf9d5d8f582fc49308fc5971b338f48fd0268236f` |
| A1 `Makefile` | `b35b8cf8671d4cb0bdc42b59f0708dd8e55ebe51842034d3046c60da82f8c31f` |
| A3 `parallel_benchmark.c` | `6da20a3aed0862adc296fe9f4c8625bc51cd777959a5403088f7960c8741dbc8` |
| A3 `Makefile` | `474e4d2f2b57a5696bd8202ac0e125b101dbe6bcd5f58bd92df2a23f8a37e364` |

Los hashes y commits anteriores se conservan como metadatos de procedencia; no autorizan a alterar los archivos. Los repositorios externos originales permanecen intactos.

### 2.3 Inputs comunes y tamaños

Se utilizarán los inputs ya congelados en `data/dense/`, identificados en `data/SHA256SUMS.txt`. Según Phase 2.1:

- generador: `scripts/generate_dataset.py`;
- semilla independiente por dimensión: `42+n`;
- generador pseudoaleatorio: `random.Random` de la biblioteca estándar;
- valores `float64` en `[0,1)`;
- generación de A y después B, en orden fila-columna;
- escritura de matrices con 17 cifras significativas y finales de línea LF, según el contrato del dataset.

**No se volverán a generar estos archivos ni se modificarán.** Antes de cada campaña se verificarán sus hashes mediante el verificador existente; dicha verificación requerirá autorización de la fase correspondiente.

**Política de almacenamiento y reconstrucción (Opción B):** De conformidad con la decisión de almacenamiento autorizada, las matrices densas generadas en `data/dense/` (~310 MB) se excluyen de Git mediante `.gitignore` (`/data/dense/`) para evitar el engrosamiento del repositorio y advertencias de GitHub por archivos superiores a 50 MB. Su generador determinista (`scripts/generate_dataset.py`) y el manifiesto criptográfico (`data/SHA256SUMS.txt`) están versionados en Git. En un clon limpio, las matrices se materializan y certifican mediante `python scripts/generate_dataset.py` y `python scripts/verify_dataset.py data/SHA256SUMS.txt` (ver guía operativa en `data/README.md`). En un espacio de trabajo con datos ya materializados, las matrices permanecen estrictamente inmutables y no deben regenerarse ni sobrescribirse.

| Uso | Dimensiones | Métodos |
|---|---|---|
| Auxiliar exclusivo de A1 | `n=100` | A1 solamente; no entra en comparaciones de tres vías. |
| Comparación común | `n={250,500,750,1000}` | A1, A3 y Dask. |
| Estudio ampliado | `n={1500,2000}` | A3 y Dask; A1 excluido por el límite fijo del baseline original. |

La comparación se hará con los mismos archivos A/B para una misma dimensión. No se afirmará que los datos de A1 y A3 históricos son iguales; se utilizará exclusivamente el dataset común congelado de A4.

### 2.4 Preparación y rutas de los baselines

Los comandos se deberán ejecutar desde la raíz de `Assignment04`. Los ejecutables se crearán en `build/bin/`; los datos runtime estarán en:

- A1: `build/a1_runtime/data/`;
- A3: `build/a3_runtime/data/dense/` y `build/a3_runtime/data/validation/`.

Comandos de ejecución previstos desde la raíz:

```powershell
.\build\bin\a1_benchmark.exe build/a1_runtime
.\build\bin\a3_parallel_benchmark.exe build/a3_runtime
```

La compilación se hará en una fase posterior, con salida a `build/bin/`. Como candidato de compilación A1 se conserva la invocación directa a `gcc` sobre `baselines/a1_sequential/benchmark.c`, con los flags originales `-Wall -Wextra -O3 -std=c11` y `-lpsapi`. Para A3 se conservarán los flags originales `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native` y la definición `BUILD_FLAGS` que necesita el harness.

**Asignación de fases para compilación e inspección:**
Phase 4 permanece estrictamente acotada a la implementación del código de Assignment 4 y a la revisión estática de Makefiles, includes, rutas relativas, macro `BUILD_FLAGS` y adaptadores runtime, sin ejecutar comandos de compilación ni benchmarks.

La compilación material de los baselines y la inspección del ensamblado de A1 se asignan explícitamente a **Phase 5 (validación de compilación y corrección)**, antes de ejecutar los casos de prueba de corrección. Dicha fase autorizará expresamente:
- La compilación del snapshot inmutable de A1 con sus flags originales (`-Wall -Wextra -O3 -std=c11 -lpsapi`) con salida a `build/bin/a1_benchmark.exe`.
- La inspección de la salida del compilador / código objeto mediante `gcc -S` o `objdump -d` para verificar si el cálculo completo sobrevive a `-O3`.
- La compilación del snapshot inmutable de A3 con sus flags originales hacia `build/bin/a3_parallel_benchmark.exe`.
- Ninguna modificación de código fuente ni de flags de compilación de los baselines.
- Ninguna ejecución de benchmarks de rendimiento.

### 2.5 Riesgo específico de A1: consumo de `c[0]`

El harness A1 consume únicamente `c[0]`. Bajo `-O3`, existe el riesgo de que el compilador elimine cálculos de elementos de C que nunca son observados.

En **Phase 5**, previa compilación autorizada, se realizará una inspección del ensamblado o desensamblado generado para el ejecutable A1 (`gcc -S` o `objdump -d`) con los flags originales. Se buscará evidencia fehaciente de que el cálculo completo del producto se conserva en código máquina, no solo de que existe un bucle o una escritura en `c[0]`.

- No se cambiará el código A1.
- No se cambiarán los flags de compilación originales.
- No se añadirá un sink artificial en el código de baseline.
- Si el cálculo completo no se conserva bajo `-O3`, A1 no se presentará como una medición fiable de multiplicación completa; la limitación se explicará explícitamente en el informe y el tiempo se mantendrá solo como contexto del harness.
- No se afirmará que la compilación o el desensamblado fueron ejecutados hasta que Phase 5 los realice efectivamente. Se mantendrá estricta separación entre inspeccionar el código máquina compilado y ejecutar benchmarks.

---

## 3. Topología y diseño de strong scaling

### 3.1 Configuraciones

La máquina conocida es un único host Windows 11 con Intel Core i5-1335U, 10 núcleos físicos y 12 procesadores lógicos, y 16 GB de RAM. El procesador es heterogéneo; no se supondrá que los workers quedan asignados a tipos concretos de núcleo.

| Configuración | Workers solicitados | Máquinas físicas | Estado |
|---|---:|---:|---|
| Local `p=1` | 1 | 1 | Autorizada y ejecutada (Phase 8 y referencia base de strong scaling). |
| Local `p=2` | 2 | 1 | **AUTORIZADA (2026-11-23, DEC-01)** *(antes BLOCKED en Phases 1–10)* — 1 máquina física, sockets loopback. |
| Local `p=4` | 4 | 1 | **AUTORIZADA (2026-11-23, DEC-01)** *(antes BLOCKED en Phases 1–10)* — 1 máquina física (P+E cores). |
| Local `p=8` | 8 | 1 | **AUTORIZADA (2026-11-23, DEC-01)** *(antes BLOCKED en Phases 1–10)* — 1 máquina física, bajo watchdog RSS/RAM. |
| Local `p=10` | 10 | 1 | **AUTORIZADA (2026-11-23, DEC-01)** *(antes BLOCKED en Phases 1–10)* — 10 núcleos físicos; bajo watchdog RSS/RAM, sin denominarla «saturación» sin evidencia. |
| Multinodo | Por determinar | 2 o más | **BLOCKED — no hay clúster físico/cloud acreditado.** Requiere una decisión de topología separada. |

En Dask se propone `LocalCluster(n_workers=p, threads_per_worker=1, processes=True)`. En Windows, el punto de entrada que cree el cluster deberá estar protegido por `if __name__ == "__main__":` debido al modelo de creación de procesos.

`p` significa número de workers Dask, no número total de procesos del sistema. Driver, scheduler, nanny y worker se contabilizarán mediante los PIDs observados; no se fijará un número teórico de procesos en la tabla de resultados.

#### Distinción conceptual de roles y procesos en LocalCluster:
- **Roles lógicos de Dask:** Client, Scheduler, Nanny y Worker son roles lógicos y de ejecución diferenciados; un rol lógico no implica necesariamente un PID independiente en el sistema operativo.
- **Topología observada de `LocalCluster` en Windows:** Para la configuración exacta evaluada en Phase 6 (Dask Distributed 2026.8.0, `LocalCluster(processes=True, n_workers=1, threads_per_worker=1)` sobre Windows 11), el proceso Driver aloja el Client, el Scheduler y la instancia de gestión Nanny de LocalCluster, mientras que el Worker 0 se ejecuta en un proceso hijo independiente del sistema operativo. Se observaron exactamente 2 PIDs únicos para $p=1$. Este es un resultado de topología empíricamente observado para este entorno y modo de arranque concreto, no una ley universal aplicable a todo despliegue, plataforma o versión de Dask.
- **Despliegues CLI independientes:** En despliegues mediante herramientas CLI independientes (`dask scheduler`, `dask worker`), los procesos pueden diferir, pero no se asume ninguna fórmula teórica fija (como el nominal $1+2p$) sin verificación separada; no se sustituye una fórmula fija no demostrada por otra.
- **Regla contable obligatoria:** El proyecto registra los PIDs reales únicos del sistema operativo observados en cada ejecución y mapea los roles lógicos a dichos PIDs. En despliegues in-process de `LocalCluster`, Scheduler y Nanny pueden compartir PID con el Driver. Para configuraciones de workers aún no autorizadas o ejecutadas, no se proyecta un recuento de procesos no verificado.

### 3.2 Strong scaling principal

Se utilizarán:

- tamaños principales: `n={1000,1500,2000}`;
- tamaño de bloque fijo: `b=250`;
- configuración de workers: `p={1,2,4,8,10}` solo cuando esté autorizada;
- inputs, `float64`, algoritmo, backend, número de hilos por worker y límite temporal constantes dentro de cada barrido.

No se variarán simultáneamente `n` y `b` durante un barrido. Para cada dimensión habrá una curva separada.

Con `b=250`, el número de tareas lógicas previstas es:

| `n` | Bloques por eje `Nb` | Multiplicaciones `Nb³` | Reducciones `Nb²(Nb−1)` | Total lógico |
|---:|---:|---:|---:|---:|
| 250 | 1 | 1 | 0 | 1 |
| 500 | 2 | 8 | 4 | 12 |
| 750 | 3 | 27 | 18 | 45 |
| 1000 | 4 | 64 | 48 | 112 |
| 1500 | 6 | 216 | 180 | 396 |
| 2000 | 8 | 512 | 448 | 960 |

Estas cantidades son tareas algebraicas previstas por el algoritmo, no un recuento garantizado de todas las operaciones internas del scheduler.

`n=250` y `n=500` se mantienen para comparar métodos y observar overhead, pero no serán la evidencia principal de strong scaling: con `b=250` producen una y ocho tareas de multiplicación, respectivamente, y ofrecen paralelismo limitado.

### 3.3 Interpretación por topología y cumplimiento de strong scaling

Se establecen distinciones terminológicas y metodológicas estrictas entre tres regímenes:

1. **Dask con un worker (`p=1`):** Mide el comportamiento y la latencia end-to-end de la aplicación bajo el framework Dask con un único proceso worker. **`p=1` por sí solo NO constituye un barrido de strong scaling**, pues no varía recursos de cómputo.
2. **Ejecución local multi-worker (`p>1`):** En caso de autorización docente expresa (DEC-01), se denominará estrictamente «ejecución distribuida multiproceso en una única máquina». Investiga la sobrecarga del framework, IPC por sockets loopback, sincronización y contención de memoria en el host heterogéneo. **No demuestra escalabilidad de red inter-máquina ni capacidad de memoria agregada multi-máquina.**
3. **Ejecución multinodo real:** Requiere múltiples máquinas físicas conectadas por una red real, con telemetría de red física independiente e inventario por nodo. Actualmente **BLOCKED / NOT AVAILABLE**.

> [!WARNING]
> **Estado formal del requisito de strong scaling:**
> Si la profesora no autoriza la ejecución local con `p>1` y no se dispone de infraestructura multinodo acreditada, el estudio obligatorio de *strong scaling* queda formalmente registrado como **BLOCKED / NOT COMPLETED**.
> Está terminantemente prohibido sugerir o implicar que el requisito de strong scaling de la asignatura ha sido satisfecho ejecutando `p=1` y contrastándolo contra los baselines de Assignment 1 (secuencial) o Assignment 3 (OpenMP). La comparación entre métodos evalúa sistemas con fronteras temporales y backends distintos, no escalabilidad de workers. Se registra la necesidad de obtener la autorización docente (DEC-01) o una alternativa explícitamente autorizada.

---

## 4. Weak scaling opcional

Weak scaling no forma parte de la campaña obligatoria inicial y queda aplazado hasta completar y auditar el estudio principal.

Para multiplicación densa, el trabajo aritmético aproximado es `W(n)=2n³`. Para mantener aproximadamente constante el trabajo teórico por worker:

`n(p) = n(1) × p^(1/3)`.

Un diseño posterior podría partir de `n(1)=500` y usar `b=125`, con dimensiones redondeadas a múltiplos de bloque:

| `p` | `n` continuo | `n` candidato | Error aproximado de FLOPs por worker frente a p=1 |
|---:|---:|---:|---:|
| 1 | 500,00 | 500 | 0 % |
| 2 | 629,96 | 625 | −2,34 % |
| 4 | 793,70 | 750 | −15,63 % |
| 8 | 1000,00 | 1000 | 0 % |
| 10 | 1077,22 | 1125 | +13,91 % |

La tabla no autoriza crear matrices de 625, 750, 1000 o 1125 para weak scaling; `750` y `1000` ya existen como dimensiones congeladas, pero no sustituyen al input de otras dimensiones ni autorizan reutilizar inputs sintéticos de forma distinta. Los inputs nuevos requerirían aprobación, generación controlada, hashes y una nueva versión del manifiesto de datos.

La regla iguala FLOPs teóricos por worker, no bytes transferidos, memoria, número de tareas ni tiempo real por worker. La heterogeneidad de los núcleos locales y la desigualdad del redondeo deben figurar como limitaciones. Weak scaling solo se ejecutará con autorización de la topología `p>1` y aprobación de un protocolo complementario.

---

## 5. Bloques, tarea de multiplicación y bordes

### 5.1 Tamaño de bloque de referencia

Se fija `b=250` como bloque de referencia del protocolo, no como tamaño demostrado óptimo.

Cada bloque `250×250` en `float64` ocupa 500.000 bytes (aproximadamente 488,3 KiB). Este volumen es manejable como objeto individual, pero su idoneidad final depende del coste real de las tareas, serialización, planificación, BLAS y transferencia.

Para una matriz `n×n`:

- `Nb = ceil(n/b)`;
- tarea de producto: `P[i,j,k] = A[i,k] @ B[k,j]`;
- resultado: `C[i,j] = sum_k P[i,j,k]`;
- reducción: árbol binario, con `Nb−1` sumas por bloque de salida.

La estimación estructural de las tres matrices densas A, B y C es `M_ABC = 3 × 8 × n² bytes`; para `n=2000`, son 96.000.000 bytes (aproximadamente 91,6 MiB), sin contar copias ni temporales.

La estimación de que todos los productos parciales estén residentes simultáneamente es:

`M_partial ≈ Nb³ × b² × 8 bytes = Nb × n² × 8 bytes`.

Para `n=2000`, `b=250` y `Nb=8`, son 256.000.000 bytes (aproximadamente 244,1 MiB). Es una estimación estructural de un escenario de residencia completa, no una predicción de RSS ni una afirmación sobre el comportamiento real del scheduler.

### 5.2 Calibración

El bloque principal de strong scaling permanece fijo en 250. Una calibración independiente podrá comparar `b∈{125,250,500}` con `n=1000`, manteniendo fijo `p` durante esa calibración.

No se mezclará la calibración con los resultados de strong scaling ni se cambiará `b` dentro de un barrido de workers. La calibración requerirá autorización de piloto; si `p>1` sigue bloqueado, sus conclusiones se limitarán a `p=1`.

### 5.3 Bloques de borde

Para dimensiones no divisibles por `b`, el último bloque de cada eje utilizará su tamaño real:

`edge = min(b, n − start)`.

No se rellenará con ceros. Las dimensiones internas de `A[i,k]` y `B[k,j]` deberán coincidir y la forma de cada bloque de salida se corresponderá con las filas/columnas efectivas.

Los fixtures pequeños incluirán un caso de partición con dimensión no divisible por el bloque configurado (por ejemplo, `n=4`, `b=3`) para comprobar los bordes sin convertirlo en una medición de rendimiento.

---

## 6. Contrato de temporización

### 6.1 Startup y carga de datos (fuera del timer principal)

1. Crear `LocalCluster` y `Client`.
2. Esperar a que los workers solicitados estén registrados y listos.
3. Registrar `cluster_startup_s`, número real de workers, procesos, nodos y PIDs.
4. Cargar A y B desde los CSV a memoria del driver.
5. Registrar `input_load_s` por dimensión.

El tiempo de arranque se informará separadamente y no se mezclará con el tiempo por multiplicación. La carga de los CSV tampoco entra en el timer principal; esta exclusión se aplicará a todas las repeticiones Dask.

### 6.2 Timer end-to-end principal de Dask

**Inicio (`t0`):** inmediatamente antes de partir/copiar A y B en bloques para la repetición actual.

Dentro del timer:

1. Particionar A y B en bloques 2D contiguos C-order.
2. Colocar/dispersar los bloques en el cluster (`scatter` o mecanismo equivalente aprobado).
3. Construir el grafo de multiplicaciones y reducciones.
4. Enviar el grafo y esperar su ejecución.
5. Ejecutar los productos `numpy.matmul`.
6. Ejecutar la reducción por árbol.
7. Recuperar todos los bloques de C en el driver.
8. Ensamblar la matriz densa C.
9. Consumir todos los elementos mediante `checksum = float(np.sum(C, dtype=np.float64))`.

**Fin (`t1`):** después de que el checksum haya leído la matriz completa y el resultado esté materializado.

`T_Dask_E2E = t1 − t0`, usando `time.perf_counter()`.

La validación elemento a elemento frente a una referencia se realiza después de detener el timer. El checksum fuerza la lectura completa, pero no demuestra por sí solo la corrección de la matriz.

### 6.3 Repeticiones sin reutilización accidental

Cada repetición utilizará claves/tareas únicas o un mecanismo equivalente para evitar que Dask reutilice silenciosamente resultados previos por deduplicación/caché. Los bloques se dispersarán en cada repetición dentro del timer. Después se liberarán referencias/futures de la repetición anterior fuera del timer, sin modificar los inputs congelados.

### 6.4 Baselines A1/A3

Los tiempos A1 y A3 se registrarán exactamente con los límites internos originales de sus harnesses. No se ampliarán artificialmente para que parezcan end-to-end. La carga de datos y asignaciones excluidas por sus timers seguirán excluidas.

En los harnesses congelados de los baselines common-input, A1 registra `WorkingSetSize` como snapshot del proceso después de la llamada cronometrada; A3 registra `PeakWorkingSetSize`, el máximo acumulado durante la vida del proceso. Son métricas distintas y no deben presentarse como equivalentes. El pico de A3 puede incluir ejecuciones anteriores de la misma invocación con varios tamaños.

---

## 7. Fases y métricas que Dask puede exponer

### 7.1 Diagnóstico disponible

La API documentada de Dask ofrece `get_task_stream`, que registra eventos por tarea, incluyendo intervalos de ejecución, transferencia y deserialización. La captura diagnóstica no se activa necesariamente por defecto y debe comprobarse/activarse en la versión fijada. `performance_report` puede guardar un informe HTML de diagnóstico. Estas ejecuciones se harán separadas de la campaña primaria para no contaminar el benchmark con instrumentación adicional.

Se podrán analizar, según versión/API:

- número, prefijo, worker y thread de cada tarea;
- tiempo de ejecución por tarea;
- eventos/intervalos de transferencia de dependencias;
- serialización/deserialización cuando aparezcan en el registro;
- spill a disco, pausas y errores observados;
- tamaño estimado de objetos gestionados por Dask.

### 7.2 Bytes y transferencia: terminología obligatoria

No se tratará `nbytes` ni `Client.nbytes()` como contador de bytes físicos de red. `Client.nbytes()` describe bytes estimados asociados a claves/objetos del cluster, no el tráfico físico completo.

Si la versión fijada permite recoger contadores o logs específicos de transferencia (`incoming`/`outgoing`, bytes totales y bytes comprimidos), se podrán registrar como **bytes de payload contabilizados por Dask**, especificando el API/campo y su alcance. Algunos detalles de esos logs son dependientes de versión o de implementación interna; se comprobarán con la versión realmente instalada y no se asumirán como API estable.

Se distinguirán:

- driver → worker (colocación inicial/scatter);
- worker → worker (dependencias entre workers);
- worker → driver (recolección de resultados);
- tamaños de objetos (`nbytes`);
- bytes de payload transferido/comprimido contabilizados por Dask, si están disponibles;
- tráfico físico de la interfaz de red, que no se afirmará haber medido salvo que exista un contador independiente y una metodología válida.

En topología de una máquina se hablará de comunicación local/IPC/socket loopback y overhead de serialización. No se inferirá latencia de switches, red física ni rendimiento multinodo.

### 7.3 Scheduler y solapamiento

Dask no proporciona por medio del Task Stream una medida completa y aislada de toda la CPU/latencia interna del scheduler como una fase wall-clock independiente. Las duraciones de tareas y dependencias pueden solaparse; no se sumarán para reconstruir el tiempo end-to-end.

Los tiempos de fases diagnósticas se interpretarán como observaciones de componentes, no como una descomposición aditiva garantizada de `T_Dask_E2E`.

### 7.4 Memoria

Se informarán por separado:

| Métrica | Definición | Limitación |
|---|---|---|
| RSS/Working Set actual | Memoria residente de cada PID en cada instante muestreado. En Windows se indicará el campo/API concreto. | Es una medida por proceso y puede incluir páginas compartidas. |
| Suma RSS de procesos | Suma de RSS del conjunto de PIDs únicos observados (driver, scheduler, nanny y workers) en cada instante. Cuando Scheduler y Nanny comparten el PID del Driver en LocalCluster, cada PID único se muestrea una sola vez. | Puede contar dos veces páginas compartidas entre PIDs distintos; no equivale exactamente a memoria física única del host. |
| Pico muestreado de suma RSS | Máximo temporal de la suma RSS simultánea, muestreada cada 250 ms. | Picos más breves que el intervalo pueden no capturarse. |
| Peak Working Set | Máximo histórico por proceso/PID. | No se sumarán máximos individuales como si fueran simultáneos. |
| Memoria gestionada Dask | Bytes que Dask contabiliza como objetos gestionados por worker. | No incluye todo el RSS, buffers, temporales BLAS, Python ni memoria nativa no gestionada. |
| Memoria estructural | Estimación analítica de A+B+C y productos parciales. | No es una medida del consumo real del proceso. |
| Memoria disponible del host | Memoria física disponible global observada durante la muestra/campaña. | No es memoria atribuible al experimento. |

La suma RSS de procesos se presentará como proxy de contabilidad de procesos, no como número exacto de bytes físicos exclusivos. Se registrará también la memoria disponible del host para contextualizarla.

### 7.5 `memory_limit` de Dask

`memory_limit` es un límite objetivo por worker y, con procesos, se aplica de forma best-effort; no es un límite duro del RSS total del experimento. Dask puede iniciar spill a disco, pausar trabajo y, según umbrales/configuración, reiniciar workers. Se registrará la configuración y el comportamiento efectivo de la versión instalada.

En Phase 6 se verificó y reconcilió la política efectiva con `ClusterManager`:
- Se pasa `memory_limit=None` al crear `LocalCluster`, lo que significa que no se aplica ningún límite de memoria por worker en Dask.
- Los mecanismos automáticos de spill a disco, pausa o terminación de workers dependientes de dicho límite interno de Dask permanecen inactivos en esta configuración.
- El watchdog agregado de RSS del proyecto (`ResourceWatchdog`, §10) se mantiene como la salvaguarda de seguridad operacional especificada.
- Se preserva íntegramente la decisión previa del protocolo de no utilizar la fórmula fija `(3 GiB − 1 GiB)/p`.
- El techo agregado de RSS de 2.5 GiB, el suelo de RAM disponible de 1.0 GiB y el intervalo de muestreo de 250 ms permanecen estrictamente inalterados.
- Se subraya que el watchdog es una salvaguarda operacional muestreada, no un límite duro del sistema operativo.

---

## 8. Warm-ups, repeticiones, orden y consumo

- Un warm-up completo por configuración `(método, n, b, p, threads)`, excluido de estadísticas.
- Cinco repeticiones medidas por configuración como mínimo objetivo.
- Una instancia de cluster por configuración Dask; se reutiliza para su warm-up y cinco repeticiones, registrando su startup una vez. No se ejecutan clusters simultáneos.
- Orden de configuraciones aleatorizado de forma reproducible con semilla fija `20261113`. Se registrará la permutación realmente ejecutada. Se evitará ordenar siempre por tamaño ascendente o ejecutar sistemáticamente el mismo método primero.
- El orden exacto, timestamps, pausas y cualquier reinicio del proceso/cluster quedarán registrados.
- No se eliminarán outliers. Solo se excluirá una muestra de un resumen de tiempos completados cuando tenga un estado fallido/no completado o se documente una razón técnica predefinida; la fila raw seguirá conservada.
- El warm-up se identifica con `is_warmup=true`; no se mezcla con las cinco medidas.
- El checksum lee la matriz completa antes de parar el timer. La comprobación elemento a elemento sucede después.

Los kernels secuenciales/OpenMP se ejecutarán usando el harness original. En Phase 4 se inspeccionará si los harnesses permiten seleccionar dimensiones/métodos por argumentos. Si no lo permiten, el timeout será aplicado a la invocación completa del ejecutable y las filas de tamaños que no lleguen a ejecutarse se registrarán como no realizadas, sin modificar el baseline.

---

## 9. Corrección numérica y referencia confiable

### 9.1 Fixtures

1. Identidad: `A × I = A`.
2. Matriz cero: `A × 0 = 0`.
3. Caso entero pequeño 4×4 con resultado conocido exactamente.
4. Caso de bloque de borde no divisible (por ejemplo, 4×4 particionado con `b=3`).
5. Caso no trivial `n=250` comparado elemento a elemento con `data/validation/reference/C_250.csv`.
6. Comprobación de `NaN`/`Inf` en todas las salidas validadas.

Las pruebas de corrección se ejecutarán en Phase 5 con autorización específica y antes de las mediciones de rendimiento. No forman parte del timer principal.

### 9.2 Referencia `C_250.csv`

Archivo: `data/validation/reference/C_250.csv`
SHA-256 reportado: `598e9a9fc8f0c8db1d5da277f4d93561ad83e2df77e3fcb5f968f67b8033909e`
Generador: `scripts/generate_dataset.py`, función `generate_reference_product()`.

Procedimiento algorítmico reportado en Phase 2.1:

- leer `A_250.csv` y `B_250.csv` congelados como números de coma flotante binary64;
- para cada elemento `C[i,j]`, inicializar `acc = 0.0`;
- recorrer `k=0,1,...,249` en orden ascendente;
- ejecutar `acc += A[i,k] * B[k,j]` en cada iteración;
- guardar `acc` en `C[i,j]`;
- escribir el resultado con la rutina/formato de serialización del generador existente, sin regenerar el archivo durante esta fase.

Es una **referencia numérica calculada**, no un resultado analítico. Su independencia se basa en el cálculo escalar de CPython y en no utilizar NumPy, BLAS ni Dask para obtener el producto. La suma sigue siendo coma flotante y su orden se fija explícitamente. El SHA-256 congelado es la evidencia de identidad del archivo; no se recalculará ni reemplazará en Phase 3.

La inspección estática posterior de `generate_reference_product()` documentará también el formato exacto de escritura usado por el código existente. No se inventarán detalles de serialización que no estén en el script.

### 9.3 Dtype y tolerancias

Tipo de referencia: IEEE 754 binary64 / NumPy `float64` y C `double`.

Criterio provisional por elemento:

`abs(C_test − C_ref) <= atol + rtol × abs(C_ref)`

con:

- `atol = 1e-9`;
- `rtol = 1e-12`.

Estos valores son tolerancias de aceptación propuestas, no una cota universal demostrada. Para binary64, `u≈1.11×10⁻¹⁶`; el factor de acumulación `γ_n = n u/(1−n u)` es una referencia útil para estudiar el error de una suma de longitud `n`, pero el error de un producto matricial también incluye redondeo de multiplicaciones, orden de reducción y posibles FMA/BLAS. Por tanto, no se justificará la tolerancia únicamente con `n·u`.

La tolerancia se validará en Phase 5 frente a la referencia y las implementaciones de baseline. Si falla, se investigará la causa antes de relajarla. No se cambiarán tolerancias después de observar resultados para forzar un PASS.

`C_250.csv` valida directamente `n=250`; no se afirmará que sea una referencia elemento a elemento para `n=500...2000`. Las dimensiones grandes se someterán a los controles de finitud, invariantes y comprobaciones cruzadas que permitan los outputs originales sin modificarlos. Cualquier comprobación limitada a checksum se describirá como control parcial, no como prueba completa de corrección.

---

## 10. Presupuestos, límites y tratamiento de fallos

### 10.1 Timeout por muestra

- **Valor adoptado:** **120 segundos por multiplicación end-to-end**.
- **Naturaleza del límite:** Es un **límite de seguridad operacional seleccionado por el estudiante** y un **límite provisional validado empíricamente en el piloto**. No debía describirse como empíricamente validado antes de que el piloto confirmase que las dimensiones seleccionadas se ejecutan dentro de este margen.
- **Baselines C:** Durante la preparación/auditoría de ejecución se comprobará si los ejecutables admiten aislar cada tamaño/método por argumentos sin modificar los fuentes congelados. Si no, el límite se aplicará a la invocación completa y el estado de cada tamaño se derivará únicamente de salidas efectivamente producidas (nunca se afirmará un timeout por tamaño cuando solo se ha limitado el proceso completo).
- **Acción ante timeout:** Cancelar inmediatamente las tareas pendientes, cerrar el cluster/proceso de forma segura, registrar `status=TIMEOUT`, `timeout_limit_s=120.0` y `elapsed_until_abort_s`. `time_e2e_s` queda nulo para la muestra fallida y se omiten las repeticiones restantes de esa configuración (`NOT_RUN_AFTER_TIMEOUT`), salvo que una revisión técnica y autorización posterior permitan reanudarla.

### 10.2 Presupuesto temporal global de campaña

La campaña máxima de referencia incluye:

- A1: cinco dimensiones del harness (`100` auxiliar y cuatro comunes);
- A3: `seq_ikj` y `omp_ikj` en seis dimensiones;
- A3 `omp_tiled_b32`: secundario, solo si el presupuesto restante lo permite, en tamaños comunes;
- Dask: `p=1` en los seis tamaños; además, para el strong scaling, `p={2,4,8,10}` en `n={1000,1500,2000}` solo si cada configuración está autorizada.
- Un warm-up y cinco repeticiones medidas por configuración.

- **Valor adoptado:** **Tope global de 10 horas de reloj** (que incluye warm-ups, medidas, startup Dask, diagnósticos y limpieza; compilación y validación de corrección pertenecen a fases diferentes).
- **Naturaleza del límite:** Es un **límite de seguridad operacional y envolvente de planificación seleccionado por el estudiante** para acotar la duración de la batería de pruebas en el host personal. No es una cota empírica demostrada.
- **Composición:** El núcleo (sin la variante tiled secundaria) suma 35 grupos de configuración en el escenario completo autorizado: 5 A1 + 12 A3 (dos métodos × seis tamaños) + 18 Dask (seis tamaños a `p=1` y doce combinaciones extra de `p>1` para los tres tamaños de strong scaling), totalizando 210 muestras (1 warm-up + 5 medidas). La variante tiled secundaria añade hasta cuatro grupos (24 muestras). Con 120 s como límite por muestra, el máximo aritmético conservador es de 7 h para el núcleo (o 7 h 48 min con la variante tiled); el margen hasta 10 h absorbe arranques Dask, diagnósticos y pausas térmicas.
- **Acción ante agotamiento y escenario bloqueado:** Si se alcanzan las 10 h, la campaña se detiene inmediatamente y las configuraciones no iniciadas se marcan como `NOT_RUN_BUDGET`; el presupuesto no se amplía automáticamente. Si `p>1` continúa bloqueado bajo DEC-01, las configuraciones Dask se reducen a los seis tamaños con `p=1` y no existe campaña de strong scaling multinivel.

### 10.3 Presupuesto de memoria, semántica de watchdog y procedimiento de parada

Los umbrales de memoria se establecen para proteger la estabilidad del host Windows y evitar contención o paginación en disco:

- **2.5 GiB como techo de la suma RSS muestreada:** **Límite de seguridad operacional seleccionado por el estudiante** y **límite provisional sujeto a validación en el piloto**.
- **1.0 GiB como suelo de memoria disponible en el host:** **Límite de seguridad operacional del sistema operativo seleccionado por el estudiante**. Si la RAM física libre del sistema desciende de 1.0 GiB, la prueba se aborta para evitar congelación del sistema operativo.
- **250 ms como intervalo de muestreo del watchdog:** **Parámetro operacional de monitorización seleccionado por el estudiante**. Representa un compromiso práctico entre latencia de detección y sobrecarga de CPU de sondeo.

#### Semántica obligatoria de métricas y procesos:
1. **No equivalencia de la suma RSS con RAM física exclusiva:** La suma de RSS de múltiples procesos (`rss_process_sum`) **no es equivalente** a memoria física RAM única consumida, ya que páginas de memoria compartida (bibliotecas dinámicas de C/Windows, intérprete Python, código compartido) se contabilizan simultáneamente en cada PID. La suma RSS es un proxy conservador de contabilidad de procesos.
2. **Pico muestreado vs. pico real:** El intervalo de muestreo de 250 ms mide el *pico muestreado* (*sampled peak aggregate RSS*) y **puede no registrar picos transitorios breves** que ocurran entre dos instantes de sondeo.
3. **Rastreo estricto de PIDs:** El watchdog debe identificar y rastrear mediante PIDs únicos individuales todos los procesos activos del árbol: Driver/Client, Scheduler, Nannies y Workers. En `LocalCluster` donde Scheduler y Nanny comparten el PID del Driver, se rastrean los PIDs únicos del conjunto sin duplicar muestras.
4. **Semántica de `memory_limit` de Dask:** Con `memory_limit=None`, Dask no aplica límites de derrame o pausa por worker; la salvaguarda de seguridad operacional primaria reside exclusivamente en el watchdog global externo (2.5 GiB de techo agregado y 1.0 GiB de suelo en host como límites muestreados, no límites duros del sistema operativo).

#### Procedimiento formal de parada segura del watchdog:
Si el watchdog detecta que la suma RSS muestreada alcanza o supera 2.5 GiB, o que la memoria disponible del host cae por debajo de 1.0 GiB:
1. **Paso 1 (Detección y señal de parada):** Registrar la alerta de memoria con los PIDs y valores exactos en el instante de corte.
2. **Paso 2 (Cancelación de tareas):** Invocar inmediatamente la cancelación de todos los futuros y tareas en vuelo en el cliente Dask (`client.cancel`).
3. **Paso 3 (Cierre ordenado):** Ejecutar el cierre formal del cliente y del cluster (`client.close()`, `cluster.close()`) compartiendo un presupuesto unificado de hasta 5 segundos (objetivo operacional acotado por un plazo compartido; en Windows la latencia del sistema operativo para limpieza de procesos y sockets puede situar la ventana efectiva observada entre 5,0 y 5,2 s).
4. **Paso 4 (Terminación forzosa):** Si transcurrido el tiempo restante del presupuesto compartido algún proceso hijo (nanny o worker) sigue activo, terminarlo forzosamente mediante la API de procesos (`psutil.Process.terminate() / kill()`). El proceso driver (`os.getpid()`) queda explícitamente preservado para registrar los resultados y sidecar.
5. **Paso 5 (Registro de estado):** Marcar la muestra como `OUT_OF_MEMORY` o `ABORTED_WATCHDOG`, dejando `time_e2e_s = NULL` y guardando la telemetría observada hasta la interrupción. Detener las repeticiones restantes de esa configuración (`NOT_RUN_AFTER_OOM`).

> [!NOTE]
> La fase de piloto (Phase 7) tendrá como objetivo empírico determinar si las matrices seleccionadas y el número de workers pueden ejecutarse de forma estable dentro de estos límites operacionales. En esta fase de documentación no se lanza ningún piloto.

### 10.4 Timeout, OOM y fallos

- Si hay OOM, reinicio de nanny/worker, fallo de tarea o spill inesperado que invalide el régimen de medición, se conserva el registro y los logs.
- Un worker reiniciado se registra; no se presenta la repetición como una ejecución normal sin documentar el incidente.
- Tras el primer OOM de una configuración se detienen sus repeticiones restantes y se marca `NOT_RUN_AFTER_OOM`.
- Si el host alcanza el umbral de memoria disponible, se detiene el experimento completo.
- No se inventan duraciones de ejecución para muestras fallidas ni se borran intentos incompletos.
- `largest_attempted_n` y `largest_completed_n` se reportan por separado.

### 10.5 Especificación formal y vinculante del piloto de Phase 7 (Prompt 7)

Conforme a la autorización estudiantil independiente de la Etapa 1 (formalización documental) y en cumplimiento estricto de las directrices docentes (`BD_Individual_Assignments_2026_2027.pdf`) y `SPEC.md`, se establece la especificación completa, acotada y vinculante para el piloto experimental de Phase 7:

#### 1. Matriz de prueba acotada del piloto:
- **Método evaluado:** `dask_block_2d` exclusivamente (implementación distribuida objeto del estudio en `src/dask_matrix/distributed_engine.py`). No se ejecutan baselines de A1 ni de A3 en el piloto.
- **Dimensiones de matriz ($n$):** $n \in \{250, 500\}$ exclusivamente.
  - $n = 250$: $N_b = 1$, 1 tarea de multiplicación de bloque, 0 reducciones intermedias. Validación de latencia base del framework y verificación de sobrecarga elemental.
  - $n = 500$: $N_b = 2$, 8 tareas de multiplicación de bloque, 4 tareas de reducción en árbol binario. Primera dimensión con particionado 2D y reducción intermedia activa.
- **Tamaño de bloque ($b$):** $b = 250$ fijo (bloque de referencia del estudio).
- **Asignación de workers ($p$):** $p = 1$ exclusivamente. Cumplimiento riguroso de DEC-01 (Gate G1); no se inician múltiples workers locales sin permiso docente registrado en `docs/decisions/`.
- **Topología de procesos y máquinas:** Un único host físico (`localhost` / `127.0.0.1`). Un proceso Driver/Scheduler/Nanny y un único proceso Worker independiente (`LocalCluster(processes=True, n_workers=1, threads_per_worker=1)`), verificando exactamente 2 PIDs de sistema operativo observados en runtime.
- **Concurrencia de hilos:** `threads_per_worker = 1` a nivel framework Dask; `OpenBLAS num_threads = 1` a nivel backend numérico (mediante herencia de `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
- **Esquema de repeticiones:**
  - 1 warm-up completo (`is_warmup=true`, `repetition=0`) por tamaño.
  - 2 repeticiones medidas (`is_warmup=false`, `repetition=1` y `repetition=2`) por tamaño.
- **Volumen total de ejecuciones:** Exactamente **6 ejecuciones de muestra en total** ($2 \text{ tamaños} \times 3 \text{ corridas} = 6$).

| Muestra ID | Tamaño $n$ | Bloque $b$ | Tareas mult ($N_b^3$) | Tareas red | Workers $p$ | Repetición | Tipo |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pilot_250_w0` | 250 | 250 | 1 | 0 | 1 | 0 | Warm-up (`is_warmup=true`) |
| `pilot_250_r1` | 250 | 250 | 1 | 0 | 1 | 1 | Medida (`is_warmup=false`) |
| `pilot_250_r2` | 250 | 250 | 1 | 0 | 1 | 2 | Medida (`is_warmup=false`) |
| `pilot_500_w0` | 500 | 250 | 8 | 4 | 1 | 0 | Warm-up (`is_warmup=true`) |
| `pilot_500_r1` | 500 | 250 | 8 | 4 | 1 | 1 | Medida (`is_warmup=false`) |
| `pilot_500_r2` | 500 | 250 | 8 | 4 | 1 | 2 | Medida (`is_warmup=false`) |

#### 2. Presupuestos y límites operacionales del piloto:
- **Taxonomía formal y distinción de límites temporales:**
  1. **Presupuesto temporal global de sesión (900.0 s):** Límite monotónico global que cubre todo el ciclo de vida (`session_start + 900.0 s`), incluyendo lectura de matrices, arranque del cluster, 6 ejecuciones, validación y serialización. Su agotamiento cancela la sesión de forma inmediata y marca todas las muestras incompletas como `ABORTED_BUDGET`.
  2. **Timeout individual por muestra (120.0 s):** Límite estricto aplicado exclusivamente a la multiplicación distribuida end-to-end ($T_{\text{Dask\_e2e}}$). Su superación marca la muestra como `TIMEOUT` y cancela las repeticiones restantes de esa dimensión (`NOT_RUN_AFTER_TIMEOUT`).
  3. **Timeout de arranque de cluster (60.0 s nominales o remanente de sesión):** Límite aplicado a la conexión del Client y registro de workers (`wait_for_workers`). Si el arranque falla o agota sus 60 s mientras resta presupuesto global, se clasifica rigurosamente como `FAILED` (no como `ABORTED_BUDGET`).
  4. **Construcción síncrona no interrumpible de `LocalCluster`:** La llamada a `LocalCluster(...)` en Dask es síncrona y no dispone de parámetro de timeout nativo en la API de Dask. Su ejecución (~0.5–2.0 s) no puede interrumpirse preventivamente a mitad de llamada; el arnés detecta el rebasamiento a posteriori tras su retorno y dispara el cierre inmediato en caso de overrun.
  5. **Timeout nominal de limpieza ($\le 5.0$ s) frente a latencias del SO:** El desmontaje ordenado (`ClusterManager.close()` y safe-stop del watchdog) comparte un plazo unificado nominal de hasta 5.0 s para cerrar Client, Cluster y terminar procesos hijos (`terminate()` y `kill()`), excluyendo estrictamente el PID del driver. Debido a la latencia de conmutación y terminación de procesos en el SO (Windows), puede observarse un retraso residual de decenas de milisegundos (~5.0 a 5.2 s), constituyendo una envolvente de seguridad software y no una cota dura de tiempo real.
- **Techo de memoria agregada (Watchdog):** 2.560,0 MiB (2,5 GiB) de suma RSS simultánea sobre los PIDs del cluster.
- **Suelo de memoria física disponible:** 1.024,0 MiB (1,0 GiB) en el host.
- **Intervalo del watchdog:** Muestreo cada 250 ms en hilo daemon (`ResourceWatchdog`).
- **Límite interno Dask:** `memory_limit=None` (sin spill ni pausas internas dependientes de Dask; la salvaguarda primaria es el watchdog externo).
- **Lectura de matrices desde disco:** La lectura de matrices (`input_load_s`) mediante `np.loadtxt()` es una operación síncrona local no interrumpible; el arnés comprueba el plazo inmediatamente tras su carga y descarta el arranque del cluster si el presupuesto expiró.
- **Límite temporal global efectivo y corte seguro:** El arnés del piloto calcula un tiempo límite de sesión monotónico (`deadline = session_start + 900.0 s`). El intervalo de muestreo del watchdog (250 ms) introduce una latencia máxima de detección operacional de hasta 250 ms ante tareas de cómputo bloqueadas. Si se alcanza dicho plazo en cualquier punto, se detiene la secuencia de inmediato, se cancelan futuros en vuelo, se cierra el cluster de forma segura y las muestras no ejecutadas se marcan formalmente como `ABORTED_BUDGET`. El presupuesto representa una envolvente de seguridad operacional acotada y cooperativa en software, no una cota de tiempo real externa o forzosa a nivel de kernel de SO.

#### 3. Frontera temporal exacta ($T_{\text{Dask\_e2e}}$):
- **Inicio del timer ($t_0$):** Inmediatamente antes del particionado en bloques 2D de $A$ y $B$ y su dispersión (*scatter*) al cluster.
- **Fase medida ($t_0 \to t_1$):** Particionado 2D, envío de tareas $N_b^3$, envío de árbol de reducción, espera de completitud, recolección (*gather*) de bloques $C_{ij}$ al driver, ensamblado contiguo en memoria local de $C$, y cálculo del checksum completo `checksum = float(np.sum(C, dtype=np.float64))`.
- **Parada del timer ($t_1$):** Inmediatamente tras la computación del checksum escalar. $T_{\text{Dask\_e2e}} = t_1 - t_0$ medido mediante `time.perf_counter()`.
- **Fases excluidas del timer de muestra (pero incluidas en el presupuesto global de 900 s):**
  - Fuera del timer (previo): Arranque del cluster Dask (`cluster_startup_s`) y lectura de ficheros CSV desde disco a arrays NumPy en el driver (`input_load_s`).
  - Fuera del timer (posterior):
    - Para $n=250$: Verificación numérica elemento a elemento contra la referencia numérica calculada (`data/validation/reference/C_250.csv`) con tolerancias `atol=1e-9` y `rtol=1e-12`.
    - Para $n=500$: Verificación de finitud (ausencia de NaN/Inf) y comprobación de determinismo de checksums entre repeticiones medidas 1 y 2. Al no existir un fichero congelado `C_500.csv` en el repositorio, esta verificación se reporta con rigor como un **control parcial de invariantes** conforme a §9.3, no como una prueba matemática completa contra ground-truth analítico.
    - Registro de telemetría de red/memoria y serialización append-only a disco.

#### 4. Política estricta de registro y metadatos:
- **Adición no destructiva (*append-only*):** Las observaciones se agregan exclusivamente al final de `results/raw/raw_measurements.csv` sin sobrescribir ni truncar datos previos.
- **Identificación de warm-ups:** Cada warm-up se registra explícitamente con `is_warmup=true` y `repetition=0`, quedando inequívocamente aislado y excluido de los resúmenes estadísticos posteriores (medianas e IQRs de tiempo).
- **Trazabilidad de metadatos:** Se genera un fichero companion JSON en `results/metadata/sidecar_{run_id}.json` con identificador único de ejecución (`run_id = f"pilot_{timestamp}_{uuid.uuid4().hex}"` para piloto o `f"campaign_{timestamp}_{uuid.uuid4().hex}"` para campaña; UUIDv4 aportando 122 bits aleatorios de entropía en representación de 32 caracteres hexadecimales, conforme a RFC 4122 §4.4) registrando la jerarquía de PIDs, hardware, SO, versiones del stack, límites del piloto y estado final (`status`, `failure_reason`). Si ocurre un fallo en arranque o ejecución, el sidecar captura los PIDs disponibles y el estado de error; su generación se ejecuta durante el desmontaje final fuera de los timers de cómputo.
- **Preservación de fallos:** Cualquier interrupción (por timeout, RSS, suelo de RAM o presupuesto de 15 min) se guarda con su estado formal (`TIMEOUT`, `OUT_OF_MEMORY`, `ABORTED_WATCHDOG`, `ABORTED_BUDGET`), conservando los logs para su auditoría técnica.

---

## 11. Raw data, metadatos, análisis y figuras

### 11.1 Una fila por muestra

Esquema normativo y vinculante de 44 columnas para `results/raw/raw_measurements.csv` (implementado en `MeasurementRecord`):

| Campo | Descripción |
|---|---|
| `run_id`, `sample_id`, `timestamp_utc` | Identificación única y marca temporal UTC. |
| `method`, `framework`, `framework_version` | Método y stack empleado. |
| `source_revision`, `compiler_version`, `build_flags` | Procedencia/build de C; nulo/no aplicable en métodos Python. |
| `n`, `dtype`, `input_A_sha256`, `input_B_sha256` | Problema y inputs exactos. |
| `block_size`, `algorithm_variant` | Parámetros del método; nulo cuando no aplique. |
| `worker_count`, `process_count`, `physical_machine_count` | Workers, procesos y máquinas físicas por separado. |
| `worker_threads`, `openmp_threads`, `blas_threads` | Paralelismo configurado/observado. |
| `repetition`, `is_warmup`, `execution_order` | Calentamiento, repetición y orden. |
| `time_kernel_s`, `time_e2e_s` | Tiempo con el límite que realmente corresponde; no rellenar uno con otro. |
| `timing_boundary_id` | Identificador del contrato temporal. |
| `cluster_startup_s`, `input_load_s` | Tiempos fuera del timer principal. |
| `client_to_worker_payload_bytes`, `worker_to_worker_payload_bytes`, `worker_to_client_payload_bytes` | Contadores por dirección, nullable si no disponibles. |
| `compressed_transfer_bytes`, `transfer_metric_source` | Payload comprimido y API/campo que lo aporta, nullable si no disponibles. |
| `object_nbytes` | Tamaños estimados de objetos; no denominar bytes físicos transferidos. |
| `rss_driver_mib`, `rss_process_sum_peak_sampled_mib` | RSS del driver y máximo muestreado de la suma RSS. |
| `peak_working_set_mib` | Máximo Working Set individual observado en los procesos Dask supervisados en Windows durante la ventana de la muestra, obtenido mediante sondeos `psutil` `.rss` / Working Set y acompañado del mapeo de PIDs en el sidecar. En el resumen separado de baselines common-input, A1 conserva el snapshot `WorkingSetSize` posterior a la llamada cronometrada y A3 conserva `PeakWorkingSetSize` durante la vida del proceso. |
| `dask_managed_memory_mib` | Memoria gestionada por Dask, nullable. |
| `checksum_sum` | Suma de C calculada antes de parar el timer. |
| `status`, `error_reason` | `COMPLETED`, `TIMEOUT`, `OUT_OF_MEMORY`, `FAILED`, `NOT_RUN_BUDGET`, etc. |
| `timeout_limit_s`, `elapsed_until_abort_s` | Límite y tiempo transcurrido cuando se aborta; nullable si no aplica. |

Los valores ausentes serán vacíos/NULL con motivo, nunca cero por conveniencia. Las muestras raw serán append-only.

### 11.2 Sidecars por campaña

Se guardará un manifiesto JSON/YAML por campaña con:

- commit Git A4 y estado de trabajo;
- commits A1/A3 y hashes de snapshots;
- SHA-256 de cada input empleado;
- CPU, núcleos físicos/lógicos, RAM, sistema operativo y versión;
- Python, NumPy, Dask, Distributed, BLAS/runtime y GCC/OpenMP;
- flags de compilación y variables de entorno de threads;
- configuración cluster, workers solicitados y observados, PIDs y ubicación por máquina;
- semilla/orden aleatorio, warm-ups, repeticiones, timeout y presupuesto;
- identificador del límite temporal y herramientas de memoria/transferencia;
- incidencias, reinicios, spill, fallos y desviaciones.

### 11.3 Estadísticos y fórmulas

Por configuración completada:

- mediana de las cinco muestras medidas;
- IQR = Q3 − Q1, con el método de cuantiles lineal (equivalente a `method="linear"` en NumPy) fijado para que el cálculo sea reproducible;
- número de muestras válidas y fallidas;
- unidades en segundos, MiB o bytes explícitos.

Strong scaling Dask, para una misma `n` y `b`:

`S_Dask(p,n) = median(T_Dask(1,n)) / median(T_Dask(p,n))`

`E_Dask(p,n) = S_Dask(p,n) / p`

Para OpenMP se podrá calcular `S_OMP(t,n)=median(T_seq_ikj(n))/median(T_omp_ikj(t,n))` y `E_OMP=S_OMP/t` únicamente si la inspección confirma que ambos tiempos A3 tienen límites comparables y si las configuraciones de threads están identificadas.

Para A1/A3 frente a Dask se calcularán, si aportan valor, ratios de tiempos observados (`T_kernel/T_Dask_E2E`). Se etiquetarán expresamente como ratios entre límites diferentes, no como speedups algorítmicos.

No se calcularán speedups si falta el denominador/base de referencia, si `p=1` no fue medido con el mismo protocolo, o si una de las muestras está fallida/timeout.

### 11.4 Figuras planificadas

1. Dask end-to-end vs `p`, una curva por cada `n∈{1000,1500,2000}`.
2. Speedup y eficiencia strong scaling de Dask, solo cuando haya al menos dos valores autorizados de `p`.
3. Comparación de medianas A1/A3/Dask en dimensiones comunes, con una nota visible sobre los límites temporales diferentes.
4. Comparación OpenMP frente a `seq_ikj` de A3, si los límites y el número de threads lo permiten.
5. Memoria: suma RSS muestreada de procesos Dask vs workers, acompañada por memoria disponible de host; Peak Working Set A1/A3 separado.
6. Diagnóstico Task Stream: ejecución, transferencia, deserialización y spill, como figura diagnóstica separada.
7. Bytes transferidos contabilizados por Dask, si los contadores están disponibles y su alcance está validado.

Todas las figuras tendrán ejes, unidades, `n`, `b`, workers/threads, topología y número de repeticiones.

---

## 12. Presupuesto de decisiones, puertas y autorización

### 12.1 Decisiones aceptadas para el diseño

| Decisión | Valor adoptado |
|---|---|
| Tamaños de comparación común | `250, 500, 750, 1000`. |
| Tamaños ampliados A3/Dask | `1500, 2000`. |
| `n=100` | Auxiliar exclusivo de A1. |
| Strong scaling principal | `n=1000,1500,2000`, `b=250`. |
| Workers candidatos | `p=1,2,4,8,10`; `p>1` autorizado en 1 máquina física vía `docs/decisions/DEC-01-instructor-permission.md` (2026-11-23; históricamente bloqueado en Phases 1–10). |
| Bloque | `b=250` referencia; calibración separada si se autoriza. |
| Weak scaling | Aplazado, opcional, nuevo protocolo/input si se activa. |
| Warm-up/repeticiones | 1 + 5 por configuración. |
| Timeout Dask | 120 s por muestra. |
| Campaña global | 10 h hard cap, sin ampliación automática. |
| Memoria | 2.5 GiB suma RSS muestreada y reserva mínima de 1 GiB disponible en el host. |
| Tolerancias candidatas | `atol=1e-9`, `rtol=1e-12`, pendientes de validación autorizada. |

### 12.2 Puertas de aceptación

| Gate | Nombre | Estado | Condiciones para avanzar |
|---|---|---|---|
| G0 | Discovery | **PASSED** | Repositorio, host y baselines auditados e inventariados. |
| G1 | Permission / Topology | **PASSED (Single-Host Local `p >= 1`, 2026-11-23) / BLOCKED (Multi-Node)** | Históricamente bloqueado para `p>1` en Phases 1–10; desbloqueado el 2026-11-23 mediante `docs/decisions/DEC-01-instructor-permission.md` para ejecución local multiproceso en 1 máquina física (`physical_machine_count=1`). Multi-nodo continúa **BLOCKED / NOT AVAILABLE**. |
| G2 | Framework / Architecture | **PASSED** | Dask Distributed y multiplicación por bloques 2D con reducción en árbol aprobados (ADR-001). |
| G3 | Provenance / Inputs | **PASSED** | Snapshots inmutables (`baselines/`) e inputs comunes certificados en `data/SHA256SUMS.txt`. |
| G4 | Protocol / Budget | **PASSED** | Protocolo experimental formalmente sincronizado y auditado en este documento. |
| G5 | Implementation | **PASSED** | Implementación Dask (`src/dask_matrix/`), CLI, watchdog, logger, sidecar y adaptadores runtime implementados y verificados con tests unitarios. |
| G6 | Correctness & Build | **PASSED** | Baselines compilados en `build/bin/`, ensamblado A1 (`-O3`) inspeccionado, y fixtures (`4×4` exactos `0.0`) + referencia $n=250$ (`max_abs_diff = 1.137e-13` en $b=250$ / `1.208e-13` en $b=100$, dentro de `atol=1e-9, rtol=1e-12`; diferencia de checksum `0.0`) verificados. |
| G7 | Environment / Topology | **PASSED** | Topología `1+p` PIDs en LocalCluster Windows, 1 hilo/worker, OpenBLAS a 1 hilo vía `threadpoolctl` y memoria base verificados. |
| G8 | Pilot & Full Experiment | **PASSED** | Pilot 1 (timeout 30 s) diagnosticado, Pilot 2 (6/6) y campaña Phase 8 (36/36, $p=1$; 48 filas históricas preservadas) más campaña autorizada DEC-01 de strong scaling local (`p ∈ {1,2,4,8,10}`, 90 filas anexadas) y pase diagnóstico separado ($n=1000$) registrados. |
| G9 | Data Audit & Analysis | **PASSED** | 138 registros brutos (48 históricos + 90 strong scaling) auditados; tablas derivadas, 3 figuras y `analysis_provenance.json` (13/13 SHA-256) generados y verificados por `scripts/analyze_campaign.py` y `scripts/verify_provenance.py`. |
| G10 | Final Audit | **PASSED** | Código, reproducibilidad en clon limpio, `README.md`, `report/Assignment04_Report.tex`/`.pdf` y especificaciones reconciliados. |

### 12.3 Secuencia posterior de fases y reglas de coherencia

1. **Resolución de DEC-01:** Registrar permiso docente cuando se disponga de él. Mantener `p>1` estrictamente como **BLOCKED** hasta entonces.
2. **Phase 4 (Prompt 4 — Implementación):** Estrictamente acotada a implementar el código de Assignment 4 en `src/`, escribir adaptadores runtime y realizar revisión estática de Makefiles. **Prohibido compilar código, ejecutar tests, lanzar pilots o realizar benchmarks.**
3. **Phase 5 (Prompt 5 — Validación de compilación y corrección):** Autoriza expresamente:
   - Compilar los snapshots inmutables de A1 y A3 hacia `build/bin/` bajo sus flags originales.
   - Generar e inspeccionar el ensamblado/desensamblado de A1 (`gcc -S` o `objdump -d`) para comprobar si el cálculo completo de $O(n^3)$ sobrevive a la optimización `-O3` o si el sumidero exclusivo en `c[0]` lo eliminó.
   - Ejecutar los fixtures de validación de corrección pequeños (identidad, cero, analítico 4x4, borde $n=4, b=3$, y $n=250$ frente a `C_250.csv`).
   - **Prohibido ejecutar benchmarks de rendimiento.**
4. **Phase 6 (Prompt 6 — Validación de entorno):** Verificación de procesos activos, mapeo de roles lógicos a PIDs únicos observados (Driver/Scheduler/Nanny en proceso padre, Worker en proceso hijo; 2 PIDs para $p=1$ en LocalCluster Windows), verificación dinámica de OpenBLAS a 1 hilo (`num_threads=1`, `nthreads=1` vía `threadpoolctl`; sin OpenMP cargado), y confirmación de `memory_limit=None` respaldado por el watchdog de memoria de 2.5 GiB sobre mediciones de memoria ociosa.
5. **Phase 7 (Prompt 7 — Piloto experimental):** Ejecución del piloto acotado bajo la especificación formal vinculante (§10.5): 6 muestras exclusivas de `dask_block_2d` ($n \in \{250, 500\}$, $p=1$, $b=250$, 1 warm-up + 2 repeticiones, timeout individual 120 s, 2.5 GiB RSS, tope global 15 min con interrupción segura). Su ejecución requiere autorización estudiantil independiente (Etapa 2).
6. **Phase 8 (Prompt 8 — Campaña experimental completa):** Ejecución de la batería de pruebas autorizada. **Bajo ninguna circunstancia se sustituirá $p=1$ por una configuración $p>1$ bloqueada pretendiendo que se ha realizado strong scaling. Si $p>1$ sigue bloqueado, strong scaling se reporta con rigor como BLOCKED / NOT COMPLETED.**
7. **Phases 9–10 (Prompts 9–12 — Análisis, Memoria y Auditoría):** Preservar en todo momento la distinción rigurosa entre estados `BLOCKED`, `NOT STARTED`, `PASSED`, `NOT RUN` y `NOT VERIFIED`.

---

## 13. Declaración histórica de no ejecución (Gate G4 / Phase 3)

La elaboración y aprobación inicial de este protocolo en Phase 3 (Gate G4) no implicaron:

- instalación, actualización o eliminación de dependencias;
- generación o regeneración de datasets;
- modificación de A1/A3, snapshots o datos congelados;
- compilación de ejecutables;
- ejecución de tests o validaciones numéricas;
- creación o arranque de clusters/workers;
- benchmarks, pilotos ni campañas de medición;
- modificación del repositorio Assignment04 o de su historial Git.

> **Nota de estado posterior (26-11-2026):** Esta sección conserva íntegramente el registro histórico de congelación inicial exigido en la aprobación de Phase 3 (`Gate G4`, 13-11-2026). Las fases posteriores debidamente autorizadas ya se han completado: compilación y validación de corrección (`Phase 5`, GCC 16.1.0), campaña Dask de un solo worker $p=1$ (`Phases 7–8`, 14-11-2026, 48 filas preservadas), campaña local multi-worker de strong scaling y diagnóstico separado autorizados el 23-11-2026 bajo `docs/decisions/DEC-01-instructor-permission.md` (90 filas anexadas), y ejecución controlada con matrices comunes `data/dense/` de los baselines congelados A1 y A3 (26-11-2026, `results/raw/a1_common_input_raw.csv` y `results/raw/a3_common_input_raw.csv`). La ejecución multinodo física continúa **BLOCKED / NOT AVAILABLE** y el weak scaling permanece aplazado (`DEC-07`).

---

## Referencias técnicas Dask consultadas

- Dask Distributed, *API Reference*: `LocalCluster`, `Client.get_task_stream`, `Client.nbytes` y `performance_report`. https://distributed.dask.org/en/stable/api.html (consultado el 13-11-2026).
- Dask Distributed, *Worker Memory Management*: límites por worker, managed memory, spill, pause y terminación/reinicio. https://distributed.dask.org/en/stable/worker-memory.html (consultado el 13-11-2026).
- Dask, *Diagnostics (distributed)*: capturas de `get_task_stream`, perfiles e informes de rendimiento. https://docs.dask.org/en/latest/diagnostics-distributed.html (consultado el 13-11-2026).
