# Manifiesto de Proveniencia de Baselines y Dataset Común (Fase 2)
**Assignment 4: Distributed Matrix Multiplication**
**Curso:** BIG DATA · Grado en Ciencia e Ingeniería de Datos · ULPGC (2026–2027)
**Fecha de Materialización:** 13 de noviembre de 2026

---

## 1. Proveniencia de Repositorios Originales de Referencia

Ambos repositorios externos fueron confirmados por el estudiante e inspeccionados en modo estrictamente de solo lectura en las rutas locales:
- `external_repos/Assignment01/` (URL: `https://github.com/abelcasv/Assignment01_Abel.git`)
- `external_repos/Assignment03/` (URL: `https://github.com/abelcasv/Assignment03_Abel.git`)

### 1.1 Commits de Origen Confirmados
- **Assignment 1:** `90681e85949b6d65bdbcfef702b6306b0003f971` (HEAD en `main`, 16 de octubre de 2026)
- **Assignment 3:** `23cffbc82c155b6c3de24b3c149b509b63fc870b` (HEAD en `main`, 12 de noviembre de 2026)

Ambos árboles de trabajo de los repositorios externos permanecen estrictamente limpios y no han sido modificados.

---

## 2. Inventario de Snapshots Congelados en Assignment04

Para cumplir con la directiva de no introducir submódulos Git ni historiales completos, se han extraído snapshots congelados (*frozen vendor snapshots*) en el directorio `baselines/`.

| Snapshot Local | Archivo de Origen en Repo Externo | Commit SHA de Origen | SHA-256 Checksum del Archivo | Verificación Byte a Byte |
|---|---|---|---|---|
| `baselines/a1_sequential/benchmark.c` | `external_repos/Assignment01/src/c/benchmark.c` | `90681e85949b6d65bdbcfef702b6306b0003f971` | `4893fc553e97a0ec596c3c1cf9d5d8f582fc49308fc5971b338f48fd0268236f` | **IDÉNTICO** |
| `baselines/a1_sequential/Makefile` | `external_repos/Assignment01/src/c/Makefile` | `90681e85949b6d65bdbcfef702b6306b0003f971` | `b35b8cf8671d4cb0bdc42b59f0708dd8e55ebe51842034d3046c60da82f8c31f` | **IDÉNTICO** |
| `baselines/a3_openmp/parallel_benchmark.c` | `external_repos/Assignment03/src/parallel_benchmark.c` | `23cffbc82c155b6c3de24b3c149b509b63fc870b` | `6da20a3aed0862adc296fe9f4c8625bc51cd777959a5403088f7960c8741dbc8` | **IDÉNTICO** |
| `baselines/a3_openmp/Makefile` | `external_repos/Assignment03/src/Makefile` | `23cffbc82c155b6c3de24b3c149b509b63fc870b` | `474e4d2f2b57a5696bd8202ac0e125b101dbe6bcd5f58bd92df2a23f8a37e364` | **IDÉNTICO** |

---

## 3. Especificación Técnica de los Baselines Reutilizados

### 3.1 Baseline Secuencial Canónico (Assignment 1)
- **Función Reutilizada:** `multiply(const double* a, const double* b, double* c, int n)` ([benchmark.c](file:///c:/Users/abelc/OneDrive/Documentos/Universidad%2026-27/BD/Assignment04_Abel/baselines/a1_sequential/benchmark.c#L48-L59)).
- **Algoritmo:** Triple bucle anidado $i-j-k$ canónico sin optimizaciones de caché ($O(n^3)$ operaciones aritméticas).
- **Tipo Numérico y Disposición:** IEEE 754 `binary64` (`double`), arreglo 1D contiguo en memoria dinámica indexado como `matrix[i * n + j]`.
- **Opciones de Compilación (`CFLAGS`):** `-Wall -Wextra -O3 -std=c11`
- **Opciones de Enlace (`LDFLAGS` / `LDLIBS`):** `-lpsapi` (Windows PSAPI para Working Set RSS).
- **Límite Temporal:** Cronometraje exclusivo del kernel `multiply()` mediante `QueryPerformanceCounter`. Carga de datos y asignación fuera del timer. Consumo de salida mediante `volatile double result_sink = c[0]`.

### 3.2 Baseline Paralelo OpenMP y Secuencial Emparejado (Assignment 3)
- **Funciones Reutilizadas:**
  1. `seq_multiply_ikj(const double* a, const double* b, double* c, int n)`: Bucle $i-k-j$ optimizado con `#pragma GCC ivdep` en $j$ para acceso contiguo a memoria en $B$ y $C$ (stride 1). Autovectorizado por GCC.
  2. `omp_multiply_ikj(const double* a, const double* b, double* c, int n, int threads)`: Kernel paralelo primario que paraleliza el bucle exterior $i$ mediante `#pragma omp parallel for schedule(static) num_threads(threads)`.
  3. `omp_multiply_tiled_b32()`: Kernel paralelo secundario con bloqueo de caché ($32 \times 32$).
  4. `multiply_ijk()`: Triple bucle canónico de referencia.
- **Opciones de Compilación (`CFLAGS`):** `-Wall -Wextra -O3 -std=c11 -fopenmp -march=native`
- **Opciones de Enlace (`LDFLAGS` / `LDLIBS`):** `-lpsapi` (en Windows), `-lm` (en POSIX).
- **Límite Temporal:** Cronometraje del kernel incluyendo `memset(c, 0, ...)` y la gestión del equipo de hilos OpenMP. Consumo de salida mediante suma acumulada de todos los elementos en `volatile double result_sink`.

---

## 4. Adaptadores de Compatibilidad en Runtime (Sin Modificación de Fuentes)

Dado que los ejecutables originales buscan rutas fijas relativas a su argumento raíz, se han implementado adaptadores externos que generan directorios de ejecución independientes.

### 4.1 Adaptador de Runtime para Assignment 1 (`scripts/prepare_a1_runtime.py`)
- **Ruta Objetivo:** `build/a1_runtime/`
- **Estructura Creada:** `build/a1_runtime/data/`
- **Archivos Suministrados:**
  - Matrices densas comunes: `A_100.csv`, `B_100.csv`, `A_250.csv`, `B_250.csv`, `A_500.csv`, `B_500.csv`, `A_750.csv`, `B_750.csv`, `A_1000.csv`, `B_1000.csv`.
  - Fixtures de validación originales de A1: `identity_4.csv`, `zero_4.csv`, `known_A_4.csv`, `known_B_4.csv`, `expected_C_4.csv`.
- **Mecanismo de Ejecución:** El ejecutable original de A1 se invoca pasando `build/a1_runtime` como `argv[1]`:
  ```powershell
  ./baselines/a1_sequential/benchmark.exe ../../build/a1_runtime
  ```
- **Tamaños Fijos y Dimensión Auxiliar $n=100$:**
  - El ejecutable de A1 tiene fijada en su código la lista `SIZES = {100, 250, 500, 750, 1000}`.
  - La dimensión $n=100$ se incluye como dimensión auxiliar de compatibilidad para permitir que el ejecutable de A1 complete su ciclo sin errores.
  - **Identificación Expresa:** La dimensión $n=100$ **no forma parte de la comparación común de tres vías** (A1/A3/Dask), ya que A3 no contempla $n=100$ en su protocolo.
  - A1 no ejecutará $n=1500$ ni $n=2000$ debido a que su harness original limita estrictamente el barrido hasta $n=1000$.

### 4.2 Adaptador de Runtime para Assignment 3 (`scripts/prepare_a3_runtime.py`)
- **Ruta Objetivo:** `build/a3_runtime/`
- **Estructuras Creadas:**
  - `build/a3_runtime/data/dense/`
  - `build/a3_runtime/data/validation/`
- **Archivos Suministrados:**
  - Matrices densas comunes: `A_<n>.csv` y `B_<n>.csv` para $n \in \{250, 500, 750, 1000, 1500, 2000\}$.
  - Fixtures de validación exactos de A3: `identity_4.csv`, `zero_4.csv`, `known_A_4.csv`, `known_B_4.csv`, `expected_C_4.csv` (hashes certificados en `A3_RUNTIME_MANIFEST.txt`).
- **Mecanismo de Ejecución:** El ejecutable original de A3 se invoca pasando `build/a3_runtime` como argumento:
  ```powershell
  ./baselines/a3_openmp/parallel_benchmark.exe ../../build/a3_runtime
  ```

---

## 5. Especificación y Verificación del Dataset Común

El generador determinista ([scripts/generate_dataset.py](file:///c:/Users/abelc/OneDrive/Documentos/Universidad%2026-27/BD/Assignment04_Abel/scripts/generate_dataset.py)) implementa las siguientes reglas:
1. **Semilla Aislada por Dimensión:** $\text{seed}(n) = 42 + n$, utilizando `random.Random` de la biblioteca estándar de Python (Mersenne Twister MT19937). Esto desacopla las dimensiones, evitando que la adición o eliminación de un tamaño altere los números pseudoaleatorios de las demás matrices.
2. **Rango de Valores:** $[0.0, 1.0)$ generado mediante `rng.random()`.
3. **Orden de Generación:** $A$ en orden *row-major*, seguido de $B$ en orden *row-major*.
4. **Formato Numérico y Precisión:** CSV sin cabeceras, valores separados por comas, formateados estrictamente con **`%.17g`** para preservar la fidelidad binaria exacta de 64 bits IEEE 754.
5. **Finales de Línea:** Unix LF (`\n`) explícito y reproducible en cualquier sistema operativo.
6. **Referencia Numérica Independiente ($n=250$):**
   - Archivo: `data/validation/reference/C_250.csv`.
   - Algoritmo: Triple bucle estándar en Python puro ejecutado sobre los números de coma flotante parseados desde los CSVs de entrada.
   - **Independencia:** No utiliza NumPy, BLAS ni Dask, constituyendo una referencia numérica independiente de la implementación distribuida. No se etiqueta como "analítica" porque proviene de números pseudoaleatorios.

---

## 6. Inventario Completo y Checksums SHA-256 (`data/SHA256SUMS.txt`)

Todos los archivos generados fueron certificados con SHA-256 en [data/SHA256SUMS.txt](file:///c:/Users/abelc/OneDrive/Documentos/Universidad%2026-27/BD/Assignment04_Abel/data/SHA256SUMS.txt) y verificados de forma independiente por [scripts/verify_dataset.py](file:///c:/Users/abelc/OneDrive/Documentos/Universidad%2026-27/BD/Assignment04_Abel/scripts/verify_dataset.py):

| Archivo Relativo | Rol en Assignment 4 | Dimensiones | SHA-256 Checksum |
|---|---|---|---|
| `dense/A_100.csv` | Compatibilidad Auxiliar A1 | $100 \times 100$ | `99c7730a58a915c01b2967b21d9bd3cd61f5e20278ea0f12bc30e0b679b28b36` |
| `dense/B_100.csv` | Compatibilidad Auxiliar A1 | $100 \times 100$ | `cc53ac9fef8c85356e547c0eb49841a680bd4ffc73b3ae3cc85d4bf3bf67bf51` |
| `dense/A_250.csv` | Comparación Común 3-Vías | $250 \times 250$ | `8fc7f538e6047d2fc11d5076154d106935ebd6a34a6ccc67e8c13536016c0d40` |
| `dense/B_250.csv` | Comparación Común 3-Vías | $250 \times 250$ | `0002373f420cd18209555e5d14557d2d0cbebc4da3a42f872c0886a79f8067c3` |
| `dense/A_500.csv` | Comparación Común 3-Vías | $500 \times 500$ | `f27913d865d29276ef24f53f5a6862447b330e7f888182aff0246a0532acf756` |
| `dense/B_500.csv` | Comparación Común 3-Vías | $500 \times 500$ | `dee2a7abadff44e7bda78f877eff28914d9210c7a6525d540f37b66db2067c5b` |
| `dense/A_750.csv` | Comparación Común 3-Vías | $750 \times 750$ | `3087c1228c76d369ed34bbb2f99b327ebb62894ef2103cb05e43f43351f9bc05` |
| `dense/B_750.csv` | Comparación Común 3-Vías | $750 \times 750$ | `d99bbcb0c0000343f921822ebbdc1ea10999dd4a4e4a31e67af4520cd8783c81` |
| `dense/A_1000.csv` | Comparación Común 3-Vías | $1000 \times 1000$ | `1727aa2f3e620fe5e83e523d990f2141e6dc3d5524dc3332ec73965b7f2afa91` |
| `dense/B_1000.csv` | Comparación Común 3-Vías | $1000 \times 1000$ | `b68ebbd44e43933c01afcdd6da96560083c76012aecfb871440efe900abf6062` |
| `dense/A_1500.csv` | Escalabilidad Distribuida (A3/Dask) | $1500 \times 1500$ | `f69b2ddd3625383740af8c056a28c84a000bf4f8f6ed279a0143647875ac15de` |
| `dense/B_1500.csv` | Escalabilidad Distribuida (A3/Dask) | $1500 \times 1500$ | `9694e3268dd0cab5076835a0a36edfe246ed0944eee58792515efa05f49f8c6c` |
| `dense/A_2000.csv` | Escalabilidad Distribuida (A3/Dask) | $2000 \times 2000$ | `a840b5d73026670a511cbc72145e81f71c317fdda37d913a646685e329c3edc2` |
| `dense/B_2000.csv` | Escalabilidad Distribuida (A3/Dask) | $2000 \times 2000$ | `0e73154d2201c19196690485304e3c19c5d50f489192e02657df011d7caf6966` |
| `validation/identity_4.csv` | Fixture Propiedad Identidad ($A \times I = A$) | $4 \times 4$ | `1071e33f72efb352764e0ea040792695cd75360b1f1b292bf9c36059cd47f18f` |
| `validation/zero_4.csv` | Fixture Propiedad Nula ($A \times 0 = 0$) | $4 \times 4$ | `506351148090d688af818822486473480a012ee2c4e142312b7b4a8b99a56595` |
| `validation/known_A_4.csv` | Fixture Caso Analítico (Matriz A) | $4 \times 4$ | `45acf3b6c71a4be592d02bd6e221efe32512e7e3a35a9742453865876840b974` |
| `validation/known_B_4.csv` | Fixture Caso Analítico (Matriz B) | $4 \times 4$ | `280740971507426667e76a66c7bec90675edc61cfac696e1930bd9bbd4aab252` |
| `validation/expected_C_4.csv` | Fixture Caso Analítico (Producto Esperado) | $4 \times 4$ | `301e357c2477a2502677ff14790e970606f1d9251c2d7945a885a20b91995217` |
| `validation/reference/C_250.csv` | Referencia Numérica Independiente | $250 \times 250$ | `598e9a9fc8f0c8db1d5da277f4d93561ad83e2df77e3fcb5f968f67b8033909e` |
