# HF Recognition
Proyecto de Machine Learning - Programación III (ITLA)

Objetivo:
  Crear un algoritmo de reconocimiento de señas que tenga la capacidad de reconocer al menos 5 gestos distintos.
  Bono: Crear algoritmo de reconocimiento facial, que detecte y reconozca un rostro conocido.

## Módulos

- **Gestos de la mano**: cuenta los dedos dentro del recuadro y reconoce `0`–`5`, `OK` y "Suerte"
  mediante segmentación de la mano, convex hull y defectos de convexidad.
  - Sin calibrar, la mano se separa solo por color de piel (YCrCb), lo que falla si el fondo tiene un
    color parecido (madera, paredes naranjas, luz cálida).
  - Con **Calibrar fondo** (saca la mano del recuadro antes) se guarda cómo se ve el fondo, y la mano
    pasa a ser lo que cambió respecto a él y además tiene color de piel. El fondo se adapta poco a poco
    a los cambios de luz; si la cámara se mueve o la luz cambia mucho, recalibra.
  - El **umbral** controla cuánto debe cambiar un píxel: súbelo si aparece ruido, bájalo si se pierden dedos.
- **Reconocimiento facial**: identifica rostros en una foto o con la webcam comparándolos con las
  personas de `img/known`. El nombre del archivo es el nombre de la persona; para agregar a alguien
  basta con copiar su foto en esa carpeta y pulsar «Recargar». Las imágenes analizadas se guardan en
  `scannedImages/`.

## Instalación

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
pip install --no-deps face_recognition
```

`face_recognition` se instala con `--no-deps` porque depende de `dlib`, que en Windows hay que compilar;
`requirements.txt` instala en su lugar `dlib-bin`, que viene precompilado.

> En Windows, si pip falla con *"The filename or extension is too long"*, activa las rutas largas
> o crea el entorno virtual en una ruta corta (por ejemplo `C:\venvs\hfr`).

## Uso

```bash
python gui/start.py
```

En Windows también se puede abrir `HF Recognition.bat`. Cada módulo se puede ejecutar por separado
(`python gui/handHandler.py`, `python gui/faceHandler.py`), y en `scripts/` hay versiones de consola que
solo usan ventanas de OpenCV.
