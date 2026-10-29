# HF Recognition
Proyecto de Machine Learning - Programación III (ITLA)

Objetivo:
  Crear un algoritmo de reconocimiento de señas que tenga la capacidad de reconocer al menos 5 gestos distintos.
  Bono: Crear algoritmo de reconocimiento facial, que detecte y reconozca un rostro conocido.

## Módulos

- **Gestos de la mano**: cuenta los dedos dentro del recuadro (`0`–`5`) y reconoce el pulgar arriba (`OK`).
  - Busca el centro de la palma con una transformada de distancia (el círculo más grande que cabe en
    la mano) y cuenta los dedos que cruzan círculos alrededor de ella, con radios relativos al tamaño
    de la palma, así que no depende de la distancia a la cámara. Los tramos que apuntan hacia abajo son
    la muñeca, y un tramo muy ancho son dedos juntos.
  - El pulgar arriba es un único saliente hacia arriba mucho más ancho que un dedo en relación con la
    palma (el puño de lado es angosto y el pulgar es el dedo más grueso).
  - La versión de 2020 contaba los huecos entre dedos con defectos de convexidad y umbrales fijos en
    píxeles, por eso fallaba al cambiar la distancia o con los dedos juntos.
  - Sin calibrar, la mano se separa solo por color de piel (YCrCb), lo que falla si el fondo tiene un
    color parecido (madera, paredes naranjas, luz cálida).
  - Con **Calibrar fondo** (saca la mano del recuadro antes) se guarda cómo se ve el fondo, y la mano
    pasa a ser lo que cambió respecto a él y además tiene color de piel. El fondo se adapta poco a poco
    a los cambios de luz; si la cámara se mueve o la luz cambia mucho, recalibra.
  - El **umbral** controla cuánto debe cambiar un píxel: súbelo si aparece ruido, bájalo si se pierden dedos.
    Los cambios automáticos de exposición de la webcam se compensan, así que normalmente basta el valor por defecto.
  - El gesto mostrado solo cambia cuando el nuevo es mayoría en los últimos 8 frames, para que no parpadee.
  - **Modo gato**: opcional, muestra el gato de `img/cat_gestures` que corresponde al gesto (`fist` para 0,
    `1`–`5`, `2` o `peace-sign` para 2 y `ok` para el pulgar arriba).
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
