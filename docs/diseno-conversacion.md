# Lunes: cómo conversa

Este documento fija cómo debe sentirse hablar con Lunes y cómo se construye por dentro para lograrlo. Es la
referencia para reescribir la capa de conversación: si algo no está aquí, se discute antes de programarlo.

## 1. Lo que buscamos

Lunes es un compañero de banco, no un buscador por voz. Hablar con él tiene que sentirse como hablar con un colega
que está mirando la misma placa: reacciona en el momento, sabe en qué está, dice lo que ve y lo que duda, y sigue la
conversación mientras trabaja.

Metas medibles (se prueban con un script antes de cada sesión con Erlan):

| Qué | Meta |
|---|---|
| Desde que Erlan termina de hablar hasta que Lunes reacciona (sonido o palabra) | ≤ 0.4 s |
| Hasta la primera frase con contenido, en una charla o pregunta simple | ≤ 1.5 s |
| Hasta que se calla cuando Erlan lo interrumpe | ≤ 0.5 s |
| Palabra de activación | solo para empezar una sesión; después, conversación continua |
| Mientras trabaja con la CNC | sigue escuchando y respondiendo preguntas cortas |

Lo medido en el banco el 28 de octubre (RTX 3090, internet de casa en Cochabamba):

| Pieza | Tiempo |
|---|---|
| Muse Glimmer, primera carga en la GPU | 50 s (por eso queda cargado todo el día) |
| Muse Glimmer, respuesta libre corta | 2 s: razona unos 65 tokens a escondidas aunque se le pida no pensar |
| Muse Glimmer, con el formato fijo del conductor (qué decir + trabajo) | 0.9 a 1.5 s; decidió bien 6 de 6 casos |
| Muse Glimmer, decidiendo con llamadas a herramientas | 5.3 s |
| Voz MMS, una frase corta | 0.45 s (1.1 s para 3.6 s de audio) |
| Muse Voice Transcribe por lotes, una frase de 4.4 s | 1.9 a 2.1 s después de terminar la frase |
| Omnilingual ASR 1B int8 en el equipo (CPU, 8 hilos) | 0.4 a 0.7 s para "Lunes, para"; 1.2 a 2.4 s para frases de 4 a 7 s; el que mejor oye nombres y números ("pin 4 del tp405 6") |
| El mismo modelo int8 en la GPU (CUDA) | más lento, 4.4 s para 7 s: sus capas int8 se quedan en la CPU |
| Omnilingual ASR 300M en la CPU, 4 hilos (29 de octubre, por si no hay GPU) | 0.2 s para "Lunes, para"; 0.5 s para 3 s de audio y 1.1 s para 6 s |
| Omnilingual ASR 300M de precisión completa en la GPU | 0.03 a 0.3 s para cualquier frase; usa 2.3 GB de la GPU (con Glimmer quedan 21.2 de 24 GB); se equivoca más en nombres ("lúnese", "tp celo 56") |
| Glimmer cuando dos llamadas piden distinto tamaño de contexto | Ollama lo recarga: 22 a 24 s cada vez; por eso conductor y trabajador usan el mismo (8192) |
| Muse Spark, charla corta | 2.4 s |
| Muse Spark, identificar o diagnosticar buscando en internet | 14 a 16 s |
| Conductor con criterio (3.4) en Muse Spark, esfuerzo mínimo | 1.5 a 2.2 s de promedio; 19 o 20 de las 20 frases de prueba bien |
| El mismo conductor en Muse Glimmer | 1.8 s, pero dijo "te veo por el Kinect" sin haber mirado; pensando un poco, 6.3 s |
| Todo en Glimmer (sin Spark, por el tope diario) | un movimiento simple tardó 170 s y el conductor esperó de 11 a 53 s detrás del trabajador; dijo "ya moví la máquina" antes de moverla |
| Calibración del Kinect (9 posiciones, hasta 30 mm) | 54 s; error típico 1.3 mm al ubicar el cabezal |

Con esas piezas, a "Lunes, revisemos esta placa" le sigue la frase del conductor ("¿le echo una mirada rápida o la
reviso entera?") a unos 3 s de que Erlan terminó de hablar: 0.6 s para saber que terminó, 0.3 s del oído rápido,
1.5 a 2.2 s del conductor y 0.45 s de la voz; si tarda, antes suena "a ver" o "mmm".

## 2. Principios

1. **Presencia antes que perfección.** Una reacción inmediata vale más que una respuesta perfecta que llega tarde.
   Lunes nunca deja un silencio muerto: o habla, o muestra qué está haciendo.
2. **Honestidad (Hume).** Lunes dice lo que observó, lo distingue de lo que supone y de lo que sabe en general, y
   nunca finge capacidades. Su seguridad o su duda se notan en cómo habla.
3. **Carácter, no reglas.** La personalidad vive en un texto corto que describe quién es. Las garantías de seguridad
   viven en el código (límites de la máquina, Z que nunca baja del foco), no en el prompt.
4. **Lo difícil en el código.** Medir, recorrer, alinear, ubicarse y enfocar son herramientas deterministas. El
   modelo decide qué hacer y cómo contarlo; no calcula lo que el código ya sabe calcular.
5. **Cada turno deja rastro.** Qué oyó, qué entendió, qué decidió, qué hizo y cuánto tardó cada parte.
6. **Solo modelos de Meta.** Todo modelo que Lunes ejecuta mientras trabaja es de Meta, por la API o con pesos
   abiertos en el equipo. Lo que no es un modelo (detectar voz por la energía de la señal, descartar su propio
   audio, la geometría del mosaico) es código propio.

## 3. Capas

Hablar con Lunes es como estar en una llamada. El **conductor** (Muse Spark en esfuerzo mínimo; Muse Glimmer en el equipo sin internet) está siempre en línea:
escucha, contesta enseguida y, cuando Erlan pide un trabajo, se lo pasa al **trabajador** (Muse Spark con las
herramientas del banco) y le va contando cómo va. El conductor no hace el trabajo ni inventa resultados: solo sabe lo
que el trabajador publica.

```
  oído continuo ──► conductor (Muse Spark; Glimmer sin internet) ──► voz
       │                 │                 ▲
       │                 │ trabajo         │ qué contar y cuándo
       │                 ▼                 │
       │            trabajador (Muse Spark + CNC, microscopio, Kinect) ── eventos de progreso
       └── "para" ──► detener todo, al instante
```

### 3.1 Oído continuo
- Escucha siempre. Detecta que alguien habla por la energía de la señal (código, no un modelo).
- Transcribe el oído local de Meta (Omnilingual ASR, en el equipo): nada de lo que se dice en el cuarto sale del
  equipo.
- "Lunes" abre la conversación. Queda abierta mientras hablan, mientras hay un trabajo en curso y hasta 90 s después
  del último intercambio; dentro de ella todo lo que dice Erlan va al conductor, que juzga qué es para Lunes. Lo que se
  oye encima de la voz de Lunes solo cuenta si es "para" o si lo nombran: casi siempre es su propio eco.
- Son dos oídos con el mismo modelo en dos tamaños. El rápido (300M, en la GPU) oye todo y decide al instante: si
  es "para", si lo llamaron y qué contestar. El preciso (1B, en la CPU) vuelve a oír lo que fue para Lunes y le pasa
  al trabajador las palabras exactas del pedido, donde importan los números de parte.
- Muse Voice Transcribe en tiempo real, en el modo que marca el principio y el fin de cada frase, queda como segundo
  oído para cuando el local no alcance (mucho ruido, nombres del banco).
- La propia voz de Lunes se descarta: Lunes sabe qué dijo y en qué momento, y quita esas palabras de lo que oye.
- "Para", "basta", "detente" se reconocen en el equipo, sin internet, con el oído local, y detienen voz y máquina al
  instante.
- Si hay otras personas en el cuarto, la diarización separa las voces y Lunes sigue a la voz que lo llamó.

### 3.2 Entender, no clasificar
Lunes no clasifica la frase en una lista de casos: entiende qué quiere Erlan. "¿Qué ves?" puede ser "¿funcionan tus
ojos?" o "dime qué hay en la placa"; un colega no sale a recorrer la placa entera por eso, pregunta o mira una vez.
Solo parar y despedirse se resuelven con reglas, porque tienen que funcionar al instante y sin internet.

Una primera versión decidía con reglas ("si pregunta qué ves, es un trabajo"): cada regla que arreglaba un caso
creaba otro comportamiento de máquina. Por eso las reglas de casos están prohibidas en el conductor; lo que lo
mejora es darle mejor contexto y mejores pautas, y se comprueba con las frases de prueba (3.4).

### 3.3 Reaccionar ya
Si el conductor tarda más de 0.9 s, suena "a ver" o "mmm", ya grabados en la voz de Lunes. Son neutros a propósito:
cubren el tiempo en que piensa sin prometer nada que quizá no haga (un "dale" antes de que decida preguntar sonaba
mecánico).

### 3.4 Conductor (capa rápida)
- Piensa con Muse Spark en esfuerzo mínimo por la API; sin internet, Muse Glimmer en el equipo, siempre cargado.
- Cada turno recibe:
  - quién es Lunes y cómo decide un buen colega: entender qué quiere Erlan, hacer lo mínimo que le responde,
    preguntar si puede querer dos cosas que cuestan muy distinto, hacer sin preguntar lo que pidió claro, no afirmar
    lo que no vio;
  - sus sentidos ahora, leídos en ese momento: si el micrófono oye, si el microscopio manda imagen y qué tan nítida,
    si el Kinect está conectado, dónde está la CNC, si Spark está disponible;
  - lo que cuesta cada cosa: contestar es al instante, mirar unos 10 s, un trabajo de uno a varios minutos;
  - el trabajo en curso y el banco, y los últimos turnos.
- Contesta en un formato fijo: primero qué entendió (eso lo hace pensar antes de decidir y queda en el registro),
  después qué decir y qué hacer: nada, mirar (una foto, sin mover la máquina) o un trabajo para el trabajador.
- Respuestas de una a tres frases, salvo que Erlan pida explicar.
- `tests/conductor_frases.json` tiene las frases de Erlan con lo que haría un buen colega, y
  `tools/test_conductor.py` las pasa todas. Todo cambio del conductor se mide ahí antes de probarlo con Erlan.
- Mientras hay un trabajo en curso, el conductor sigue la conversación: "¿cuánto te falta?", "¿qué encontraste?" se
  contestan con el estado, sin interrumpir al trabajador.

### 3.5 Trabajador (capa profunda)
- El agente con herramientas: observa, piensa, actúa y repite hasta cumplir el objetivo.
- Es Muse Spark por la API, así la GPU queda libre para el conductor; sin internet, Muse Glimmer.
- Corre en segundo plano. Publica eventos de progreso ("foto 12 de 52", "encontré el borde izquierdo") en el estado
  que lee el conductor.
- Para identificar o diagnosticar, busca en internet: hojas de datos, notas de aplicación, foros técnicos. (Pensar
  más sin buscar casi no mejora la respuesta y la hace bastante más lenta; buscar sí trae datos que el modelo no
  tenía.) Mientras busca dice qué está haciendo ("ya leí la marca, ahora confirmo en la hoja de datos").
  Cada dato que trae de afuera va con su fuente.
- Para encontrar componentes en un mosaico usa SAM 3.1, que devuelve el contorno de cada uno; después se acerca a
  cada uno con el microscopio.
- Al terminar entrega un resultado verificado, con la evidencia (mosaico, acercamiento, medida, fuente).
- Comprueba lo que hace con sus propios sentidos, no solo con lo que cree: después de cada movimiento, el Kinect
  dice dónde ve de verdad el cabezal (por el anillo de luz del microscopio) y lo compara con la posición que cuenta
  la máquina. Si no coinciden, Lunes se lo dice a Erlan y no se mueve más hasta que él decida: verificar nunca dispara
  otra acción por su cuenta, así no hay bucles. Lo mismo al arrancar y después de soltar los motores.
- Cuando Erlan lo corrige ("se movió hacia adelante, no hacia atrás"), guarda la corrección (hacia dónde va cada eje
  visto desde Erlan) una sola vez, reemplazando lo que sabía, y la usa desde ese momento.

### 3.6 Qué contar y cuándo
Lunes cuenta lo que hace como lo contaría una persona atenta, no como una ametralladora de avisos. Las reglas viven
en el código; el conductor solo pone las palabras.

1. **Habla en los hitos:** cuando empieza, cuando encuentra algo que importa, cuando algo sale distinto de lo
   esperado, cuando necesita una decisión de Erlan y cuando termina. El avance rutinario ("foto 12 de 52") se ve en la
   página en vivo y se dice solo si Erlan pregunta.
2. **Deja aire:** al menos 15 segundos entre dos avisos que Erlan no pidió. Si se juntan varios eventos, dice uno
   solo, el más reciente o el más importante; un aviso que ya quedó viejo no se dice.
3. **No pisa a nadie:** nunca habla mientras Erlan habla ni en el segundo y medio que sigue, y nunca corta su propia
   frase para meter un aviso.
4. **No deja silencios muertos:** si un trabajo lleva más de 40 segundos sin hitos, dice en una frase en qué va.
5. **Al terminar** resume el resultado verificado en una a tres frases; el detalle y la evidencia quedan en la
   página.
6. **Sabe callarse:** si Erlan dice que no hace falta que le cuente, Lunes solo habla en los hitos que necesitan su
   decisión y al terminar.

### 3.7 Voz y expresión
- Voz natural, en español, con entonación que acompañe el estado: segura, dudosa, concentrada.
- El modelo escribe para leer (negritas, listas, cifras); antes de hablar, la capa de voz quita el formato y pasa
  números, unidades y siglas a palabras.
- Estados visibles en la página en vivo y, en el banco, con el anillo de luz del microscopio: escuchando, pensando,
  mirando, moviendo, detenido.

### 3.8 Qué modelo hace cada trabajo

| Trabajo | Modelo de Meta | Dónde corre | Costo |
|---|---|---|---|
| Conductor: entender, conversar, decidir qué hacer, contar el avance | Muse Spark 1.3, esfuerzo mínimo; Muse Glimmer sin internet | API; equipo (RTX 3090) | menos de medio centavo por turno |
| Trabajador: el agente con herramientas | Muse Spark 1.3, con búsqueda web para identificar y diagnosticar | API (Responses) | US$1.25 / 4.25 por millón de tokens (entrada / salida), más US$2.50 por 1000 búsquedas |
| Trabajador sin internet y modo privado | Muse Glimmer | equipo | gratis |
| Oír todo y decidir al instante | Omnilingual ASR 300M (pesos abiertos de Meta) | equipo (GPU) | gratis |
| Las palabras exactas de un pedido | Omnilingual ASR 1B (pesos abiertos de Meta) | equipo (CPU) | gratis |
| Segundo oído, para ruido y nombres | Muse Voice Transcribe en tiempo real | API | US$0.18 por hora de audio |
| Separar voces cuando hay más gente en el cuarto | Muse Voice Transcribe, modo diarización | API | igual |
| Encontrar y recortar componentes en un mosaico; seguir la punta del multímetro en video | SAM 3.1 | API | US$2.50 por 1000 imágenes; US$0.20 por 1000 cuadros de video |
| Entender un video corto (un LED que parpadea, una soldadura) | Muse Wonder | API | US$4 / 20 por millón de tokens |
| Hablar | MMS-TTS en español | equipo | gratis |

Decidido y por decidir:
- **Lunes no aprende la voz de Erlan.** Atiende a quien lo llama por su nombre; si hay más gente, sigue a la voz que
  lo llamó (3.1).
- **La voz.** MMS es un modelo abierto de Meta; si no se acepta como tal, la alternativa es la voz del sistema
  operativo, que no es un modelo.

## 4. Estado interno

Lunes lleva un estado pequeño y explícito que la capa rápida siempre recibe:

- **Atención:** qué está mirando (placa, componente, cara de la placa) y desde cuándo.
- **Seguridad:** qué tan seguro está de lo último que afirmó, según la evidencia (vio / leyó en una fuente / supone /
  sabe en general).
  Con poca seguridad, pregunta o propone mirar; con mucha, afirma.
- **Trabajo:** tarea en curso, paso actual, cuánto falta.
- **Conversación:** qué se está hablando, qué quedó pendiente.

El estado cambia el tono, no los hechos: Lunes nunca inventa para sonar seguro.

## 5. Contexto y memoria

| Pieza | Qué contiene | Cuándo se usa |
|---|---|---|
| `alma` | quién es Lunes, cómo habla, qué valora (≤ 1 página) | en cada turno |
| contexto del banco | la máquina, sus límites, sus herramientas, el microscopio, el Kinect | en cada turno, resumido |
| memoria Hume | hechos observados con su evidencia, y las correcciones de Erlan | se buscan los relevantes |
| procedimientos | pasos para trabajos que se repiten: identificar una placa, leer un integrado, diagnosticar un cargador | cuando la tarea coincide |
| documentos | hojas de datos y notas guardadas en el equipo, y lo que encontró en internet con su fuente | se consultan cuando hace falta; lo que ya se buscó no se vuelve a buscar |

- Lo que siempre va en el prompt es corto. Lo demás se trae cuando hace falta.
- Cuando Erlan corrige a Lunes, la corrección se guarda en la memoria con su evidencia, y Lunes la aplica desde ese
  momento: cada corrección lo hace mejor.
- Un procedimiento nuevo nace de una tarea que salió bien: los pasos se guardan para repetirla sin volver a
  explicarla.

## 6. Registro de cada turno

Cada turno deja una línea de eventos con tiempos, visible en la página en vivo:

```
oído      "Lunes, ¿qué integrado es ese?"     0.6 s
intención pregunta sobre lo visto             0.2 s
reacción  "a ver"                             0.1 s
decisión  inspeccionar componente en (0.44, 0.80)
acción    acercamiento de 4 fotos             38 s
respuesta "Leo LA7770; el último carácter no se ve claro"   1.2 s
```

Sirve para encontrar dónde se va el tiempo, para mostrar cómo piensa Lunes y para corregir sin adivinar.

## 7. Ajuste fino (más adelante)

Los hechos (datos de un integrado, de una placa) se consultan en documentos, no se entrenan. El ajuste fino de
Muse Glimmer con adaptadores (LoRA) puede servir más adelante para el estilo de Lunes, con conversaciones reales
del registro:
- La calidad de los datos importa más que la cantidad.
- Un conjunto solo de preguntas y respuestas cortas vuelve al modelo escueto: hay que mezclar respuestas cortas y
  explicaciones largas revisadas.
- Se compara siempre contra el modelo sin ajustar, con las mismas preguntas.

## 8. Orden de construcción

1. Medir el conductor, la voz y los oídos (hecho, sección 1).
2. Solo modelos de Meta (hecho): el oído detecta la voz por energía y descarta la propia voz de Lunes por sus
   palabras.
3. Oído continuo con los dos oídos locales de Meta (hecho: el rápido en la GPU, el preciso en la CPU). Falta probarlo
   con la voz real de Erlan (las pruebas fueron con la voz de Lunes, que sale distinta en cada frase) y Voice
   Transcribe en tiempo real como tercer oído para mucho ruido.
4. Conductor con criterio, sus sentidos reales, la conversación abierta y las frases de prueba (hecho:
   `lunes/conductor.py`, `tests/conductor_frases.json`, `tools/test_conductor.py`).
5. Trabajador en segundo plano con eventos de progreso, cola de trabajos y las reglas de qué contar y cuándo (hecho
   en `tools/talk.py`; probado con un trabajador simulado, falta probarlo en el banco).
6. Alma, contexto del banco, procedimientos y memoria de correcciones.
7. Pruebas contra las metas de la sección 1 antes de cada sesión con Erlan.
