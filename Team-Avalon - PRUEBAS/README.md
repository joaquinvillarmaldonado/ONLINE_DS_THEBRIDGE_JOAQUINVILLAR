# Team Challenge: Hundir la flota 🛳️💥

### ¡Bienvenido a [Hundir la flota](https://instructions.hasbro.com/api/download/A3264_en-us_Battleship-Classic-Board-Game-Strategy-Game.pdf)!

Este es un juego clásico de estrategia naval cuya finalidad es hundir todos los barcos del oponente antes de que el oponente hunda los tuyos. 
Cada jugador tiene una cuadrícula secreta donde coloca sus barcos en posiciones estratégicas, de modo que el oponente no sepa su ubicación exacta.


## **Índice**   
1. [Caracteristicas](#Caracteristicas)
2. [Requisitios](#Requisitios)
3. [Instalacion](#Instalacion)
4. [Instruccion de uso](#Instruccion-de-uso)
5. [Estructura del codigo](#Estructura-del-codigo)
6. [contribuciones](#contribuciones)
7. [Autores](#Autores)

## Caracteristicas

- Modo de juego: Jugador vs Ordenador, jugador vs jugador (no se si hay) 
- representacion visual del tablero en la consola de tu pc 
- Mensajes indicativos para guiar e informar al jugador durante la partida

## Requisitios 

Para poder ejectuar este programa necesitas:

- Tener instalado Python 3.x
- Tener instalado Numpy

## Instalacion 

- clonar el repositorio o descargar el archivo [aqui](https://github.com/rafaneda7/Team-Challenge---Team-Avalon)
- ejecutar el programa:
python main.py 

## Instruccion de uso 

1. Al iniciar el programa te pedira que introduzcas un nombre, si decides no hacerlo el juego te llamara automaticamente jugador_1.
2. Se generara un tablero de (10,10) para ti y para el ordenador con los barcos colocados aleatoriamente.
3. Te tocara elegir una posicion a la que quieres disparar, si aciertas el disparo te tocara otra vez, sino te mandara un mensaje la consola y te dira que le as dado al agua y sera el turno del ordenador.
4. El juego terminara cuando un jugador hunda todos los barcos del oponente, o si no quieres terminar la partida escribiendo Exit acabara la partida.

## Estructura del codigo

- [*main.py*](https://github.com/rafaneda7/Team-Challenge---Team-Avalon/blob/main/main.py): Contiene la logica principal del codigo y de como funciona. 
- [*funciones.py*](https://github.com/rafaneda7/Team-Challenge---Team-Avalon/blob/main/funciones.py): Contiene la estructura de las funciones del programa. 
- [*clases.py*](https://github.com/rafaneda7/Team-Challenge---Team-Avalon/blob/main/clases.py): podemos ver todo lo que tiene que ver con la creacion del tablero, colocacíon de barcos, como diparar, etc.
- [*variables.py*](https://github.com/rafaneda7/Team-Challenge---Team-Avalon/blob/main/variables.py): Contiene el nombre de las varieables utilizadas. 

## Ejemplo

                    Tu tablero:                                       Oponente:       
                ----------------------                         ---------------------- 
                | 0 1 0 1 1 0 0 1 1 1|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 0 0 0 0 0 0 0 0 0 0|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 0 0 1 1 1 1 1 0 0 0|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 1 0 0 0 0 0 0 0 0 1|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 1 0 0 0 0 0 0 1 0 0|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 1 0 0 1 0 0 0 1 0 0|                         | 0 0 0 0 0 0 0 0 0 0| 
                | 0 0 1 0 0 1 1 0 0 0|                         | 0 0 0 0 0 0 0 0 0 0| 
                ----------------------                         ---------------------- 

Los "0" en el tablero representan a los espacios vacios y los "1" representan los barcos
Cuando se dispara, si se impacta en un barco enemigo, se marcará con un numero "2" en el tablero. Si el disparo cae en el agua, se marcará con "3".

Según vaya avanzando la partida, los tableros se irán viendo de la siguiente manera: 

                    Tu tablero:                                       Oponente:       
                ----------------------                         ---------------------- 
                | 0 2 0 1 1 0 0 1 1 1|                         | 2 2 0 0 0 0 0 0 0 0| 
                | 0 0 0 0 0 0 0 0 0 0|                         | 0 0 0 0 0 0 2 2 2 0| 
                | 0 0 1 2 2 2 1 0 0 0|                         | 0 0 0 3 0 0 0 0 0 0| 
                | 1 0 3 0 0 0 0 0 0 1|                         | 3 0 0 0 0 0 0 0 0 0| 
                | 1 0 0 0 3 0 0 2 0 0|                         | 0 2 0 0 0 3 0 0 0 0| 
                | 1 0 3 1 0 0 0 1 0 0|                         | 0 2 0 0 0 0 0 0 3 0| 
                | 0 0 1 0 0 1 1 0 0 0|                         | 0 3 0 0 0 0 0 0 0 0| 
                ----------------------                         ---------------------- 

## contribuciones 

Las contribuciones son bienvenidas. Si encuentras un error o tienes una sugerencia, crea un issue.

¿Cómo crear un issue en GitHub?
Ve al repositorio del proyecto en GitHub.
Haz clic en la pestaña Issues.
Presiona el botón New Issue.
Completa el formulario con el título y la descripción, y envíalo.

Aquí tienes una pequeña [presentación](https://www.canva.com/design/DAGWBu2P05w/ZN4MPLTiatlBD0-M7gpbtA/edit) del trabajo

## Autores

- [MiguelAngel120](https://github.com/MiguelAngel120)
- [joaquinvillarmaldonado](https://github.com/joaquinvillarmaldonado)
- [MarcoFuchs98](https://github.com/MarcoFuchs98)
- [Johannkarl](https://github.com/Johannkarl)
- [rafaneda7](https://github.com/rafaneda7)
