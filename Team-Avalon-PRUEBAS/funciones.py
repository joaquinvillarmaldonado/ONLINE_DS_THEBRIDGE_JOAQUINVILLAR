import numpy as np
import random

# Se inicia el juego con un mensaje de bienvenida explicando las intrucciones de como jugar al usuario mediante un print.
def welcome():
    print(f" \
Bienvenido a la version del popular juego BattleShip (Hundir la flota) del Team Avalon\n\n \
¿No sabes como jugar? Tranquilo! Aca te lo explicamos paso a paso: \n \
    1- Para iniciar la partida deberás introducir el nombre con el que quieres jugar, sino se iniciara con uno por defecto\n \
    2- Deberás elegir en que coordenadas quieres colocar cada uno de tus barcos. Dispones de 4 tipos de barcos: \n \
       2.1- Lancha:      Con una longitud de 1 espacio,  tienes para colocar 4\n \
       2.2- Submarino:   Con una longitud de 2 espacios, tienes para colocar 3\n \
       2.3- Destructor:  Con una longitud de 3 espacios, tienes para colocar 2\n \
       2.4- Portaviones: Con una longitud de 4 espacios, tienes para colocar 1\n \
    3- ¡Como jugador tienes el primer turno! introduce las coordenadas de tu disparo si aciertas puedes disparar nuevamente sino el turno pasa al oponente.\n \
    4- El juego acaba cuando alguno de los jugadores se queda sin barcos en el tablero.\n\n \
Puedes salir del juego en cualquier momento escribiendo el comando ______\n\n \
Esperamos lo disfrutes ¡Buena Suerte!")

# Mediante un input se le solicita al usuario que indique el nombre con el que quiere jugar
def player_name():
    name = input('Introduce tu nombre: ').strip() # Se utiliza el metodo .strip() de los Strings para eliminar los espacios del inicio y final del input 
    if name != "":                                # para evitar que el usuario inicie el juego con un nombre en blanco (Solo con espacios).
        return name
    else:
        return "Jugador_1" # Si el usuario no indica algun nombre, se iniciara el juego con un nombre predeterminado "Jugador_1". 



def shoot_player(tablero_jugador, tablero_oponente, size = (10,10)):

    """
    Función que simula el disparo de un jugador sobre el tablero del oponente.
    Valida las coordenadas y actualiza los tableros según si el disparo impacta un barco o va al agua.
    
    Parámetros:
        tablero_jugador (np.array): El tablero del jugador donde se marcan los disparos.
        tablero_oponente (np.array): El tablero del oponente donde se marcan los barcos y los disparos.
        coordenadas (list): Coordenadas (x, y) donde el jugador quiere disparar.
        size (tupla): Dimensiones del tablero (número de filas, número de columnas).

    Retorna:
        tupla: El tablero actualizado del jugador y el tablero actualizado del oponente.
    """

    disparo_exitoso = False
    # Bucle principal que se repite hasta que el disparo se exitoso
    while not disparo_exitoso:

        coordenadas = input("Introduce las coordenadas de disparo (en formato x,y) o escribe 'exit' para salir: ")

        if coordenadas.lower() == "exit":
            print("Has salido del juego.")
            return print(f"Tablero del jugador:\n{tablero_jugador},\nTablero del oponente:\n{tablero_oponente}")  # Terminar el juego y retornar los tableros actuales.
        
 # Solicitar las coordenadas del disparo al usuario.
        try:
            x, y = map(int, coordenadas.split(","))
            coordenadas = x, y
        except ValueError:
            print("Error: Por favor, ingresa dos números enteros separados por una coma.")
            continue  # Reinicia el ciclo si la entrada es inválida

        x,y = coordenadas 
        # Numero maximo de columnas y filas en el tablero 
        num_max_filas = size[0]  # Número máximo de filas del tablero
        num_max_columnas = size[1]  # Número máximo de columnas del tablero

        # Validar que las coordenadas esten dentro del rango del tamaño del tablero y no se salgan de este.
        if x < 0 or y < 0 or x >= num_max_filas or y >= num_max_columnas:
            print(f"Error: Las coordenadas deben estar dentro del rango: (0, 0) a ({num_max_filas - 1}, {num_max_columnas - 1}). Intenta nuevamente.")
            break  # Sale del bucle si las coordenadas están fuera de rango.

        # Verificar el estado de la celda en el tablero del oponente
        if tablero_oponente[coordenadas] == "O":  # Impacto en un barco
            tablero_oponente[coordenadas] = "X"  # Marca el impacto en el tablero del oponente
            print(f"Disparo exitoso al barco ubicado en: {coordenadas}")
            disparo_exitoso = True
        elif tablero_oponente[coordenadas] == " ":  # Disparo al agua
            tablero_oponente[coordenadas] = "*"  # Marca agua en el tablero del oponente
            print(f"Habeis disparado al agua en: {coordenadas}")
            break
        else:
            print("Ya habeis disparado en este sitio, perdiste el turno")
            break

    return tablero_jugador,tablero_oponente

def shoot_random(tablero_jugador, tablero_oponente, size = (10,10)):
    """
    Función que simula un disparo aleatorio de la computadora sobre el tablero del jugador.
    La computadora elige un disparo aleatorio y actualiza los tableros según si el disparo impacta un barco o va al agua.
    
    Parámetros:
        tablero_jugador (np.array): El tablero del jugador donde se marcan los barcos y los disparos.
        tablero_oponente (np.array): El tablero del oponente donde se marcan los disparos.
        size (tupla): Dimensiones del tablero (número de filas, número de columnas).
    
    Retorna:
        tupla: El tablero actualizado del jugador y el tablero actualizado del oponente.
    """
    
    disparo_exitoso = False  # Variable que indica si el disparo fue exitoso o no.
    
    # Obtener las dimensiones del tablero
    num_max_filas = size[0]  # Número máximo de filas del tablero (por defecto 10)
    num_max_columnas = size[1]  # Número máximo de columnas del tablero (por defecto 10)

    # Mientras el disparo no sea exitoso, seguir intentando disparar
    while not disparo_exitoso:
        # Generar coordenadas aleatorias dentro de los límites del tablero
        # La computadora selecciona un número aleatorio entre 0 y el tamaño máximo del tablero (sin incluirlo).
        x = random.randint(0, num_max_filas - 1)
        y = random.randint(0, num_max_columnas - 1)

        # Verificar el estado de la celda en la que la computadora disparó
        if tablero_jugador[x, y] == "O":  # Si hay un barco (representado por "O")
            tablero_jugador[x, y] = "X"  # Marca el impacto en el tablero del jugador
            print(f"Disparo exitoso de la computadora al barco en: ({x}, {y})")
            disparo_exitoso = True  # El disparo fue exitoso, se termina el ciclo.
        elif tablero_jugador[x, y] == " ":  # Si el disparo fue al agua (representado por un espacio vacío " ")
            tablero_jugador[x, y] = "*"  # Marca el agua en el tablero del jugador
            print(f"La computadora disparó al agua en: ({x}, {y})")
            disparo_exitoso = True  # El disparo fue exitoso (al agua), se termina el ciclo.
        else:
            # Si el lugar ya ha sido disparado antes, la computadora pierde el turno
            print(f"Ya se ha disparado en ({x}, {y}), la computadora pierde el turno.")
            disparo_exitoso = True  # Aquí finaliza el ciclo aunque no haya sido un disparo exitoso, solo pierde el turno.

    # Retornar los tableros actualizados
    return tablero_jugador, tablero_oponente

#Este es la primera version de del disparon random si necesita correccion o alguna mejora sientanse libres de hacerlo 


def shoot_board():





####################################################################################################################################

# funciones.py

import numpy as np
from variables import *  # Importa todas las constantes de variables.py

def welcome():
    """Muestra un mensaje de bienvenida e instrucciones para el jugador."""
    print("Bienvenido a la versión del popular juego Hundir la Flota\n\n"
          "Instrucciones:\n"
          "1. Introduce tu nombre o juega con el nombre predeterminado.\n"
          "2. Dispara a las coordenadas del tablero enemigo para intentar hundir sus barcos.\n"
          "3. El juego termina cuando uno de los jugadores pierde todos sus barcos.\n"
          "Buena suerte!")

def player_name():
    """Solicita el nombre del jugador o asigna un nombre predeterminado."""
    name = input('Introduce tu nombre: ').strip()
    return name if name else "Jugador_1"  # Devuelve "Jugador_1" si no se ingresa un nombre

def shoot_player(player_board, opponent_board, size=(10, 10)):
    """
    Simula el disparo del jugador en el tablero del oponente.
    Actualiza el tablero de disparos del jugador y el tablero del oponente según el resultado.

    Parámetros:
        player_board (np.array): El tablero de disparos del jugador.
        opponent_board (np.array): El tablero del oponente con los barcos.
        size (tuple): Dimensiones del tablero.

    Retorna:
        tuple or None: Tableros actualizados del jugador y del oponente, o None si el jugador elige salir.
    """
    successful_shot = False
    while not successful_shot:
        coordinates = input("Introduce las coordenadas de disparo (formato x,y) o escribe 'exit' para salir: ")
        if coordinates.lower() == "exit":
            print("Has salido del juego.")
            return None  # Indica al juego principal que debe terminar

        try:
            x, y = map(int, coordinates.split(","))
        except ValueError:
            print("Error: Ingresa coordenadas en el formato x,y.")
            continue  # Pide nueva entrada si es inválida

        if x < 0 or y < 0 or x >= size[0] or y >= size[1]:
            print(f"Las coordenadas deben estar entre (0, 0) y ({size[0] - 1}, {size[1] - 1}). Intenta de nuevo.")
            continue

        # Verifica el resultado del disparo
        if opponent_board[x, y] == SHIP:
            opponent_board[x, y] = HIT
            player_board[x, y] = HIT
            print(f"¡Impacto en ({x}, {y})!")
            successful_shot = True
        elif opponent_board[x, y] == EMPTY:
            opponent_board[x, y] = MISS
            player_board[x, y] = MISS
            print(f"Agua en ({x}, {y}).")
            break
        else:
            print("Ya disparaste en esa posición. Intenta otra vez.")
            continue

    return player_board, opponent_board
