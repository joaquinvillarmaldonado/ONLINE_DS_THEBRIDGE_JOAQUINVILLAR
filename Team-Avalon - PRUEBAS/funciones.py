import numpy as np
from variables import *  # Importa todas las constantes de variables.py

def welcome():
    """Muestra un mensaje de bienvenida e instrucciones para el jugador."""
    print("Bienvenido a la versión del popular juego Hundir la Flota\n\n"
          "Instrucciones:\n"
          "1. Introduce tu nombre o juega con el nombre predeterminado.\n"
          "2. Dispara a las coordenadas del tablero enemigo para intentar hundir sus barcos.\n"
          "3. El juego termina cuando uno de los jugadores pierde todos sus barcos.\n"
          "4- El juego acaba cuando alguno de los jugadores se queda sin barcos en el tablero.\n\n"
          "Puedes salir del juego en cualquier momento escribiendo el comando exit.\n\n" 
          "Esperamos lo disfrutes ¡Buena Suerte!")

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