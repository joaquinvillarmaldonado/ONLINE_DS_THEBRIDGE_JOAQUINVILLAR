# main.py

from clases import Board
from funciones import welcome, player_name, shoot_player
import numpy as np
from variables import *  # Importa todas las constantes de variables.py

def main():
    welcome()  # Mostrar bienvenida e instrucciones
    name = player_name()  # Obtener el nombre del jugador
    print(f"\nHola, {name}! Comencemos el juego.")

    # Inicializar tableros del jugador y de la máquina
    user_board = Board()
    computer_board = Board()

    game_over = False
    user_turn = True  # El jugador comienza primero

    while not game_over:
        if user_turn:
            print("\nTu tablero de disparos:")
            print(user_board.shots_board)
            
            # Llamada a shoot_player para realizar el disparo y actualizar los tableros
            result = shoot_player(user_board.shots_board, computer_board.board)

            # Verificar si el jugador decidió salir del juego
            if result is None:
                game_over = True
                print("Juego terminado. Gracias por jugar.")
                break

            # Actualizar los tableros si no se ha terminado el juego
            user_board.shots_board, computer_board.board = result

            # Verificar si el jugador ganó
            if not any(cell == SHIP for row in computer_board.board for cell in row):
                print("¡Felicidades, has ganado! Todos los barcos del oponente han sido hundidos.")
                game_over = True
            else:
                user_turn = False  # Cambiar el turno

        else:
            # Turno de la máquina
            print("\nTurno de la máquina...")
            x, y = np.random.randint(0, BOARD_SIZE), np.random.randint(0, BOARD_SIZE)
            
            # La máquina dispara y actualiza el tablero del jugador
            if user_board.board[x, y] == SHIP:
                user_board.board[x, y] = HIT
                print(f"La máquina impactó en ({x}, {y})")
                if not any(cell == SHIP for row in user_board.board for cell in row):
                    print("Lo siento, la máquina ha ganado. Todos tus barcos han sido hundidos.")
                    game_over = True
            else:
                print(f"La máquina disparó a ({x}, {y}) y falló.")
                user_turn = True  # Cambiar el turno de nuevo al jugador

        # Mostrar el tablero del jugador al inicio de cada turno del jugador
        if user_turn and not game_over:
            print("\nTu tablero:")
            print(user_board.board)

if __name__ == "__main__":
    main()
