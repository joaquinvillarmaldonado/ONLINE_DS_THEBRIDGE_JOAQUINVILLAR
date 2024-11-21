import numpy as np
import random
from variables import *  # Importa todas las constantes de variables.py

class Board:
    """Clase que representa el tablero de cada jugador en el juego."""

    def __init__(self):
        # Crear un tablero vacío y un tablero para registrar los disparos
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY) 
        self.shots_board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY) 
        self.generate_ships()  # Coloca los barcos al inicio del juego

    def generate_ships(self):
        """Coloca todos los barcos en posiciones aleatorias válidas en el tablero."""
        for ship_name, ship_info in BARCOS.items():
            for i in range(ship_info['cantidad']):
                length = ship_info['longitud'] # Longitud del barco actual
                self.place_ship(length) # Coloca el barco en el tablero

    def check_position(self, x, y, length, orientation):
        """Verifica si un barco puede colocarse en la posición dada."""
        if orientation == 'H':
            if y + length > BOARD_SIZE:
                return False
            return all(self.board[x, y+i] == EMPTY for i in range(length))  # Comprueba que todas las celdas en la fila estén vacías
        else:
            if x + length > BOARD_SIZE:
                return False
            return all(self.board[x+i, y] == EMPTY for i in range(length)) # Comprueba que todas las celdas en la fila estén vacías

    def place_ship(self, length):
        """Intenta colocar un barco de longitud dada en una posición aleatoria válida."""
        placed = False
        while not placed:
            x, y = random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1) # Genera coordenadas aleatorias para la posición inicial del barco
            orientation = random.choice(['H', 'V']) # Elige aleatoriamente la orientación del barco
            if self.check_position(x, y, length, orientation): # Comprueba si el barco cabe en la posición generada
                if orientation == 'H':
                    for i in range(length):
                        self.board[x, y+i] = SHIP
                else:
                    for i in range(length):
                        self.board[x+i, y] = SHIP
                placed = True

    def shoot(self, x, y):
        """Registra un disparo en el tablero. Devuelve True si es un impacto, False si es agua."""
        if self.board[x, y] == SHIP:
            self.board[x, y] = HIT # Registra el disparo en el tablero inicial
            self.shots_board[x, y] = HIT # Registra el disparo en el tablero de disparos
            return True
        elif self.board[x, y] == EMPTY:
            self.board[x, y] = MISS
            self.shots_board[x, y] = MISS
            return False # Retorna False para indicar agua
        return None  # Ya se disparó en esa posición

