# Variables 

# ID del jugador, inicio con nombre predeterminado modificable por el jugador al inicio de la partida
jugador_ID = 'Jugador_1'

# ID oponente
oponente_ID = 'Oponente_1'

# Dimensiones del tablero
tamaño_tablero = 10

# Barcos
barcos = {
    'A1': 4,
    'B1': 3,
    'C1': 2,
    'D1': 1,
    'D2': 1,
}


####### Opinion de opcion 2  # variables.py

# variables.py

# IDs predeterminados de los jugadores (pueden ser modificados al inicio del juego)
JUGADOR_ID_DEFAULT = 'Jugador_1'
OPONENTE_ID_DEFAULT = 'Oponente_1'

# Dimensiones del tablero
BOARD_SIZE = 10

# Barcos y sus tamaños (longitudes)
# Este diccionario contiene los tipos de barcos, su cantidad y su longitud
BARCOS = {
    'lancha': {'cantidad': 4, 'longitud': 1},      # 4 barcos de 1 casilla
    'submarino': {'cantidad': 3, 'longitud': 2},   # 3 barcos de 2 casillas
    'destructor': {'cantidad': 2, 'longitud': 3},  # 2 barcos de 3 casillas
    'portaviones': {'cantidad': 1, 'longitud': 4}  # 1 barco de 4 casillas
}

# Representaciones en el tablero
# 0: vacío, 1: barco, 2: impacto, 3: fallido
EMPTY = 0
SHIP = 1
HIT = 2
MISS = 3
