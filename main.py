from typing import Callable

import numpy as np


def game_core(number: int = 1) -> int:
    """
    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0
    left, right = 1, 100
    while True:
        predict_number = (left + right) // 2
        count += 1
        if number > predict_number:
            left = predict_number + 1
        elif number < predict_number:
            right = predict_number - 1
        else:
            break

    return count


def score_game(random_predict: Callable[[int], int]) -> int:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        random_predict (Callable[[int], int]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    #np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(1000))  # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за:{score} попыток")
    return score


if __name__ == '__main__':
    # RUN
    score_game(game_core)
