from itertools import product
import pandas as pd
import random
import math
random.seed(1)
opponents = ['Team '+chr(ord('A') + i) for i in range(5)]
stadiums = ['Home', 'Away']
games = pd.DataFrame(list(product(opponents, stadiums))*2,
                     columns=['opponent', 'stadium'])
total_games = len(games)
games['result'] = random.choices(["Win", "Loss", "Draw"],
                                 k=total_games)
# print(games)

# sorry
entropy_result = sum([games['result'].str.contains(result).sum() / games['result'].__len__() * -math.log2(games['result'].str.contains(result).sum() / games['result'].__len__()) for result in ["Win", "Loss", "Draw"]])

print(f'H(result) = {entropy_result}')

home_games = games.loc[games['stadium'] == 'Home']
away_games = games.loc[games['stadium'] == 'Away']

counts = home_games['result'].value_counts()
home_games_entropy = sum([-math.log2(count / len(home_games)) * (count / len(home_games)) for count in counts])

counts = away_games['result'].value_counts()
away_games_entropy = sum([-math.log2(count / len(away_games)) * (count / len(away_games)) for count in counts])

probability_home = len(home_games) / total_games
probability_away = len(away_games) / total_games
avg_entropy_stadium = probability_home * home_games_entropy + probability_away * away_games_entropy

print(f'1)\nH(result|stadium) = {avg_entropy_stadium}')

def calc_entropy_opponent(opponent):
    opponent_games = games.loc[games['opponent'] == opponent]
    counts = opponent_games['result'].value_counts()
    entropy = sum([-math.log2(count / len(opponent_games)) * (count / len(opponent_games)) for count in counts])
    return entropy

opponents_entropies = pd.DataFrame([calc_entropy_opponent(opponent) for opponent in opponents], columns=["entropy"])
opponents_entropies['probabilities'] = [len(games.loc[games['opponent'] == opponent]) / total_games for opponent in opponents]

# print(opponents_entropies)

avg_entropy_opponents = sum(opponents_entropies['entropy'] * opponents_entropies['probabilities'])

print(f'2)\nH(result|opponent) = {avg_entropy_opponents}')

print(f"3)\nIG(result;opponent) = H(result) - H(result|opponent = {entropy_result - avg_entropy_opponents}\n is higher than \nIG(result;stadium) = H(result) - H(result|stadium = {entropy_result - avg_entropy_stadium}\ntherefore, the variable opponent is more important in deciding the result of the game")