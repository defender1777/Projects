# -*- coding: utf-8 -*-
"""
Created on Tue May 19 01:01:36 2020

@author: dmoro
"""

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys # Для sys.exit()
import random
#
# Агент с глубоким Q-обучением
class DQNAgent():
    #
    def __init__(self, observation_space, action_space):
        self.state_size = observation_space
        self.action_size = action_space
        self.memory = deque(maxlen = 20000) # Тип collections.deque
        self.alpha = 1.0 # Скорость обучения агента
        self.gamma = 0.95 # Коэффициент уменьшения вознаграждения агента
        # Уровень обучения повышается с коэффициентом exploration_decay
        # Влияет на выбор действия action (0 или 1)
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.learning_rate = 0.001 # Скорость обучения сети
        self.model = self.build_model()
        print(self.model.summary())
    #
    # Создает модель сети
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(2, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model
    #
    # Запоминаем историю
    def remember(self, state, action, reward, state_next, done):
        self.memory.append((state, action, reward, state_next, done))
    #
    # Определяет и возвращает действие
    def findAction(self, state):
        # Случайный выбор действия - 0 или 1
        if np.random.rand() <= self.exploration_rate: return random.randrange(self.action_size) # или random.randint(0, 1)
        # Выбор действия по состоянию объекта
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]) # Возвращает действие
    #
    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        # Обучение агента
        # Случайная выборка batch_size элементов для обучения агента
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, state_next, done in minibatch:
            # Пример (done = False):
            # state: [[-0.00626842 0.41118423 -0.07340495 -0.77232979]]
            # q_values (до корректировки): [[0.052909 0.05275263]] - numpy.ndarray
            # state_next: [[ 0.00195526 0.21714493 -0.08885155 -0.50361631]]
            # q_values_next: [[0.03970249 0.02732118]]
            # Qsa = 1.0377173654735088
            # reward = 1.0
            # action = 0
            # q_values (после корректировки): [[1.0377173 0.05275263]]
            # q_values (после обучения НС): [[0.07063997 0.04742151]]
            q_values = self.model.predict(state)
            if done:
                Qsa = reward
            else:
                q_values_next = self.model.predict(state_next)[0]
                # Текущая оценка полезности действия action
                Qsa = q_values[0][action]
                # Уточненная оценка полезности действия action
                Qsa = Qsa + self.alpha * (reward + self.gamma * np.amax(q_values_next) - Qsa)
            # Формируем цель обучения сети
            q_values[0][action] = Qsa
            # Обучение сети
            self.model.fit(state, q_values, epochs = 1, verbose = 0)
        if self.exploration_rate > self.exploration_min: self.exploration_rate *= self.exploration_decay
#
if __name__ == "__main__":
    env = gym.make('CartPole-v1') # Создаем среду. Тип: # <TimeLimit<CartPoleEnv<CartPole-v1>>>
    observation_space = env.observation_space.shape[0] # 4
    action_space = env.action_space.n # 2
    # DQN - глубокая Q-нейронная сеть
    dqn_agent = DQNAgent(observation_space, action_space) # Создаем агента
    episodes = 1001 # Число игровых эпизодов + 1
    # scores - хранит длительность игры в последних 100 эпизодах
    # После достижения maxlen новые значения, добавляемые в scores, будут вытеснять прежние
    scores = deque(maxlen = 100) # Тип collections.deque.
    fail = True
    seed = 2
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    for e in range(episodes):
        # Получаем начальное состояние объекта перед началом каждой игры (каждого эпизода)
        state = env.reset() # Как вариант: state = [0.0364131 -0.02130403 -0.03887796 -0.01044108]
        # state[0] - позиция тележки
        # state[1] - скорость тележки
        # state[2] - угол отклонения шеста от вертикали в радианах
        # state[3] - скорость изменения угла наклона шеста
        state = np.reshape(state, (1, observation_space))
        # Начинаем игру
        # frame - текущий кадр (момент) игры
        # Цель - как можно дольше не допустить падения шеста
        frames = 0
        while True:
            env.render() # Графическое отображение симуляции
            frames += 1
            action = dqn_agent.findAction(state) # Определяем очередное действие
            # Получаем от среды, в которой выполнено действие action, состояние объекта, награду и значение флага завершения игры
            # В каждый момент игры, пока не наступило одно из условий ее прекращения, награда равна 1
            state_next, reward, done, info = env.step(action)
            state_next = np.reshape(state_next, (1, observation_space))
            reward = reward if not done else -reward
            # Запоминаем предыдущее состояние объекта, действие, награду за это действие, текущее состояние и значение done
            dqn_agent.remember(state, action, reward, state_next, done)
            state = state_next # Обновляем текущее состояние
            # done становится равным True, когда завершается игра, например, отклонение угла превысило допустимое значение
            if done:
                # Печатаем продолжительность игры и покидаем внутренний цикл while
                print("Эпизод: {}/{}, продолжительность игры в кадрах: {}".format(e, episodes - 1, frames))
                break
        scores.append(frames)
        if e > 100:
            score_mean = np.mean(scores)
            if score_mean > 195:
                print('Цель достигнута. Средняя продолжительность игры: ', score_mean)
                fail = False
                break
        # Продолжаем обучать агента
        dqn_agent.replay(24)
    if fail: print('Задача не решена ')