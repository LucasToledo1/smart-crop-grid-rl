import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartCropIrrigationEnv(gym.Env):
    """
    Ambiente Customizado: Smart Crop Irrigation Grid
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=5, max_steps=250):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        # Posição da base
        self.base_pos = np.array([0, 0])

        # 8 Ações: 0: Cima, 1: Baixo, 2: Esq, 3: Dir, 4: Regar, 5: Colher, 6: Descarregar, 7: Esperar
        self.action_space = spaces.Discrete(8)

        # Espaço de Observação usando Dict
        self.observation_space = spaces.Dict(
            {
                "robot_pos": spaces.MultiDiscrete([self.grid_size, self.grid_size]),
                "inventory": spaces.Discrete(4),  # 0 a 3 plantas
                # Matriz de umidade: 0 (morta) a 4 (encharcada). Usando Box para facilitar a representação 2D.
                "grid_moisture": spaces.Box(
                    low=0, high=4, shape=(self.grid_size, self.grid_size), dtype=np.int8
                ),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.robot_pos = np.copy(self.base_pos)
        self.inventory = 0

        # Inicializa o grid com as plantas no nível ideal de umidade (2)
        self.grid_moisture = np.full((self.grid_size, self.grid_size), 2, dtype=np.int8)
        # (Opcional: você pode marcar a posição [0,0] como -1 ou outro valor para indicar que ali é a base e não tem planta)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.current_step += 1
        reward = -0.1  # Penalidade padrão por passo (tempo)
        terminated = False
        truncated = False

        # 1. Processar a Ação do Agente
        if action == 0:  # Cima
            self.robot_pos[0] = max(0, self.robot_pos[0] - 1)
        elif action == 1:  # Baixo
            self.robot_pos[0] = min(self.grid_size - 1, self.robot_pos[0] + 1)
        elif action == 2:  # Esquerda
            self.robot_pos[1] = max(0, self.robot_pos[1] - 1)
        elif action == 3:  # Direita
            self.robot_pos[1] = min(self.grid_size - 1, self.robot_pos[1] + 1)

        elif action == 4:  # Regar
            r, c = self.robot_pos
            if self.grid_moisture[r, c] > 0 and self.grid_moisture[r, c] < 4:
                self.grid_moisture[
                    r, c
                ] += 1  # Aumenta 1 nível (pode ajustar para +2 se quiser)
                reward += 1  # Recompensa por regar
            else:
                reward -= (
                    1  # Penalidade por regar planta morta ou tentar encharcar mais
                )

        elif action == 5:  # Colher
            r, c = self.robot_pos
            if self.inventory < 3:
                if self.grid_moisture[r, c] == 2:
                    self.inventory += 1
                    self.grid_moisture[r, c] = (
                        0  # Planta colhida (vira 'célula vazia/morta')
                    )
                    reward += 5
                else:
                    reward -= 2  # Penalidade por colher fora do nível ideal
            else:
                reward -= 1  # Inventário cheio

        elif action == 6:  # Descarregar
            if np.array_equal(self.robot_pos, self.base_pos) and self.inventory > 0:
                reward += 20 * self.inventory  # +20 por planta entregue
                self.inventory = 0
            else:
                reward -= 1  # Tentar descarregar fora da base ou vazio

        elif action == 7:  # Esperar
            pass

        # 2. Dinâmica do Ambiente (Secagem estocástica)
        self._apply_moisture_decay()

        # Verificar se matou alguma planta neste turno para dar penalidade extra (opcional)
        # reward -= (plantas que morreram neste passo * 2)

        # 3. Condições de Parada
        if self.current_step >= self.max_steps:
            truncated = True

        # Condição de Sucesso: Todas as plantas colhidas e inventário vazio
        # (Considerando que 0 significa sem planta ou morta, e precisamos validar se todas as viáveis foram entregues)
        if np.all(self.grid_moisture == 0) and self.inventory == 0:
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_moisture_decay(self):
        # 15% de chance de secar (-1 nível) para cada planta viva
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Ignora se estiver morta/colhida (0) ou se for a base [0,0]
                if self.grid_moisture[r, c] > 0 and not (r == 0 and c == 0):
                    if self.np_random.random() < 0.15:
                        self.grid_moisture[r, c] -= 1

    def _get_obs(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "inventory": self.inventory,
            "grid_moisture": self.grid_moisture.copy(),
        }

    def _get_info(self):
        return {"current_step": self.current_step}

    def render(self):
        # A Fase 1.2 pode seguir daqui
        pass
