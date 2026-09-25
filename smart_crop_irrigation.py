import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartCropIrrigationEnv(gym.Env):
    """
    Ambiente Customizado: Smart Crop Irrigation Grid
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size=5, max_steps=250, render_mode=None, dry_rate=0.08):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        # Chance por passo de cada planta secar 1 nível. Era 0.15 fixo no código
        # (testamos: matava uma planta esquecida em ~13 passos, rápido demais pro
        # agente conseguir colher o grid inteiro a tempo). Exposto como parâmetro
        # pra poder testar valores diferentes nos experimentos.
        self.dry_rate = dry_rate

        # Posição da base
        self.base_pos = np.array([0, 0])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Estado da janela do Pygame (só é criada na primeira chamada de render)
        self.cell_size = 60
        self.info_bar_height = 40
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None

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
        # Correção: a base [0,0] não é uma planta, então não pode nascer com umidade 2
        # (senão ela vira colhível/regável como qualquer célula, e o agente precisaria
        # "colher a base" pra vencer). Fica em 0, o mesmo valor usado pra célula sem planta.
        self.grid_moisture[0, 0] = 0

        # Total de plantas que existem no grid (tudo, menos a base) e quantas já
        # foram entregues. Usado pra condição de sucesso (ver step()).
        self.total_plants = self.grid_size * self.grid_size - 1
        self.plants_delivered = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

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
            # Correção: só recompensa regar quando a planta está abaixo do nível ideal (2).
            # Antes também recompensava regar uma planta já no nível ideal, o que a
            # empurrava pra fora do nível colhível — um incentivo contra o próprio objetivo.
            if self.grid_moisture[r, c] > 0 and self.grid_moisture[r, c] < 2:
                self.grid_moisture[r, c] += 1  # Aumenta 1 nível
                reward += 1  # Recompensa por regar
            else:
                reward -= 1  # Penalidade por regar planta morta, já ideal ou encharcada

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
                self.plants_delivered += self.inventory
                self.inventory = 0
            else:
                reward -= 1  # Tentar descarregar fora da base ou vazio

        elif action == 7:  # Esperar
            pass

        # 2. Dinâmica do Ambiente (Secagem estocástica)
        deaths = self._apply_moisture_decay()
        # Penalidade por negligência: sem isso, deixar uma planta morrer só custava a
        # recompensa futura perdida (colher/entregar), um sinal indireto e atrasado.
        reward -= 2 * deaths

        # 3. Condições de Parada
        if self.current_step >= self.max_steps:
            truncated = True

        # Condição de Sucesso: todas as plantas do grid foram colhidas E entregues na base.
        # Correção: antes checava só "grid todo zerado", mas uma planta que morre de sede
        # também vira 0 — então o agente "vencia" só esperando o grid secar sozinho, sem
        # nunca colher nada. Agora exige contar entregas de verdade (self.plants_delivered).
        if self.plants_delivered >= self.total_plants:
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _apply_moisture_decay(self):
        # Chance por passo (self.dry_rate) de cada planta secar 1 nível. Retorna
        # quantas morreram neste passo (chegaram a 0), pra aplicar penalidade em step().
        deaths = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Ignora se estiver morta/colhida (0) ou se for a base [0,0]
                if self.grid_moisture[r, c] > 0 and not (r == 0 and c == 0):
                    if self.np_random.random() < self.dry_rate:
                        self.grid_moisture[r, c] -= 1
                        if self.grid_moisture[r, c] == 0:
                            deaths += 1
        return deaths

    def _get_obs(self):
        return {
            "robot_pos": self.robot_pos.copy(),
            "inventory": self.inventory,
            "grid_moisture": self.grid_moisture.copy(),
        }

    def _get_info(self):
        return {"current_step": self.current_step}

    # Cores por nível de umidade (0 = sem planta/morta, 2 = ideal, 4 = encharcada).
    # Níveis 3 e 4 hoje nunca ocorrem de fato: a ação Regar para de funcionar a partir
    # do nível 2 (ver step()). Ficam mapeados aqui caso essa regra mude no futuro.
    _MOISTURE_COLORS = {
        0: (120, 100, 80),
        1: (194, 178, 128),
        2: (76, 175, 80),
        3: (46, 125, 50),
        4: (33, 90, 160),
    }

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "Chamou render() sem definir render_mode no construtor (use 'human' ou 'rgb_array')."
            )
            return
        return self._render_frame()

    def _render_frame(self):
        import pygame

        if not pygame.get_init():
            pygame.init()  # inicializa também o módulo de fontes, usado em ambos os modos

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Smart Crop Irrigation Grid")
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size + self.info_bar_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + self.info_bar_height))
        canvas.fill((255, 255, 255))

        # Grid: uma célula por planta, colorida conforme o nível de umidade
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size
                )
                level = int(self.grid_moisture[r, c])
                pygame.draw.rect(canvas, self._MOISTURE_COLORS[level], rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, width=1)

        # Base: contorno dourado sobre a célula [0,0], pra não confundir com planta morta
        base_rect = pygame.Rect(
            int(self.base_pos[1]) * self.cell_size,
            int(self.base_pos[0]) * self.cell_size,
            self.cell_size,
            self.cell_size,
        )
        pygame.draw.rect(canvas, (255, 215, 0), base_rect, width=4)

        # Robô
        robot_center = (
            int(self.robot_pos[1]) * self.cell_size + self.cell_size // 2,
            int(self.robot_pos[0]) * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(canvas, (20, 20, 20), robot_center, self.cell_size // 3)

        # Inventário: até 3 quadrados, preenchido = planta carregada
        slot_size = 16
        for i in range(3):
            slot_rect = pygame.Rect(
                8 + i * (slot_size + 4), self.window_size + 12, slot_size, slot_size
            )
            color = (76, 175, 80) if i < self.inventory else (255, 255, 255)
            pygame.draw.rect(canvas, color, slot_rect)
            pygame.draw.rect(canvas, (0, 0, 0), slot_rect, width=1)

        # Passos: barra de progresso até o timeout
        bar_x = 8 + 3 * (slot_size + 4) + 12
        bar_width = self.window_size - bar_x - 8
        progress = self.current_step / self.max_steps
        pygame.draw.rect(
            canvas, (200, 200, 200), (bar_x, self.window_size + 12, bar_width, slot_size)
        )
        pygame.draw.rect(
            canvas,
            (33, 90, 160),
            (bar_x, self.window_size + 12, int(bar_width * progress), slot_size),
        )
        pygame.draw.rect(
            canvas, (0, 0, 0), (bar_x, self.window_size + 12, bar_width, slot_size), width=1
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        # Só limpa o pygame se uma janela de verdade foi criada (render_mode="human").
        # Em "rgb_array" o pygame.init() roda mas nunca é encerrado aqui - inofensivo
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
