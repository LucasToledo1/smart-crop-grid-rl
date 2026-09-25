#!/usr/bin/env bash
set -e

# Instala o ambiente do projeto num venv local (.venv) e valida que tudo
# funciona: import dos pacotes, o ambiente do Gymnasium e os 3 algoritmos do SB3.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Criando venv em .venv ..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Instalando dependencias..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Validando instalacao..."
SDL_VIDEODRIVER=dummy python3 - <<'EOF'
from smart_crop_irrigation import SmartCropIrrigationEnv
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN, PPO, A2C

env = SmartCropIrrigationEnv(render_mode="rgb_array")
check_env(env)
frame = env.render()
assert frame.shape == (env.window_size + env.info_bar_height, env.window_size, 3)
env.close()
print("Ambiente OK (check_env + render).")

for name, Algo in [("DQN", DQN), ("PPO", PPO), ("A2C", A2C)]:
    env = SmartCropIrrigationEnv()
    Algo("MultiInputPolicy", env, verbose=0).learn(total_timesteps=100)
    env.close()
    print(f"{name}: treina sem erro.")
EOF

echo ""
echo "Tudo certo. Pra usar o venv: source .venv/bin/activate"
