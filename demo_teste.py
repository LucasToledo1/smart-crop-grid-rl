"""Demo do DQN: lê o config.json ao lado do modelo e não altera o treino.

Uso: python demo_teste.py --model models/dqn_organizado_s42/best_model.zip
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, A2C
import smart_crop_irrigation_V2 as crop
from smart_crop_irrigation_V2 import SmartCropIrrigationEnv


def load_model(path, algo):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f'Falha de integridade do ZIP: {bad}')
    original_load = torch.load

    def load_via_memory(file, *args, **kwargs):
        if isinstance(file, zipfile.ZipExtFile):
            file.seek(0)
            file = io.BytesIO(file.read())
        return original_load(file, *args, **kwargs)

    with patch.object(torch, 'load', new=load_via_memory):
        algorithms = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
        return algorithms[algo].load(str(path), device='cpu')


def episode_metrics(env, terminated, truncated, waterings, waits, borders):
    return {
        'plants_dead': int(np.count_nonzero(env.dead_cells)),
        'plants_alive': int(np.count_nonzero(env.grid_moisture > 0)),
        'plants_ready': int(np.count_nonzero(env.grid_moisture == 2)),
        'plants_need_water': int(np.count_nonzero(env.grid_moisture == 1)),
        'inventory': int(env.inventory),
        'successful_waterings': waterings,
        'wait_actions': waits,
        'blocked_moves': borders,
        'episode_end_reason': (
            'success'
            if env.plants_delivered >= env.total_plants
            else 'no_work_left_with_losses'
            if terminated
            else 'time_limit'
            if truncated
            else 'running'
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Caminho do modelo .zip.')
    parser.add_argument('--algo', choices=['dqn', 'ppo', 'a2c'], default='dqn')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--no-render', action='store_true', help='Executa rapidamente, sem janela.')
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--dry-rate', type=float, default=0.005)

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error('episodes deve ser positivo')
    model_path = Path(args.model)
    config_path = model_path.parent / 'config.json'

    # if not config_path.is_file():
    #     parser.error(f'Configuração não encontrada: {config_path}')
    # config = json.loads(config_path.read_text(encoding='utf-8'))
    # current_hash = hashlib.sha256(Path(crop.__file__).read_bytes()).hexdigest()
    # if config.get('environment_sha256') != current_hash:
    #     parser.error('O smart_crop_irrigation.py difere do arquivo usado no treino. Use a mesma versão do treinamento.')

    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    else:
        config = {}

    # Completa configurações ausentes com os argumentos do comando.
    for key in ('grid_size', 'max_steps', 'dry_rate'):
        value = getattr(args, key)

        if value is not None:
            if key in config and config[key] != value:
                parser.error(f'{key} difere do config.json do treinamento.')
            config[key] = value

    missing = [
        key for key in ('grid_size', 'max_steps', 'dry_rate')
        if key not in config
    ]
    if missing:
        flags = ', '.join('--' + key.replace('_', '-') for key in missing)
        parser.error(f'Informe os valores usados no treino: {flags}')

    if config['grid_size'] < 2 or config['max_steps'] < 1:
        parser.error('grid-size deve ser >= 2 e max-steps deve ser >= 1.')
    if not 0 <= config['dry_rate'] <= 1:
        parser.error('dry-rate deve estar entre 0 e 1.')

    current_hash = hashlib.sha256(Path(crop.__file__).read_bytes()).hexdigest()
    saved_hash = config.get('environment_sha256')

    if saved_hash is not None:
        if saved_hash != current_hash:
            parser.error(
                'O smart_crop_irrigation_V2.py difere do usado no treino.'
            )
    else:
        print(
            'Sem hash registrado: confira se o V2 e seus parâmetros '
            'são os mesmos usados no treinamento.'
        )

    env = SmartCropIrrigationEnv(
        grid_size=config['grid_size'], 
        max_steps=config['max_steps'],
        dry_rate=config['dry_rate'], 
        render_mode=None if args.no_render else 'human',
    )
    print(f"Ambiente: grid={config['grid_size']}, dry_rate={config['dry_rate']}, limite={config['max_steps']}")
    try:
        model = load_model(model_path, args.algo)
        if env.observation_space != model.observation_space or env.action_space != model.action_space:
            raise ValueError('Os espaços do modelo e do ambiente não correspondem. Use um modelo treinado com estes arquivos.')
        if not args.no_render:
            import pygame
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            total_reward = 0.0
            waterings = waits = borders = 0
            terminated = truncated = False
            for _ in range(config['max_steps']):
                if not args.no_render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                action, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(action).item())
                local_moisture = int(env.grid_moisture[tuple(env.robot_pos)])
                waterings += int(action == 4 and local_moisture == 1)
                waits += int(action == 7)
                obs, reward, terminated, truncated, info = env.step(action)
                borders += int(info['blocked_move'])
                total_reward += reward
                if terminated or truncated:
                    break
            stats = episode_metrics(env, terminated, truncated, waterings, waits, borders)
            print(f'Episodio {ep + 1}: recompensa total = {total_reward:.1f}, passos = {env.current_step}, '
                  f'entregues = {env.plants_delivered}/{env.total_plants}')
            print(f"Mortas: {stats['plants_dead']} | Vivas no campo: {stats['plants_alive']} | "
                  f"Prontas: {stats['plants_ready']} | Precisando de água: {stats['plants_need_water']} | "
                  f"Inventário: {stats['inventory']} | Irrigações: {stats['successful_waterings']} | "
                  f"Esperas: {stats['wait_actions']} | Bordas: {stats['blocked_moves']} | "
                  f"Fim: {stats['episode_end_reason']}")
    finally:
        env.close()


if __name__ == '__main__':
    main()
