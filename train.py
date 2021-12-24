from pathlib import Path

import hydra

from workspaces import Workspace


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()
    pass


if __name__ == '__main__':
    main()
