"""确保仓库根目录进入 sys.path，使测试可 `import calculator`。

注意：pytest 以不同方式启动（console script / ``python -m pytest``）时对
sys.path 的处理不同；这里显式注入仓库根目录，不依赖隐式行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
