from dataclasses import dataclass

from rl.core import Action, Direction, Environment, State


@dataclass(frozen=True)
class GridPosition:
    row: int
    col: int


class GridWorld(Environment[GridPosition, Direction]):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (3, 3),
        walls: set[tuple[int, int]] | None = None,
        step_penalty: float = -0.1,
        goal_reward: float = 10.0,
    ):
        self.rows = rows
        self.cols = cols
        self.start = GridPosition(*start)
        self.goal = GridPosition(*goal)
        self.walls = {GridPosition(*w) for w in (walls or set())}
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self._current: GridPosition = self.start

    def reset(self) -> State[GridPosition]:
        self._current = self.start
        return State(self._current)

    def step(self, action: Action[Direction]) -> tuple[State[GridPosition], float, bool]:
        dr, dc = action.value.value
        new_pos = GridPosition(self._current.row + dr, self._current.col + dc)
        if self._is_valid(new_pos):
            self._current = new_pos
        done = self._current == self.goal
        reward = self.goal_reward if done else self.step_penalty
        return State(self._current), reward, done

    def actions(self, state: State[GridPosition]) -> list[Action[Direction]]:
        return [Action(d) for d in Direction]

    def _is_valid(self, pos: GridPosition) -> bool:
        in_bounds = 0 <= pos.row < self.rows and 0 <= pos.col < self.cols
        return in_bounds and pos not in self.walls

    def render(self) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                pos = GridPosition(r, c)
                if pos == self._current:
                    row.append("A")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        return "\n".join(lines)
