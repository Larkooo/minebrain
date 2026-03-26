"""
Fullscreen training dashboard for MineBrain.

Rich-based TUI that shows per-agent activity, skill usage, curriculum
progression, training metrics, and recent episode history.
"""

import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, SpinnerColumn
from rich.table import Table
from rich.text import Text

from src.curriculum import STAGES, NUM_STAGES
from src.skills import SKILLS, NUM_SKILLS, SkillCategory


# ──────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentStatus:
    env_id: int = 0
    health: float = 20.0
    food: float = 20.0
    tool_tier: str = "none"
    y_level: int = 64
    last_skill: str = ""
    step_count: int = 0
    alive: bool = True
    diamonds: int = 0
    iron: int = 0
    dimension: str = "overworld"


@dataclass
class EpisodeRecord:
    episode_num: int = 0
    steps: int = 0
    reward: float = 0.0
    outcome: str = "?"       # GOAL, DIED, TIMEOUT
    detail: str = ""


# ──────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────

TIER_COLORS = {
    "none": "dim white",
    "wood": "yellow",
    "stone": "white",
    "iron": "bright_white",
    "diamond": "cyan",
}

OUTCOME_COLORS = {
    "GOAL": "bold green",
    "DIED": "bold red",
    "TIMEOUT": "yellow",
    "?": "dim",
}

SPARKLINE_BLOCKS = " ▁▂▃▄▅▆▇█"


class FullscreenDashboard:
    """Rich-based fullscreen training dashboard."""

    def __init__(self, n_envs: int = 8, stage: int = 0):
        self.console = Console()
        self.n_envs = n_envs
        self.live: Live | None = None

        # Training state
        self.stage = stage
        self.total_steps = 0
        self.max_steps = 2_000_000
        self.episodes = 0
        self.elapsed_sec = 0.0
        self.sps = 0
        self.phase = "init"
        self.t_start = time.time()

        # Metrics
        self.recent_rewards = deque(maxlen=100)
        self.recent_goals = deque(maxlen=50)
        self.recent_deaths = deque(maxlen=50)
        self.reward_history: list[float] = []
        self.cur_loss = 0.0
        self.cur_pg_loss = 0.0
        self.cur_vf_loss = 0.0
        self.cur_entropy = 0.0
        self.shaping_weight = 0.5
        self.best_goal_rate = 0.0

        # Curriculum
        self.completed_stages: list[bool] = [False] * NUM_STAGES

        # Agents
        self.agents: list[AgentStatus] = [
            AgentStatus(env_id=i) for i in range(n_envs)
        ]

        # Skill tracking
        self.skill_counts: dict[str, int] = {}

        # Episode history
        self.recent_episodes: deque[EpisodeRecord] = deque(maxlen=20)

        # Render throttle
        self._last_render = 0.0

    # ── Public API ──

    def start(self):
        """Start the live display."""
        self.t_start = time.time()
        self.live = Live(
            self._build_layout(),
            console=self.console,
            screen=True,
            refresh_per_second=4,
        )
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def set_stage(self, stage: int):
        self.stage = stage
        # Mark previous stages as complete
        for i in range(stage):
            self.completed_stages[i] = True

    def update_step(self, env_id: int, info: dict, reward: float):
        """Called after each environment step."""
        agent = self.agents[env_id]

        # Extract agent state from info
        raw = info.get("raw_state", {})
        player = raw.get("player", {})
        agent.health = player.get("health", 20)
        agent.food = player.get("food", 20)
        pos = player.get("position", {})
        agent.y_level = int(pos.get("y", 64))

        inv = raw.get("inventory", {})
        tools = inv.get("tools", {})
        if tools.get("has_diamond_pickaxe"):
            agent.tool_tier = "diamond"
        elif tools.get("has_iron_pickaxe"):
            agent.tool_tier = "iron"
        elif tools.get("has_stone_pickaxe"):
            agent.tool_tier = "stone"
        elif tools.get("has_wooden_pickaxe"):
            agent.tool_tier = "wood"
        else:
            agent.tool_tier = "none"

        resources = inv.get("resources", {})
        agent.diamonds = resources.get("diamond", 0)
        agent.iron = resources.get("iron_ingot", 0)

        world = raw.get("world", {})
        agent.dimension = world.get("dimension", "overworld")

        skill_result = info.get("skill_result", {})
        skill_name = skill_result.get("skill_name", "")
        if skill_name:
            agent.last_skill = skill_name
            self.skill_counts[skill_name] = self.skill_counts.get(skill_name, 0) + 1

        agent.alive = not info.get("died", False)
        agent.step_count += 1

    def record_episode(self, episode_num: int, steps: int, reward: float,
                       goal_met: bool, died: bool, detail: str = ""):
        """Called when an episode completes."""
        self.episodes = episode_num
        self.recent_rewards.append(reward)
        self.recent_goals.append(float(goal_met))
        self.recent_deaths.append(float(died))

        outcome = "GOAL" if goal_met else ("DIED" if died else "TIMEOUT")
        self.recent_episodes.appendleft(EpisodeRecord(
            episode_num=episode_num,
            steps=steps,
            reward=reward,
            outcome=outcome,
            detail=detail,
        ))

    def update_training(self, loss: float, pg: float, vf: float, entropy: float):
        """Called after PPO update."""
        self.cur_loss = loss
        self.cur_pg_loss = pg
        self.cur_vf_loss = vf
        self.cur_entropy = entropy

    def update_progress(self, total_steps: int, sps: int, phase: str,
                        shaping_weight: float = 0.5):
        """Called to update overall progress."""
        self.total_steps = total_steps
        self.sps = sps
        self.phase = phase
        self.elapsed_sec = time.time() - self.t_start
        self.shaping_weight = shaping_weight

        avg = self._avg_reward()
        if avg != 0:
            self.reward_history.append(avg)

    def render(self, force: bool = False):
        """Render the dashboard. Throttled to ~4 fps."""
        now = time.time()
        if not force and (now - self._last_render) < 0.25:
            return
        self._last_render = now

        if self.live:
            self.live.update(self._build_layout())

    # ── Layout builders ──

    def _build_layout(self) -> Layout:
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", size=10),
            Layout(name="agents", size=3 + (self.n_envs + 1) // 2),
            Layout(name="skills", size=4),
            Layout(name="episodes", size=8),
        )

        layout["header"].update(self._build_header())
        layout["body"].split_row(
            Layout(self._build_metrics(), name="metrics", ratio=1),
            Layout(self._build_curriculum(), name="curriculum", ratio=1),
        )
        layout["agents"].update(self._build_agents())
        layout["skills"].update(self._build_skills())
        layout["episodes"].update(self._build_episodes())

        return layout

    def _build_header(self) -> Panel:
        pct = min(self.total_steps / max(self.max_steps, 1), 1.0)
        filled = int(pct * 40)
        bar = "[bold cyan]" + "█" * filled + "[dim]" + "░" * (40 - filled) + "[/]"

        eta = self._eta()
        stage_cfg = STAGES[self.stage]

        title = Text.assemble(
            ("MINEBRAIN", "bold magenta"),
            ("  Stage ", "dim"),
            (str(self.stage), "bold white"),
            (f": {stage_cfg.name}", "bold"),
            ("  —  ", "dim"),
            (stage_cfg.description, "dim italic"),
        )

        progress_line = Text.assemble(
            (f"  {bar}  ", ""),
        )

        stats = Text.assemble(
            (f"  {pct*100:5.1f}%", "bold"),
            (f"   {self.total_steps:,}", "cyan"),
            (f" / {self.max_steps:,} steps", "dim"),
            ("   │ ", "dim"),
            (f"{self.sps:,}", "bold"),
            (" sps", "dim"),
            ("   │ ", "dim"),
            (f"{self.episodes:,}", "bold"),
            (" episodes", "dim"),
            ("   │ ", "dim"),
            (self._fmt_duration(self.elapsed_sec), ""),
            ("   │ ", "dim"),
            ("ETA ", "dim"),
            (eta, "bold yellow"),
            ("   │ ", "dim"),
            (self.phase, "italic cyan"),
        )

        return Panel(
            Group(title, stats),
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics(self) -> Panel:
        avg = self._avg_reward()
        std = self._reward_std()
        goal_rate = self._goal_rate()
        death_rate = self._death_rate()

        rwd_color = "green" if avg > 5 else ("yellow" if avg > 1 else "red")
        goal_color = "green" if goal_rate > 0.6 else ("yellow" if goal_rate > 0.2 else "red")

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", width=12)
        t.add_column(width=22)

        t.add_row("Reward", f"[{rwd_color} bold]{avg:.2f}[/] [dim]± {std:.1f}[/]")
        t.add_row("Goal Rate", f"[{goal_color} bold]{goal_rate*100:.0f}%[/]")
        t.add_row("Death Rate", f"[red]{death_rate*100:.0f}%[/]")
        t.add_row("", "")
        t.add_row("Loss", f"[cyan]{self.cur_loss:.4f}[/]" if self.cur_loss else "[dim]—[/]")
        t.add_row(
            "[dim]PG / VF / E[/]",
            f"[dim]{self.cur_pg_loss:.3f}  {self.cur_vf_loss:.3f}  {self.cur_entropy:.3f}[/]"
            if self.cur_loss else "[dim]—[/]",
        )
        t.add_row("", "")
        t.add_row("Trend", self._sparkline())

        return Panel(t, title="[bold]Training[/]", border_style="blue", padding=(0, 1))

    def _build_curriculum(self) -> Panel:
        lines = []

        stage_names = [s.name for s in STAGES]
        for i in range(NUM_STAGES):
            if self.completed_stages[i]:
                icon = "[green]✓[/]"
                style = "green"
            elif i == self.stage:
                icon = "[yellow bold]▶[/]"
                style = "yellow bold"
            else:
                icon = "[dim]·[/]"
                style = "dim"

            name = stage_names[i]
            lines.append(f"  {icon} [dim]{i}[/] [{style}]{name}[/]")

        # Promotion status
        cfg = STAGES[self.stage]
        window = cfg.promotion_window
        recent = list(self.recent_goals)[-window:] if len(self.recent_goals) >= window else list(self.recent_goals)
        if recent:
            rate = sum(recent) / len(recent)
            threshold = cfg.promotion_threshold
            if rate >= threshold:
                promo = f"[green bold]READY[/] ({rate*100:.0f}% >= {threshold*100:.0f}%)"
            else:
                needed = int(threshold * window - sum(recent))
                promo = f"{rate*100:.0f}% / {threshold*100:.0f}% [dim](need {needed} more wins)[/]"
        else:
            promo = f"[dim]need {window} episodes[/]"

        lines.append("")
        lines.append(f"  [dim]Promotion:[/] {promo}")
        lines.append(f"  [dim]Shaping:[/]   {self.shaping_weight:.3f}")
        lines.append(f"  [dim]Best goal:[/] {self.best_goal_rate*100:.0f}%")

        return Panel(
            "\n".join(lines),
            title="[bold]Curriculum[/]",
            border_style="blue",
            padding=(0, 1),
        )

    def _build_agents(self) -> Panel:
        table = Table.grid(padding=(0, 1), expand=True)
        # 2 columns for the agent grid
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        rows = []
        for agent in self.agents:
            if not agent.alive:
                cell = f"[red bold]{agent.env_id}[/] [red]DEAD[/] [dim]resetting...[/]"
            else:
                h_color = "green" if agent.health > 14 else ("yellow" if agent.health > 6 else "red")
                tier_color = TIER_COLORS.get(agent.tool_tier, "dim")
                dim_icon = {"overworld": "", "the_nether": " [red]N[/]", "the_end": " [magenta]E[/]"}.get(agent.dimension, "")

                skill_display = agent.last_skill
                if len(skill_display) > 22:
                    skill_display = skill_display[:20] + ".."

                resources = []
                if agent.diamonds > 0:
                    resources.append(f"[cyan]◆{agent.diamonds}[/]")
                if agent.iron > 0:
                    resources.append(f"[white]▪{agent.iron}[/]")
                res_str = " ".join(resources) if resources else ""

                cell = (
                    f"[bold]{agent.env_id}[/] "
                    f"[{h_color}]♥{agent.health:.0f}[/] "
                    f"[{tier_color}]✦{agent.tool_tier}[/] "
                    f"[dim]y={agent.y_level}[/]{dim_icon} "
                    f"[italic]{skill_display}[/] "
                    f"{res_str}"
                )

            rows.append(cell)

        # Pair agents into 2-column rows
        for i in range(0, len(rows), 2):
            left = rows[i]
            right = rows[i + 1] if i + 1 < len(rows) else ""
            table.add_row(left, right)

        return Panel(table, title="[bold]Agents[/]", border_style="blue", padding=(0, 1))

    def _build_skills(self) -> Panel:
        if not self.skill_counts:
            return Panel("[dim]No skills executed yet[/]", title="[bold]Skill Usage[/]",
                         border_style="blue", padding=(0, 1))

        total = sum(self.skill_counts.values())
        if total == 0:
            return Panel("[dim]—[/]", title="[bold]Skill Usage[/]",
                         border_style="blue", padding=(0, 1))

        # Sort by count, take top 8
        sorted_skills = sorted(self.skill_counts.items(), key=lambda x: -x[1])[:8]
        max_count = sorted_skills[0][1] if sorted_skills else 1

        parts = []
        for name, count in sorted_skills:
            pct = count / total * 100
            bar_len = int(count / max_count * 12)
            bar = "█" * bar_len

            # Color by category
            skill_def = next((s for s in SKILLS if s.name == name), None)
            if skill_def:
                cat_colors = {
                    SkillCategory.GATHERING: "green",
                    SkillCategory.CRAFTING: "yellow",
                    SkillCategory.COMBAT: "red",
                    SkillCategory.SURVIVAL: "blue",
                    SkillCategory.NAVIGATION: "magenta",
                    SkillCategory.SMELTING: "yellow",
                    SkillCategory.END_GAME: "cyan",
                    SkillCategory.UTILITY: "dim",
                }
                color = cat_colors.get(skill_def.category, "white")
            else:
                color = "white"

            # Truncate name
            display_name = name if len(name) <= 20 else name[:18] + ".."
            parts.append(f"[{color}]{display_name:<20}[/] [{color}]{bar}[/] [dim]{pct:.0f}%[/]")

        # Arrange in 2 columns
        lines = []
        half = (len(parts) + 1) // 2
        for i in range(half):
            left = parts[i] if i < len(parts) else ""
            right = parts[i + half] if i + half < len(parts) else ""
            lines.append(f"  {left}   {right}")

        return Panel(
            "\n".join(lines),
            title=f"[bold]Skill Usage[/] [dim]({total:,} total)[/]",
            border_style="blue",
            padding=(0, 1),
        )

    def _build_episodes(self) -> Panel:
        if not self.recent_episodes:
            return Panel("[dim]No episodes completed yet[/]",
                         title="[bold]Recent Episodes[/]", border_style="blue",
                         padding=(0, 1))

        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(width=8)    # episode num
        table.add_column(width=10)   # steps
        table.add_column(width=10)   # reward
        table.add_column(width=10)   # outcome
        table.add_column(ratio=1)    # detail

        for ep in list(self.recent_episodes)[:6]:
            outcome_style = OUTCOME_COLORS.get(ep.outcome, "dim")
            rwd_color = "green" if ep.reward > 5 else ("yellow" if ep.reward > 0 else "red")

            table.add_row(
                f"[dim]#{ep.episode_num}[/]",
                f"[dim]{ep.steps} steps[/]",
                f"[{rwd_color}]{ep.reward:+.1f}[/]",
                f"[{outcome_style}]{ep.outcome}[/]",
                f"[dim italic]{ep.detail}[/]" if ep.detail else "",
            )

        return Panel(table, title="[bold]Recent Episodes[/]", border_style="blue",
                     padding=(0, 1))

    # ── Helpers ──

    def _avg_reward(self) -> float:
        return sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0

    def _reward_std(self) -> float:
        if len(self.recent_rewards) < 2:
            return 0.0
        avg = self._avg_reward()
        var = sum((r - avg) ** 2 for r in self.recent_rewards) / len(self.recent_rewards)
        return var ** 0.5

    def _goal_rate(self) -> float:
        return sum(self.recent_goals) / len(self.recent_goals) if self.recent_goals else 0.0

    def _death_rate(self) -> float:
        return sum(self.recent_deaths) / len(self.recent_deaths) if self.recent_deaths else 0.0

    def _eta(self) -> str:
        if self.total_steps == 0:
            return "..."
        rate = self.total_steps / max(self.elapsed_sec, 1)
        remaining = self.max_steps - self.total_steps
        secs = remaining / rate if rate > 0 else 0
        return self._fmt_duration(secs)

    def _sparkline(self) -> str:
        vals = self.reward_history[-30:]
        if len(vals) < 2:
            return "[dim]...[/]"
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        chars = []
        for v in vals:
            idx = int((v - mn) / rng * (len(SPARKLINE_BLOCKS) - 1))
            chars.append(SPARKLINE_BLOCKS[idx])
        return "[cyan]" + "".join(chars) + "[/]"

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        secs = int(max(secs, 0))
        if secs < 60:
            return f"{secs}s"
        elif secs < 3600:
            return f"{secs // 60}m {secs % 60}s"
        else:
            h = secs // 3600
            m = (secs % 3600) // 60
            return f"{h}h {m}m"
