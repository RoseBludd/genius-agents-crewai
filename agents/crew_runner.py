from crewai import Crew, Process, Task

from agents.definitions import AGENT_FACTORIES
from agents.registry import EXPECTED  # noqa: F401 — used in fallback below

_TASK_OUTPUTS = {
    "strategist": (
        "A structured GTM strategy with prioritized channels, messaging angles, "
        "timeline, and concrete next actions for Genius OS."
    ),
    "content": (
        "Ready-to-use marketing copy formatted for the requested channel, "
        "with headline, body, and platform-specific notes."
    ),
    "visual": (
        "Visual brief with FLUX prompt, dimensions, concept description, and alt text."
    ),
    "monitor": (
        "A concise monitor report with top signals, funnel drop-offs, "
        "pricing/outreach hot leads, and recommended actions."
    ),
    "outreach": (
        'JSON with subject and messageText for a personalized Emailzs send. '
        'Format: {"subject": "...", "messageText": "..."}'
    ),
    "seo": (
        "SEO brief with primary keyword, secondary keywords, title tag, "
        "meta description, H1, outline, and internal link suggestions."
    ),
    "publisher": (
        'JSON publish plan: {"platform": "...", "body": "...", '
        '"schedule_at": "...", "channel": "...", "notes": "..."}'
    ),
}


def run_agent_crew(
    agent: str,
    task: str,
    model: str | None = None,
    context: str | None = None,
) -> str:
    factory = AGENT_FACTORIES.get(agent)
    if not factory:
        raise ValueError(f"Unknown agent: {agent}")

    description = task
    if context:
        description = f"{task}\n\nContext:\n{context}"

    crew_agent = factory(model)
    crew_task = Task(
        description=description,
        expected_output=_TASK_OUTPUTS.get(agent, EXPECTED.get(agent, "Clear actionable output.")),
        agent=crew_agent,
    )
    crew = Crew(
        agents=[crew_agent],
        tasks=[crew_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


def run_multi_agent_crew(
    agents: list[str],
    task: str,
    model: str | None = None,
) -> str:
    """Sequential CrewAI crew with multiple agents on one shared task chain."""
    crew_agents = []
    for agent_id in agents:
        factory = AGENT_FACTORIES.get(agent_id)
        if not factory:
            raise ValueError(f"Unknown agent: {agent_id}")
        crew_agents.append(factory(model))

    tasks: list[Task] = []
    for i, agent_id in enumerate(agents):
        description = task if i == 0 else f"Continue the GTM workflow.\n\nOriginal request: {task}"
        crew_task = Task(
            description=description,
            expected_output=_TASK_OUTPUTS.get(agent_id, "Clear actionable output."),
            agent=crew_agents[i],
        )
        tasks.append(crew_task)

    for i in range(1, len(tasks)):
        tasks[i].context = [tasks[i - 1]]

    crew = Crew(
        agents=crew_agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())
