from crewai import Crew, Process, Task

from agents.definitions import content_agent, strategist_agent


def run_strategist_task(task: str, model: str | None = None) -> str:
    agent = strategist_agent(model)
    crew_task = Task(
        description=task,
        expected_output=(
            "A structured GTM strategy with prioritized channels, messaging angles, "
            "timeline, and concrete next actions for Genius OS."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[crew_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


def run_content_task(task: str, model: str | None = None) -> str:
    agent = content_agent(model)
    crew_task = Task(
        description=task,
        expected_output=(
            "Ready-to-use marketing copy formatted for the requested channel, "
            "with headline, body, and platform-specific notes."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[crew_task],
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())
