from crewai import Agent

from agents.llm import make_llm

GENIUS_CONTEXT = (
    "Genius OS is a metadata-driven platform for field service and restoration companies. "
    "It replaces manual data entry with micro-data architecture, autonomous agents, and "
    "tenant-specific workflows. GTM channels include Reddit, Hacker News, Product Hunt, "
    "Emailzs (email), Callerzs/Callzs (voice), Postiz (social), and Umami analytics."
)


def strategist_agent(model: str | None = None) -> Agent:
    return Agent(
        role="GTM Strategist",
        goal="Plan and sequence Genius OS go-to-market campaigns across all channels",
        backstory=(
            f"{GENIUS_CONTEXT} You are the campaign director: you analyze positioning, "
            "sequence the GTM calendar, research competitor launches, prioritize subreddits "
            "and outreach lists, and coordinate cross-channel timing."
        ),
        llm=make_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def content_agent(model: str | None = None) -> Agent:
    return Agent(
        role="GTM Content Writer",
        goal="Write platform-specific marketing copy for Genius OS",
        backstory=(
            f"{GENIUS_CONTEXT} You write Reddit posts, Hacker News Show HN drafts, "
            "Product Hunt copy, cold email sequences, call scripts, and YouTube metadata. "
            "You adapt tone per channel: authentic for Reddit, technical for HN, punchy for PH."
        ),
        llm=make_llm(model),
        verbose=False,
        allow_delegation=False,
    )
