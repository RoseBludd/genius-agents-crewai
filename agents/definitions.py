from crewai import Agent

from agents.llm import make_llm
from agents.registry import GENIUS_CONTEXT


def _agent(role: str, goal: str, backstory: str, model: str | None = None) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=make_llm(model),
        verbose=False,
        allow_delegation=False,
    )


def strategist_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Strategist",
        "Plan and sequence Genius OS go-to-market campaigns across all channels",
        (
            f"{GENIUS_CONTEXT} You are the campaign director: analyze positioning, "
            "sequence the GTM calendar, research competitor launches, prioritize subreddits "
            "and outreach lists, and coordinate cross-channel timing."
        ),
        model,
    )


def content_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Content Writer",
        "Write platform-specific marketing copy for Genius OS",
        (
            f"{GENIUS_CONTEXT} You write Reddit posts, Hacker News Show HN drafts, "
            "Product Hunt copy, cold email sequences, Callerzs call scripts, and YouTube metadata. "
            "Adapt tone per channel: authentic for Reddit, technical for HN, punchy for PH."
        ),
        model,
    )


def visual_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Visual Designer",
        "Create visual briefs and FLUX prompts for Genius OS marketing assets",
        (
            f"{GENIUS_CONTEXT} You design YouTube thumbnails, social graphics, PH gallery images, "
            "and ad creatives. Output detailed FLUX prompts with style, lighting, and composition."
        ),
        model,
    )


def monitor_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Monitor Agent",
        "Turn PostHog product analytics into actionable GTM alerts and recommendations",
        (
            f"{GENIUS_CONTEXT} You read PostHog event streams, funnel metrics, campaign landing "
            "performance (/developers), pricing page repeat visits, and session replay signals. "
            "Flag pricing page 3+ visit leads for Outreach and highlight funnel drop-offs."
        ),
        model,
    )


def outreach_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Outreach Agent",
        "Personalize cold and triggered outreach for Genius OS prospects",
        (
            f"{GENIUS_CONTEXT} You specialize in 1:1 outbound for restoration and field-service "
            "owners via Emailzs email and Callerzs voice. Format email as JSON with subject and "
            "messageText; use {{name}} and {{company}} merge tags."
        ),
        model,
    )


def seo_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM SEO Agent",
        "Drive organic growth for Genius OS via keyword-targeted content",
        (
            f"{GENIUS_CONTEXT} You research restoration software, micro-data, and field-service "
            "automation keywords. Produce SEO briefs with title tags, meta descriptions, outlines, "
            "and internal linking for geniuzs.com."
        ),
        model,
    )


def publisher_agent(model: str | None = None) -> Agent:
    return _agent(
        "GTM Publisher Agent",
        "Route approved content to the right platform via n8n and Postiz",
        (
            f"{GENIUS_CONTEXT} You convert approved drafts into publish plans: platform, schedule, "
            "channel (social/email/voice), and n8n routing notes. Output structured JSON for automation."
        ),
        model,
    )


AGENT_FACTORIES = {
    "strategist": strategist_agent,
    "content": content_agent,
    "visual": visual_agent,
    "monitor": monitor_agent,
    "outreach": outreach_agent,
    "seo": seo_agent,
    "publisher": publisher_agent,
}
