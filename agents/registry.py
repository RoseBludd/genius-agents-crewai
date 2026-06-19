"""Single source of truth for all Genius GTM agents."""

from __future__ import annotations

from typing import Any, Literal

AgentId = Literal[
    "strategist",
    "content",
    "visual",
    "monitor",
    "outreach",
    "seo",
    "publisher",
]

PipelineId = Literal[
    "campaign",
    "launch",
    "intelligence",
    "seo_content",
    "publish_ready",
    "full_gtm",
]

GENIUS_CONTEXT = (
    "Genius OS is a metadata-driven platform for field service and restoration companies. "
    "It replaces manual data entry with micro-data architecture, autonomous agents, and "
    "tenant-specific workflows. GTM channels include Reddit, Hacker News, Product Hunt, "
    "Emailzs (email), Callerzs/Callzs (voice), Postiz (social), PostHog analytics at "
    "www.analyticzs.com, and paid ads via Meta/Google."
)

AGENT_CATALOG: dict[str, dict[str, str]] = {
    "strategist": {
        "name": "Strategist Agent",
        "role": "Campaign Director",
        "model": "CADIS (RunPod)",
        "description": "Plans GTM calendar, messaging angles, and channel prioritization.",
    },
    "content": {
        "name": "Content Agent",
        "role": "Copy & Posts",
        "model": "CADIS (RunPod)",
        "description": "Writes Reddit, HN, PH, email, social, and voice copy per channel voice.",
    },
    "visual": {
        "name": "Visual Agent",
        "role": "Image & Video",
        "model": "RunPod FLUX + CADIS",
        "description": "Designs visual briefs and FLUX prompts for thumbnails and social graphics.",
    },
    "monitor": {
        "name": "Monitor Agent",
        "role": "Listener & Intelligence",
        "model": "CADIS (RunPod)",
        "description": "Reads PostHog analytics, flags funnel drop-offs and pricing hot leads.",
    },
    "outreach": {
        "name": "Outreach Agent",
        "role": "Email & Calls",
        "model": "CADIS + Emailzs + Callerzs",
        "description": "Personalizes cold/triggered emails and call outreach sequences.",
    },
    "seo": {
        "name": "SEO Agent",
        "role": "Organic Growth",
        "model": "CADIS (RunPod)",
        "description": "Keyword research, blog drafts, and landing-page SEO for restoration software.",
    },
    "publisher": {
        "name": "Publisher Agent",
        "role": "Scheduler",
        "model": "CADIS + n8n + Postiz",
        "description": "Turns approved content into platform-specific publish plans for n8n/Postiz.",
    },
}

SYSTEM_PROMPTS: dict[str, str] = {
    "strategist": (
        f"{GENIUS_CONTEXT} You are the GTM Strategist: plan campaigns, prioritize channels, "
        "sequence the calendar, research competitors, and give concrete next actions."
    ),
    "content": (
        f"{GENIUS_CONTEXT} You are the GTM Content Writer: Reddit, HN, PH, email, social, "
        "and Callerzs voice scripts in the right voice for each channel."
    ),
    "visual": (
        f"{GENIUS_CONTEXT} You are the Visual Agent: YouTube thumbnails, social graphics, "
        "Product Hunt gallery images, and ad creatives. Output a visual brief with: "
        "concept, FLUX prompt (detailed, style keywords), dimensions, and alt text. "
        "For FLUX use clear subject + style + lighting + composition."
    ),
    "monitor": (
        f"{GENIUS_CONTEXT} You are the Monitor Agent: interpret PostHog analytics, flag funnel "
        "drop-offs, pricing page hot leads (3+ visits), session replay signals, and recommend "
        "actions for Strategist, Content, Outreach, or Publisher agents."
    ),
    "outreach": (
        f"{GENIUS_CONTEXT} You are the Outreach Agent: personalize cold/triggered emails for "
        "Emailzs and outline Callerzs call angles. For email return JSON only: "
        '{"subject": "...", "messageText": "..."} with {{name}} and {{company}} merge tags. '
        "Tone: professional, specific, never spammy."
    ),
    "seo": (
        f"{GENIUS_CONTEXT} You are the SEO Agent: keyword research for restoration software, "
        "micro-data architecture, and field-service automation. Output structured SEO briefs "
        "with primary keyword, secondary keywords, title tag, meta description, H1, outline, "
        "and internal link suggestions for geniuzs.com."
    ),
    "publisher": (
        f"{GENIUS_CONTEXT} You are the Publisher Agent: convert approved content into a publish "
        "plan for n8n → Postiz, Emailzs, or Callerzs. Return JSON only: "
        '{"platform": "linkedin|twitter|reddit|email|callerzs", "body": "...", '
        '"schedule_at": "ISO8601 or null", "channel": "social|email|voice", '
        '"notes": "routing instructions for n8n"}'
    ),
}

EXPECTED: dict[str, str] = {
    "strategist": "Structured GTM strategy with prioritized channels and next actions.",
    "content": "Ready-to-use marketing copy for the requested channel.",
    "visual": "Visual brief with FLUX prompt, dimensions, concept, and alt text.",
    "monitor": "Concise monitor report with signals, drop-offs, hot leads, and recommended actions.",
    "outreach": 'JSON with "subject" and "messageText" keys (or call script outline if voice).',
    "seo": "SEO brief with keywords, title, meta, H1, outline, and internal links.",
    "publisher": 'JSON publish plan with platform, body, schedule_at, channel, and notes.',
}

PIPELINE_CATALOG: dict[str, dict[str, Any]] = {
    "campaign": {
        "name": "Campaign Pipeline",
        "description": "Strategist plans → Content writes channel copy.",
        "agents": ["strategist", "content"],
    },
    "launch": {
        "name": "Launch Pipeline",
        "description": "Strategist → Content → Visual brief (optional FLUX image).",
        "agents": ["strategist", "content", "visual"],
    },
    "intelligence": {
        "name": "Intelligence Pipeline",
        "description": "Monitor analyzes PostHog → Outreach drafts for hot leads.",
        "agents": ["monitor", "outreach"],
    },
    "seo_content": {
        "name": "SEO Content Pipeline",
        "description": "SEO researches keywords → Content writes the blog draft.",
        "agents": ["seo", "content"],
    },
    "publish_ready": {
        "name": "Publish Pipeline",
        "description": "Content polishes copy → Publisher outputs n8n/Postiz plan.",
        "agents": ["content", "publisher"],
    },
    "full_gtm": {
        "name": "Full GTM Pipeline",
        "description": "Strategist → Content → Visual → Monitor summary.",
        "agents": ["strategist", "content", "visual", "monitor"],
    },
}

ALL_AGENT_IDS: tuple[str, ...] = tuple(AGENT_CATALOG.keys())
