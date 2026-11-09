"""Model prices. Ported 2026-11-07 from jordannedyck-ai-web/lib/cost.ts."""

# Per-1M-token prices in USD (input, output). Source: OpenAI pricing page.
PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
}


def estimate_cost_usd(model, prompt_tokens, completion_tokens):
    """estimateCostUsd(): 0 for a model not in PRICING, as the site does."""
    price = PRICING.get(model)
    if not price:
        return 0
    return (prompt_tokens / 1_000_000) * price[0] + (completion_tokens / 1_000_000) * price[1]
