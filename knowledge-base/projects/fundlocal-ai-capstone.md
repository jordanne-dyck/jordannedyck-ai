---
title: "FundLocal - AI Product Management Capstone"
date: "2026"
status: "Completed"
role: "AI Product Manager (Capstone Project)"
company: "Maven AI Product Management Certification"
tags: ["ai", "product-management", "fintech", "capstone", "certification", "agentic-ai", "rag", "community-investment"]
impact_metrics: ["compliance_cost_reduction", "unit_economics", "market_sizing"]
technologies: ["Claude API", "Agentic AI", "RAG", "Compliance Automation", "Values Matching"]
context_type: "project_detail"
context_priority: "high"
---

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 1.0
chunk_id: fundlocal-001
```
# FundLocal - AI-Native Community Investment Feature

## Executive Summary

Capstone project for Maven's AI Product Management Certification. Designed **FundLocal** — an AI-native community investment feature for Wealthsimple that enables values-conscious retail investors to discover, evaluate, and fund community projects aligned with their personal values. Applied the full 4D product methodology (Discovery → Design → Develop → Deploy) to take the concept from market research through working prototype with live AI demo.

**Key Insight**: Rather than trying to improve an existing system with AI, I went looking for something broken at a deeper level. Community investing is time consuming, unreliable, disconnected, and difficult to scale. The closest competitor (Mainvest) shut down in 2024 because manual compliance review costs ($800–$3,200 per proposal) made the unit economics unworkable. AI reduces this to $40–$80 — a 10–40x cost reduction that makes the business model viable. But the bigger opportunity isn't just operational — it's conceptual.

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_weight: 0.7
chunk_id: fundlocal-002
```
## The Problem

**The problem**: People want to put their money into things they care about — with real accountability and outcomes beyond just financial returns. The demand is there ($629B impact investing market → $1.54T projected, 97% of Millennials want sustainable investing), but trust isn't — 85% of investors see greenwashing as a growing problem because current options are opaque. You pick an "ESG" portfolio but never know what your money actually funds. The alternatives don't solve it either: GoFundMe is donations with no returns and no accountability for outcomes, government grants are slow and disconnected from individual investors, and community bonds require financial expertise most people don't have. None combine accessibility, returns, and real accountability. The infrastructure for everyday people to invest in specific, visible community projects simply doesn't exist.


<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.9
chunk_id: fundlocal-003
```
## AI Solution: Three MVP Components

**1. AI Values Matching Engine** (Impact: 5, Feasibility: 4)
- Reads unstructured project proposals, scores against investor values across 4 dimensions (Values, Location, Returns, Credibility)
- Natural-language reasoning explains why each project matches — and surfaces limitations
- Not just filters — fuzzy matching on human values

**2. Agentic Compliance Verification** (Impact: 5, Feasibility: 4)
- AI calls external registries (CRA, municipal permits), interprets results, flags discrepancies
- Reduces review cost from $800–$3,200 to $40–$80 per proposal (10–40x reduction)
- This is what makes the business model viable — without AI, the economics don't work

**3. AI Proposal Coaching** (Impact: 4, Feasibility: 5)
- Evaluates creator proposals and gives specific, actionable coaching
- Surfaces unverified claims ("partnership claimed, no evidence")
- Raises proposal quality upstream, reducing compliance burden downstream

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.9
chunk_id: fundlocal-004
```
## Product Process Applied

**Discovery**: Business Value Map, competitive analysis (Mainvest failure analysis), persona development (Maya Chen — 34, UX Designer, Wealthsimple user), 5-stage user journey mapping, pain point sizing using 5-question framework (Magnitude × Frequency × Severity × Competition × Contrast)

**Design**: UX flows and [wireframes](https://gist.github.com/jordanne-dyck/b77a1bac05923e0033a04d9e29e38108), applied Sheridan's Levels of Autonomy (Level 5: AI suggests, human decides), Overton Window for AI acceptance, human-in-the-loop at every critical decision point, 3Ps Framework (Prioritization, Placement, Prominence)

**Develop**: Live working prototype with Claude API — not mockups. [Investor Experience](https://fundlocal-ljmo.vercel.app/) (values intake → AI matching → project detail → portfolio), [Compliance Experience](https://fundlocal-ljmo.vercel.app/?view=compliance) (proposal builder → AI evaluation → coaching). Prompt engineering with versioning. Evaluation criteria defined upfront.

**Deploy**: Staged rollout plan (MVP → V1.1 with milestone verification → V2 with predictive scoring), operational readiness checklist, metrics framework (business + AI + product), feedback engine design

## Prioritization: Diverge → Score → Converge

[Customer Journey Map & Prioritization Matrix](https://lucid.app/lucidspark/b7cb54b4-1e18-4bf4-922a-3844b51a801f/view?page=0_0&invitationId=inv_649bd4d6-f9ea-4f55-8708-b4e3ddc94ae0#)

Generated 10 solution ideas, scored on Impact (1–5) × Feasibility (1–5):
- **MVP** (Score 20 each): Values Matching, Compliance Verification, Proposal Coaching
- **V1.1**: Automated Milestone Verification, AI-Generated Impact Narratives, Social Proof
- **V2** (requires data/scale): Predictive Success Scoring, Community Sentiment Analysis

Selection rationale: The three MVP ideas address the two highest-severity pain points (discovery 25/25, due diligence 24/25), are technically feasible today, and create a defensible moat.

## Key Metrics Framework

**Business**: Adoption rate, retention/repeat investment, conversion, capital deployed, time to first investment
**AI**: Values matching accuracy, hallucination rate, compliance false positive/true positive rates, explanation quality
**Product**: Time to complete values intake (~1 min target), user confidence in recommendations, milestone completion rate

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.8
chunk_id: fundlocal-005
```
## Frameworks & Skills Demonstrated

This project demonstrates applied knowledge of:

- **4D Product Methodology**: Discovery → Design → Develop → Deploy
- **Pain Sizing Framework**: 5-question scoring (Magnitude × Frequency × Severity × Competition × Contrast)
- **Diverge → Score → Converge**: Structured ideation and prioritization
- **Sheridan's Levels of Autonomy**: Appropriate AI intervention levels
- **Human-in-the-Loop Design**: AI augments, humans decide
- **Overton Window**: Public acceptance spectrum for AI features
- **AI PRD Components**: 12-element structured PRD
- **Agentic AI Architecture**: Multi-step agentic patterns (matching, compliance verification, coaching)
- **RAG System Design**: Knowledge retrieval for project evaluation
- **Prompt Engineering**: Structured inputs, versioning, treating prompts like code
- **Evaluation Design**: Defining "good" upfront, quality checklists, regression prevention
- **Staged Rollout Planning**: Pilot → staged → full release with rollback capability

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.9
chunk_id: fundlocal-006
```
## Maven AI PM Certification Context

The Maven AI Product Management course provided structured frameworks for building AI products:
- **Prompts for reasoning, RAG for knowledge storage** — separation of concerns in AI systems
- **Evaluation sets as insurance** — protecting against quality degradation as you iterate
- **Model selection tradeoffs** — cost/speed/quality evaluation
- **Adversarial testing** — red teaming for PII leaks, prompt injection, harmful content
- **Learning loops** — monthly prompt reviews, quarterly eval updates, biweekly tuning hours
- **Data security** — residency, zero retention policies, secure handling

<!--chunk:end-->

---

## Tags for AI Retrieval

**Project Type**: AI product management, fintech, community investment, capstone, certification
**Technologies**: Claude API, agentic AI, compliance automation, values matching, RAG
**Skills**: AI product strategy, user research, prioritization frameworks, prototype development, evaluation design
**Impact**: 10-40x compliance cost reduction, viable unit economics for community investment
**Innovation**: AI-native approach to community investment matching and compliance
**Certification**: Maven AI Product Management (2026)
