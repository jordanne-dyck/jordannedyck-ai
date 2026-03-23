---
title: "Agentic Personal Shopper - AI Agent System"
date: "2024 - 2025"
status: "In Production"
role: "Product Lead & AI Architect"
company: "DECIEM"
parent_initiative: "abnormal-innovation"
tags: ["ai", "agentic-ai", "conversational-commerce", "customer-experience", "salesforce-agentforce", "abnormal-innovation"]
impact_metrics: ["conversion_rate", "customer_satisfaction", "aov", "return_rate"]
technologies: ["Salesforce Agentforce", "LLMs", "Multi-agent Systems", "Human-in-the-loop"]
context_type: "project_detail"
context_priority: "critical"
---

<!--chunk:start-->
```yaml
context_priority: critical
embedding_weight: 1.0
chunk_id: shopper-001
```
# Agentic Personal Shopper - AI Agent System

## Executive Summary

**Concept to production to global scale.** Personally built the first working version — unifying product discovery, order management, and checkout into a single conversational experience. Tested with pilots, identified failure modes, and iterated fast before scaling. Defined AI operating model for continuous optimization post-production. Concept to production in 6 months. Featured at NRF 2025.

Built a unified agentic AI personal shopping system that orchestrates multiple specialized agents to guide customers through product discovery, education, routine building, and post-purchase support. Featured by Salesforce at NRF 2025 as a pioneering implementation of autonomous commerce.

**Key Innovation**: Multiple AI agents working together seamlessly while maintaining human connection and brand voice through strategic human-in-the-loop design.

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_weight: 0.7
chunk_id: shopper-002
```
## The Problem & Opportunity

### Customer Confusion at Scale

**Customer Pain Points:**
- Overwhelmed by 100+ products across 15+ categories
- Confused about ingredient compatibility and routine order
- Unsure which products address specific skin concerns
- Time-consuming to research and compare options
- Post-purchase questions about usage and results

**Business Challenge:**
- High cart abandonment due to decision paralysis
- Lower average order value (customers buying single products vs. routines)
- Increased customer service volume for basic questions
- Low retention rates from incorrect product selection and/or confusion on how to use
- Inconsistent customer experience across channels

**The Opportunity:**
Create AI-powered personal shopper that feels like talking to a knowledgeable beauty advisor in-store, but at scale online. Evolution of the rules-based Regimen Builder tool using agentic AI.

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.9
chunk_id: shopper-003
```
## Solution Design

### Multi-Agent Architecture

Built **unified system** where specialized agents collaborate through shared customer context:

**Agent Roles:**

**1. Product Recommendation Agent**
- Understands skin concerns and suggests personalized products
- Product catalog knowledge (ingredients, benefits, compatibility)
- Personalization based on skin type, concerns, preferences

**2. Regimen Builder Agent**
- Guides through creating multi-step skincare routines
- Routine architecture (cleanse → treat → moisturize → protect)
- Layering order logic and AM/PM differentiation

**3. Education Agent**
- Explains ingredients, benefits, science behind recommendations
- Sets realistic expectations for results
- Science-backed information without overwhelming jargon

**4. Order Management Agent**
- Places orders
- Handles post-purchase order tracking, modifications, returns
- Autonomous order status lookup and issue resolution
- Escalates complex issues to human agents

**5. FAQ & Support Agent**
- Handles common questions autonomously
- Product usage instructions and ingredient safety
- Smart escalation when needed

### Human-in-the-Loop Design

**Philosophy**: AI enhances human connection, doesn't replace it.

**Human Involvement:**
- Complex or sensitive skin conditions requiring medical judgment
- Emotional or frustrated customers needing empathy
- Edge cases outside agent training
- Brand reputation risk scenarios
- VIP customer relationships

**Escalation Triggers:**
- Customer explicitly requests human agent
- Sentiment analysis detects frustration
- Agent confidence below threshold
- Medical or safety concerns detected

### Implementation Approach

**Build vs. Buy Testing:**
- Built MVP of custom conversational product recommender
- Simultaneously piloted Salesforce Agentforce
- Decision: Agentforce for production (faster, robust, scalable, cheaper)

**Phased Rollout:**
- Started with "Where is my order" agent (high impact, low LOE)
- Gradually added more features and functionality
- Piloted to smaller market (Canada) ahead of global rollout
- Full unified experience in rollout at time of departure

**Change Management:**
- Trained customer service team on new workflows
- Established monitoring and escalation protocols
- Regular review sessions with stakeholders

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_weight: 0.9
chunk_id: shopper-004
```
## Results & Impact

- Agents handled a growing share of interactions autonomously at time of departure; humans focused on high-value, high-empathy interactions.
- Improved customer experience
- Improved retention rate
- Featured at Salesforce NRF 2025 (showcase for innovation)
- Competitive differentiation in customer experience
- Generated insights on customer concerns and product gaps
- Enhanced brand perception as digital-first and innovative
- Laid foundation for scaling personal shopper experience to new channels (ie. whatsapp, chatGPT, marketing channels, etc.)

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_weight: 0.6
chunk_id: shopper-005
```
## Challenges & Learnings

### What Worked Well

✅ **Multi-agent architecture**: Specialized agents perform better than single general-purpose bot
✅ **Human-in-the-loop**: Maintains empathy and handles edge cases gracefully
✅ **Shared context**: Agents hand off seamlessly without customer repeating information
✅ **Brand voice training**: Extensive examples kept responses on-brand
✅ **Iterative rollout**: Gradual deployment allowed learning and refinement

### Challenges Overcome

**Medical Claims Compliance**
- Problem: Beauty industry has strict regulations on health claims
- Solution: Built guardrails preventing medical advice, trained agents to use compliant language, escalate uncertain cases
- Result: Zero compliance issues, protected brand reputation

**Complex Product Interactions**
- Problem: Some ingredient combinations require nuanced guidance
- Solution: Encoded compatibility matrix, flagged complex cases for human review, continuous refinement with dermatologist input, collaborated with internal experts to create an aligned knowledge base
- Result: Safe, effective recommendations even for advanced routines

**Edge Cases and Optimizations**
- Problem: Sometimes customers or agents do things we don't anticipate
- Solution: Implemented continuous monitoring and optimization process, fallback to human when uncertain
- Result: Reliability improved, customer and brand trust maintained

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_weight: 0.5
chunk_id: shopper-006
```
## Skills Demonstrated

**AI Product Strategy**: Identified opportunity, evaluated build vs. buy, designed multi-agent architecture aligned with customer journey

**Technical Architecture**: Designed system integrating multiple agents, data sources, and human handoffs

**Agentic AI Implementation**: Hands-on experience with Salesforce Agentforce, multi-agent systems, LLM integration, human-in-the-loop design

**Customer Experience Design**: Deep understanding of customer pain points, conversation design, balancing automation with human touch

**Change Management**: Trained teams on new workflows, managed stakeholder expectations, drove adoption

**Business Acumen**: Connected AI capabilities to business outcomes (conversion, AOV, satisfaction), articulated ROI

<!--chunk:end-->

---

## Tags for AI Retrieval

**Project Type**: Customer-facing AI, conversational commerce, multi-agent system, agentic AI
**Technologies**: Salesforce Agentforce, LLMs, agentic AI, human-in-the-loop, multi-agent orchestration
**Skills**: AI product strategy, technical architecture, customer experience, change management
**Impact**: conversion improvement, AOV increase, ticket reduction
**Innovation**: First-in-industry unified personal shopper, NRF 2025 showcase
**Parent Initiative**: Abnormal Innovation