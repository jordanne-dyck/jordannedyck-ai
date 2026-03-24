---
title: "AI-Powered Resume & Portfolio System"
date: "2025 - Present"
status: "In Production"
role: "Creator & Developer"
company: "Personal Project"
parent_initiative: "personal-portfolio"
tags: ["ai", "mcp", "claude", "rag", "vector-search", "portfolio", "personal-branding", "developer-tools"]
impact_metrics: ["engagement_quality", "time_efficiency", "context_accuracy", "professional_brand"]
technologies: ["Claude Desktop", "MCP (Model Context Protocol)", "Vector Embeddings", "RAG", "Python", "Markdown"]
context_type: "project"
context_source: "jordanne-dyck"
context_version: "v1.0"
---

<!--chunk:start-->
```yaml
context_priority: critical
embedding_scope: global
embedding_weight: 1.0
chunk_id: proj-ai-resume-001
```
# AI-Powered Resume & Portfolio System

## Overview
A custom Model Context Protocol (MCP) server that transforms my professional experience, skills, and projects into an AI-searchable knowledge base. This enables Claude to provide contextually relevant information about my background, making conversations with recruiters, hiring managers, and collaborators more efficient and accurate.

**The Challenge:** Traditional resumes are fundamentally broken. They're static, one-dimensional documents that force you to compress years of complex experience into bullet points and buzzwords. They tell people about your skills rather than showing them. I found myself in conversations saying "I have experience with AI/ML" or "I'm good at product thinking" without being able to demonstrate those capabilities in real-time. 

I needed a resume that more accurately depicted my actual experience—one that could provide relevant context for any conversation, adapt to different audiences, and most importantly, prove my capabilities through its very existence rather than just claiming them.

**My Thought Process:** This is how I approach most problems:
1. **Identify the real problem**: Not "I need a better resume," but "How do I demonstrate capabilities vs. just talking about them?"
2. **Reimagine possibilities with new technology**: Don't accept today's constraints as permanent. AI is fundamentally changing how we interact with information. If we can have conversational interfaces with documents, why should a resume be a static PDF? What technology, process, or people solution could transform this experience? This pattern recognition—seeing how new capabilities can evolve traditional experiences—is how I identify opportunities before they're obvious.
3. **Prototype quickly**: Built a minimal viable version in 5 days to test the concept.
4. **Gather diverse feedback**: Sent it to friends and family to test with real conversations and different perspectives
5. **Iterate based on real use**: Used it in actual conversations with recruiters and refined based on what worked
6. **Build for extensibility**: Designed it to grow with my experience and expand to new use cases


**The Solution:** Built a custom MCP server that integrates with Claude Desktop, allowing AI-powered semantic search across my entire professional history. This creates a dynamic, conversational interface to my experience that adapts to the context of each query. The project itself demonstrates the exact skills I'm trying to showcase—AI implementation, product thinking, rapid prototyping, and problem-solving.

**The Impact:** 
- Enables instant, contextually relevant responses about my experience
- Provides detailed technical context without overwhelming conversations
- Demonstrates practical AI implementation and developer capabilities
- Showcases ability to identify problems and build custom solutions
- Proves I can ship functional products quickly (5 days from concept to working prototype)
<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_scope: global
embedding_weight: 1.0
chunk_id: proj-ai-resume-002
```
## Problem Statement

### The Traditional Portfolio Problem
**Static & Inflexible**: Traditional resumes are one-size-fits-all documents that can't adapt to different audiences or contexts. A conversation about AI capabilities requires different details than one about product management.

**Context Loss**: When discussing projects, important details are often scattered across multiple documents, making it difficult to provide comprehensive yet relevant information quickly.

**Time-Intensive**: Manually searching through past work, code repositories, and documentation to answer specific questions about experience is inefficient and breaks conversational flow.

**Limited Demonstration**: A static portfolio doesn't demonstrate technical capabilities or problem-solving approach in action—it just describes them.

### The Opportunity
With advances in AI and tools like Claude Desktop's MCP, there's an opportunity to create a truly intelligent portfolio system that:
- Understands semantic meaning, not just keywords
- Provides relevant context automatically
- Demonstrates technical capabilities through implementation
- Adapts to different conversation contexts
<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_scope: global
embedding_weight: 0.9
chunk_id: proj-ai-resume-003
```
## Solution Architecture

### Technical Approach
Built a custom MCP server that processes my professional experience into a searchable knowledge base:

**1. Knowledge Base Structure**
- Professional overview and career narrative
- Detailed project documentation (10+ major projects)
- Technical skills and expertise
- Work style and personality traits
- Structured in markdown with metadata and chunking

**2. Vector Search Implementation**
- Embedded content using vector embeddings for semantic search
- Implemented context-aware chunking to maintain meaning
- Optimized chunk sizes for relevance vs. context trade-off
- Metadata tagging for filtering (projects, skills, experience)

**3. MCP Server Integration**
- Custom Python-based MCP server
- Exposed search function to Claude Desktop
- Supports parameterized queries (query, n_results)
- Returns contextually relevant chunks with metadata

**4. Content Organization**
Structured knowledge base hierarchy:
```
knowledge-base/
├── experience/
│   └── professional-overview.md
├── projects/
│   ├── ai-customer-service-optimization.md
│   ├── agentic-personal-shopper.md
│   ├── google-gemini-enterprise-deployment.md
│   └── [8+ more projects]
├── personality/
│   └── work-style-and-personality.md
└── technical-skills.md
```
<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_scope: global
embedding_weight: 0.9
chunk_id: proj-ai-resume-004
```
## Implementation Details

### Key Technical Decisions

**MCP over Traditional API**
- Leverages Claude Desktop's native integration
- Simpler authentication and setup
- Better for personal use cases
- Demonstrates understanding of emerging AI tooling

**Vector Search vs. Keyword Search**
- Semantic understanding captures intent, not just matching words
- "AI projects" finds relevant work even when labeled differently
- Better handles ambiguous or exploratory queries
- Returns contextually relevant results ranked by similarity

**Markdown-Based Content**
- Human-readable and version-controllable
- Easy to update and maintain
- Supports rich formatting and metadata
- Can be used for multiple purposes (resume, portfolio, documentation)

**Chunking Strategy**
- Balanced chunk sizes (200-500 words) for context vs. precision
- Preserved semantic boundaries (sections, paragraphs)
- Included metadata for each chunk (priority, scope, weight)
- Maintained relationships between related content

**Eval-Driven Development**
- Built evaluation sets with retrieval and response quality criteria
- Used eval-driven iteration: define expected outcomes, measure, improve, re-measure
- Automated scoring across retrieval accuracy, response inclusion, and LLM-judged qualitative criteria
- Context engineering: designed knowledge base structure, chunking strategies, and metatag enrichment that directly shape AI output quality

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: high
embedding_scope: global
embedding_weight: 0.8
chunk_id: proj-ai-resume-005
```
## Results & Impact

### Measurable Outcomes

**Efficiency Gains**
- Instant retrieval of relevant experience details vs. 5-10 minutes of manual searching
- Comprehensive context provided in single query
- No need to reference multiple documents during conversations

**Quality Improvements**
- More contextually relevant information shared
- Consistent narrative across different conversations
- Technical details readily available when needed
- Better demonstration of capabilities through implementation

**Validation Through Real Use**
- Tested with friends and family across different conversation types
- Used in actual recruiter and networking conversations
- Gathered diverse feedback revealing unexpected use cases
- Refined based on real-world patterns vs. theoretical assumptions

**Professional Branding**
- Demonstrates practical AI/ML implementation skills
- Shows product thinking (identifying problem, building solution)
- Exhibits technical competency in emerging tools
- Highlights ability to learn and adapt quickly

<!--chunk:end-->

<!--chunk:start-->
```yaml
context_priority: medium
embedding_scope: global
embedding_weight: 0.7
chunk_id: proj-ai-resume-008
```
## Why This Project Matters

### Show, Don't Tell
The fundamental insight behind this project: **anyone can claim to have skills, but building something proves you have them.**

Instead of saying "I have experience with AI/ML," this project demonstrates:
- Actual implementation of RAG architecture
- Understanding of vector embeddings and semantic search
- Ability to integrate with emerging AI tools (MCP)
- Product thinking from problem identification to solution

Instead of saying "I'm a fast learner," the 5-day build timeline proves it. Instead of saying "I can identify and solve problems," the very existence of this project shows my problem-solving process in action.

### Demonstrates Key Capabilities

**Technical Skills**
- AI/ML implementation (RAG, vector embeddings)
- Python development
- API/tool development (MCP)
- System architecture and design

**Product Thinking**
- Identified real problem from personal experience (resumes that show vs. tell)
- Recognized pattern: AI enables conversational interfaces that can transform static experiences
- Questioned fundamental assumptions (why should resumes be static documents?)
- Designed solution balancing complexity and utility
- Gathered feedback from diverse audiences (friends, family, recruiters)
- Iterated based on real-world usage
- Built for extensibility and future enhancement

**Problem-Solving Approach**
- Start with the actual problem, not the assumed solution
- Question existing assumptions about how things "should" work
- Recognize patterns: how can new technologies fundamentally transform experiences?
- Don't confine thinking to today's constraints—AI changes what's possible
- Prototype quickly to test hypotheses (5 days to MVP)
- Gather diverse feedback from different perspectives
- Ship and iterate based on real-world feedback
- Think meta: the solution itself can demonstrate the capability

**Learning Agility**
- Adopted new technology (MCP) within days of its release
- Self-directed learning and implementation
- Practical application of theoretical knowledge
- Continuous improvement mindset

**Professional Development**
- Investment in personal brand and positioning
- Forward-thinking approach to career tools
- Willingness to experiment and share learnings
- Understanding of AI's impact on professional work

### Meta-Value
This project itself demonstrates the capabilities it describes. By using it in conversations about my experience, I'm simultaneously:
- Showing technical implementation skills (not just talking about them)
- Demonstrating product thinking (identified problem, built solution)
- Proving ability to identify and solve problems (the project is the proof)
- Exhibiting learning agility with new tools (5-day build with brand new technology)

It's a recursive portfolio piece—a tool that helps explain why the tool exists and what it says about my capabilities. The resume demonstrates the very skills it's designed to showcase.
<!--chunk:end-->

## Technologies Used
- **AI/ML**: Vector embeddings, semantic search, RAG architecture
- **Development**: Python, MCP (Model Context Protocol)
- **Data**: Markdown, structured metadata, chunking strategies
- **Tools**: Claude Desktop, embedding models
- **Architecture**: Client-server model, function-based API