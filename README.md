# multi-agent-research-hub
Useful thoughts and experience about the MAS developing 
https://cognition.ai/blog/dont-build-multi-agents#principles-of-context-engineering 
<details>
  <summary>Overview of “Don’t Build Multi-Agents”</summary>

  The article argues that parallel multi-agent chains are fragile because context fragments and implicit decisions get siloed, causing errors to compound. Instead, it introduces **Context Engineering**—sharing the full trace of prior actions and treating each action as carrying hidden assumptions—and shows that the most reliable pattern is a **single-threaded linear agent**, possibly augmented with a **history-compressor** LLM to summarize long interactions into key events without overflowing context windows.

  **Main conclusion:**  
  For robust, long-running AI agents, avoid parallel multi-agent setups and focus on seamless context management—either via one coherent agent or via intelligent history compression—so that every decision is consistently informed by the complete task context.
</details>

https://www.anthropic.com/engineering/built-multi-agent-research-system
<details>
  <summary>Overview of “How we built our multi-agent research system”</summary>

  This article describes Anthropic’s Research feature, which leverages a lead Claude agent to orchestrate multiple parallel subagents for complex, open-ended research tasks—allowing dynamic, breadth-first exploration of information by distributing work across specialized agents with separate context windows. :contentReference[oaicite:0]{index=0}

  **Main conclusion:**  
  Building robust, production-grade multi-agent systems requires careful engineering around orchestration patterns, prompt and tool design, evaluation frameworks, and operational practices (like fault recovery and phased deployments). When done correctly, these systems can dramatically accelerate research workflows, but the gap between prototype and reliable production is often wider than anticipated. :contentReference[oaicite:1]{index=1}
</details>
