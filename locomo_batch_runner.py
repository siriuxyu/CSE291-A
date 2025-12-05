#!/usr/bin/env python
"""
LoCoMo batch runner that minimizes LLM invocations.

For each persona we:
1) Send a single call that contains all of their conversation transcripts so the
   agent can capture the long-term context (LangMem).
2) Reset short-term state and send a single call that contains all QA questions
   for that persona, asking the agent to answer them in JSON.

This keeps the number of Claude calls to two per persona instead of per turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from benchmark_runner import (
    load_json,
    normalize_text,
    call_agent,
    reset_short_term_session,
)

# Token limits (approximate - 1 token ≈ 4 characters)
# Claude rate limits: 30K input tokens/min for Sonnet/Opus, 50K for Haiku
# With 2s delay between calls, we can do ~15 requests/min
# Budget per request: ~2000 tokens to stay safe
DEFAULT_MAX_TRANSCRIPT_CHARS = 6000  # ~1500 tokens
DEFAULT_MAX_QA_CHARS = 4000  # ~1000 tokens


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def throttle(delay_sec: float) -> None:
    if delay_sec > 0:
        time.sleep(delay_sec)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    
    # Try to break at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    break_point = max(last_period, last_newline)
    if break_point > max_chars * 0.7:  # Only use if we keep at least 70%
        truncated = truncated[:break_point + 1]
    
    return truncated + "\n[... truncated for length ...]"


def build_transcript(
    conversations: List[List[Dict[str, Any]]],
    max_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> str:
    """Build transcript with optional length limit."""
    sections: List[str] = []
    total_chars = 0
    
    for idx, convo in enumerate(conversations, start=1):
        lines = [f"[Session {idx}]"]
        session_chars = len(lines[0])
        
        for turn in convo or []:
            speaker = turn.get("speaker") or turn.get("role") or "unknown"
            content = turn.get("content", "")
            dia_id = turn.get("dia_id")
            prefix = f"[{dia_id}] " if dia_id else ""
            
            line = f"{prefix}{speaker}: {content}"
            line_chars = len(line) + 1  # +1 for newline
            
            # Check if adding this line would exceed limit
            if total_chars + session_chars + line_chars > max_chars:
                # Truncate remaining content
                remaining_budget = max_chars - total_chars - session_chars - 50
                if remaining_budget > 100:
                    truncated_content = content[:remaining_budget] + "..."
                    lines.append(f"{prefix}{speaker}: {truncated_content}")
                lines.append("[... remaining content truncated ...]")
                break
            
            lines.append(line)
            session_chars += line_chars
        
        section = "\n".join(lines)
        sections.append(section)
        total_chars += len(section) + 2  # +2 for double newline
        
        if total_chars >= max_chars:
            sections.append("[... remaining sessions truncated ...]")
            break
    
    return "\n\n".join(sections)


def build_qa_prompt(
    persona: str,
    qa_items: List[Dict[str, Any]],
    max_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build QA prompt with length and question count limits."""
    instructions = [
        "You previously memorized this persona's conversations. "
        "Answer each question using only that stored memory. "
        "Return JSON with the schema: {\"answers\": [{\"question_id\": \"...\", \"answer\": \"...\"}]}."
    ]
    payload: List[Dict[str, Any]] = []
    total_chars = len(instructions[0])
    
    # Limit number of questions
    limited_qa = qa_items[:max_questions]
    
    for idx, qa in enumerate(limited_qa, start=1):
        qid = qa.get("id") or f"{persona}_q{idx}"
        question = qa.get("question", "")
        
        # Truncate very long questions
        if len(question) > 500:
            question = question[:500] + "..."
        
        question_text = f"{idx}. question_id={qid}\nQuestion: {question}"
        
        # Check if adding this question would exceed limit
        if total_chars + len(question_text) + 10 > max_chars:
            break
        
        instructions.append(question_text)
        payload.append({"id": qid, "question": question, "expected": qa.get("answer")})
        total_chars += len(question_text) + 2
    
    prompt = "\n\n".join(instructions)
    return prompt, payload


def parse_json_answers(raw: str) -> Dict[str, str]:
    """Parse JSON answers from agent response, handling markdown code blocks."""
    import re
    
    # Remove markdown code blocks if present
    clean_raw = raw.strip()
    
    # Handle ```json ... ``` or ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', clean_raw)
    if code_block_match:
        clean_raw = code_block_match.group(1).strip()
    
    # Try to extract JSON object
    try:
        data = json.loads(clean_raw)
    except json.JSONDecodeError:
        # Try to find JSON object in raw text
        json_match = re.search(r'\{[\s\S]*\}', clean_raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return {}
        else:
            return {}
    
    if isinstance(data, dict) and "answers" in data:
        answers = data["answers"]
    else:
        answers = data
    
    result: Dict[str, str] = {}
    if isinstance(answers, dict):
        for key, value in answers.items():
            result[str(key)] = str(value)
    elif isinstance(answers, list):
        for entry in answers:
            if isinstance(entry, dict):
                qid = entry.get("question_id") or entry.get("id")
                ans = entry.get("answer")
                if qid and ans:
                    result[str(qid)] = str(ans)
    return result


def evaluate_answers(raw: str, qa_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate parsed answers against expected answers."""
    parsed = parse_json_answers(raw)
    results: List[Dict[str, Any]] = []
    raw_norm = normalize_text(raw)
    
    for entry in qa_payload:
        qid = entry["id"]
        expected = entry.get("expected")
        exp_norm = normalize_text(expected) if expected else ""
        answer_text = parsed.get(qid)
        
        # If no parsed answer found for this question
        if answer_text is None:
            # Check if expected answer appears anywhere in raw response (fallback)
            if exp_norm and exp_norm in raw_norm:
                answer_text = f"[Found in response: {expected}]"
                correct = True
            else:
                answer_text = "[Answer not found in response]"
                correct = False
        else:
            ans_norm = normalize_text(answer_text)
            correct = bool(exp_norm) and exp_norm in ans_norm
        
        results.append({
            "question_id": qid,
            "question": entry["question"],
            "expected_answer": expected,
            "model_answer": answer_text,
            "answer_correct": correct,
        })
    
    return results


def run_persona(
    persona: str,
    conversations: List[List[Dict[str, Any]]],
    qa_items: List[Dict[str, Any]],
    *,
    userid_prefix: str,
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    delay_sec: float,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    max_qa_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = 10,
) -> Dict[str, Any]:
    userid = f"{userid_prefix}_{persona}"
    
    # Build transcript with length limit
    transcript = build_transcript(conversations, max_chars=max_transcript_chars)
    transcript_tokens = estimate_tokens(transcript)
    print(f"  Transcript: {len(transcript)} chars (~{transcript_tokens} tokens)")
    
    preload_message = (
        "Store the following conversation history for future reference. "
        "Acknowledge when complete.\n\n"
        + transcript
    )
    
    throttle(delay_sec)
    try:
        preload_response, _ = call_agent(
            [{"role": "user", "content": preload_message}],
            server_url=server_url,
            userid=userid,
            system_prompt=system_prompt,
            model=model,
            max_search_results=max_search_results,
            timeout=timeout,
        )
    except Exception as e:
        print(f"  [ERROR] Preload failed: {e}")
        preload_response = f"Error: {e}"

    try:
        reset_short_term_session(server_url, userid, timeout)
    except Exception as exc:
        print(f"  [WARN] Failed to reset session for {userid}: {exc}")

    if not qa_items:
        return {
            "persona": persona,
            "preload_response": preload_response,
            "qa_results": [],
        }

    # Build QA prompt with limits
    qa_prompt, qa_payload = build_qa_prompt(
        persona, qa_items,
        max_chars=max_qa_chars,
        max_questions=max_questions,
    )
    qa_tokens = estimate_tokens(qa_prompt)
    print(f"  QA prompt: {len(qa_prompt)} chars (~{qa_tokens} tokens), {len(qa_payload)} questions")
    
    throttle(delay_sec)
    try:
        qa_response, _ = call_agent(
            [{"role": "user", "content": qa_prompt}],
            server_url=server_url,
            userid=userid,
            system_prompt=system_prompt,
            model=model,
            max_search_results=max_search_results,
            timeout=timeout,
        )
    except Exception as e:
        print(f"  [ERROR] QA failed: {e}")
        qa_response = f"Error: {e}"

    qa_results = evaluate_answers(qa_response, qa_payload)
    return {
        "persona": persona,
        "preload_response": preload_response,
        "qa_response": qa_response,
        "qa_results": qa_results,
    }


def build_persona_map(data: Dict[str, Any]) -> Tuple[Dict[str, List[List[Dict[str, Any]]]], Dict[str, List[Dict[str, Any]]]]:
    personas_conversations: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    personas_qa: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dia_persona: Dict[str, str] = {}

    for tc in data.get("test_cases", []):
        persona = (tc.get("id") or "persona_unknown").split("_session")[0]
        personas_conversations[persona].append(tc.get("conversation") or [])
        for turn in tc.get("conversation") or []:
            dia_id = turn.get("dia_id")
            if dia_id:
                dia_persona[dia_id] = persona

    for qa in data.get("qa", []):
        persona = None
        for ev in qa.get("evidence") or []:
            persona = dia_persona.get(ev)
            if persona:
                break
        if persona:
            personas_qa[persona].append(qa)

    return personas_conversations, personas_qa


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LoCoMo batch runner (two calls per persona).")
    parser.add_argument(
        "--benchmark-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark"),
    )
    parser.add_argument(
        "--locomo-file",
        default=None,
        help="Path to locomo1_converted.json (default: <benchmark-dir>/locomo1_converted.json).",
    )
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--system-prompt", default="You are a helpful AI assistant.")
    parser.add_argument("--max-search-results", type=int, default=10)
    parser.add_argument("--userid-prefix", default="batch")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--delay-sec", type=float, default=3.0, help="Delay between successive /invoke calls (default: 3s to respect rate limits).")
    parser.add_argument("--limit-personas", type=int, default=None, help="Optional cap on number of personas to run.")
    
    # Token limit arguments
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=DEFAULT_MAX_TRANSCRIPT_CHARS,
        help=f"Max characters for conversation transcript (default: {DEFAULT_MAX_TRANSCRIPT_CHARS}, ~{DEFAULT_MAX_TRANSCRIPT_CHARS//4} tokens)",
    )
    parser.add_argument(
        "--max-qa-chars",
        type=int,
        default=DEFAULT_MAX_QA_CHARS,
        help=f"Max characters for QA prompt (default: {DEFAULT_MAX_QA_CHARS}, ~{DEFAULT_MAX_QA_CHARS//4} tokens)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Max number of questions per persona (default: 10)",
    )

    args = parser.parse_args(argv)
    locomo_file = args.locomo_file or os.path.join(args.benchmark_dir, "locomo1_converted.json")
    data = load_json(locomo_file)

    personas_convos, personas_qa = build_persona_map(data)
    personas = list(personas_convos.keys())
    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]

    print(f"Running with token limits:")
    print(f"  Max transcript: {args.max_transcript_chars} chars (~{args.max_transcript_chars//4} tokens)")
    print(f"  Max QA prompt: {args.max_qa_chars} chars (~{args.max_qa_chars//4} tokens)")
    print(f"  Max questions: {args.max_questions}")
    print(f"  Delay between calls: {args.delay_sec}s")

    all_results: List[Dict[str, Any]] = []
    for persona in personas:
        print(f"\n=== Persona {persona} ===")
        qa_items = personas_qa.get(persona, [])
        result = run_persona(
            persona,
            personas_convos[persona],
            qa_items,
            userid_prefix=args.userid_prefix,
            server_url=args.server_url,
            system_prompt=args.system_prompt,
            model=args.model,
            max_search_results=args.max_search_results,
            timeout=args.timeout,
            delay_sec=args.delay_sec,
            max_transcript_chars=args.max_transcript_chars,
            max_qa_chars=args.max_qa_chars,
            max_questions=args.max_questions,
        )
        correct = sum(1 for r in result["qa_results"] if r["answer_correct"])
        total = len(result["qa_results"])
        print(f"  QA accuracy: {correct}/{total}")
        all_results.append(result)

    summary = {
        "personas_run": len(all_results),
        "qa_accuracy_total": sum(sum(1 for r in res["qa_results"] if r["answer_correct"]) for res in all_results),
        "qa_questions_total": sum(len(res["qa_results"]) for res in all_results),
        "details": all_results,
    }

    output_path = os.path.join(os.getcwd(), "locomo_batch_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()

