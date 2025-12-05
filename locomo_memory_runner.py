#!/usr/bin/env python
"""
LoCoMo Memory Benchmark Runner

This runner tests cross-session memory by:
1) Storing conversation history directly in the memory system (ChromaDB)
2) Resetting short-term session while preserving long-term memory
3) Asking QA questions that require recalling from long-term memory

This properly tests the memory system's ability to persist and retrieve
information across sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import requests
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from benchmark_runner import (
    load_json,
    normalize_text,
    call_agent,
    reset_short_term_session,
)

# Token limits (Claude: 30K input tokens/min for Sonnet/Opus)
# Budget ~1500 tokens per request to stay within limits
DEFAULT_MAX_CONTENT_CHARS = 2000  # ~500 tokens per memory item
DEFAULT_MAX_QA_CHARS = 4000  # ~1000 tokens


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def throttle(delay_sec: float) -> None:
    if delay_sec > 0:
        time.sleep(delay_sec)


def store_memory(
    server_url: str,
    userid: str,
    key: str,
    content: str,
    timeout: float,
) -> bool:
    """Store a memory via the /memory API."""
    url = f"{server_url.rstrip('/')}/memory/{userid}/store"
    try:
        resp = requests.post(
            url,
            json={"key": key, "content": content},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[WARN] Failed to store memory: {e}")
        return False


def clear_user_memories(
    server_url: str,
    userid: str,
    timeout: float,
) -> bool:
    """Clear all memories for a user via DELETE /memory/{userid}."""
    url = f"{server_url.rstrip('/')}/memory/{userid}"
    try:
        resp = requests.delete(url, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        print(f"[WARN] Failed to clear memories: {e}")
        return False


def list_memories(
    server_url: str,
    userid: str,
    timeout: float,
) -> List[Dict[str, Any]]:
    """List all memories for a user."""
    url = f"{server_url.rstrip('/')}/memory/{userid}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("memories", [])
    except Exception as e:
        print(f"[WARN] Failed to list memories: {e}")
    return []


def build_conversation_memories(
    conversations: List[List[Dict[str, Any]]],
    persona: str,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> List[Dict[str, str]]:
    """
    Convert conversations into memory items.
    
    Each session becomes a separate memory with context about the speakers
    and what was discussed. Content is truncated to stay within token limits.
    """
    memories = []
    
    for session_idx, convo in enumerate(conversations, start=1):
        if not convo:
            continue
            
        # Build a summary of this session
        lines = []
        total_chars = 0
        header = f"Conversation session {session_idx}:\n"
        total_chars += len(header)
        
        for turn in convo:
            speaker = turn.get("speaker") or turn.get("role") or "unknown"
            content = turn.get("content", "")
            
            if content:
                line = f"{speaker}: {content}"
                line_chars = len(line) + 1
                
                # Check if adding this line exceeds limit
                if total_chars + line_chars > max_content_chars:
                    # Truncate remaining
                    remaining = max_content_chars - total_chars - 50
                    if remaining > 50:
                        truncated = content[:remaining] + "..."
                        lines.append(f"{speaker}: {truncated}")
                    lines.append("[... truncated ...]")
                    break
                
                lines.append(line)
                total_chars += line_chars
        
        if lines:
            session_content = header + "\n".join(lines)
            memories.append({
                "key": f"{persona}_session_{session_idx}",
                "content": session_content,
            })
    
    return memories


def store_persona_memories(
    persona: str,
    conversations: List[List[Dict[str, Any]]],
    server_url: str,
    userid: str,
    timeout: float,
    delay_sec: float,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> int:
    """
    Store all conversation memories for a persona.
    Returns the number of memories stored.
    """
    # First clear existing memories for this user
    clear_user_memories(server_url, userid, timeout)
    throttle(delay_sec)
    
    # Build memory items from conversations with size limit
    memories = build_conversation_memories(conversations, persona, max_content_chars)
    
    stored_count = 0
    total_chars = 0
    for mem in memories:
        content_chars = len(mem["content"])
        total_chars += content_chars
        
        success = store_memory(
            server_url=server_url,
            userid=userid,
            key=mem["key"],
            content=mem["content"],
            timeout=timeout,
        )
        if success:
            stored_count += 1
            print(f"    Stored {mem['key']}: {content_chars} chars (~{content_chars//4} tokens)")
        throttle(delay_sec * 0.5)  # Shorter delay between memory stores
    
    print(f"    Total stored: {total_chars} chars (~{total_chars//4} tokens)")
    return stored_count


def build_qa_prompt_with_memory(
    persona: str,
    qa_items: List[Dict[str, Any]],
    max_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build QA prompt that instructs the agent to use memory search.
    Limited by character count and question count.
    """
    header = (
        "You have access to stored memories about past conversations. "
        "Use your memory search tool to recall relevant information and answer each question. "
        "Return JSON with the schema: {\"answers\": [{\"question_id\": \"...\", \"answer\": \"...\"}]}.\n\n"
        "Questions to answer:"
    )
    
    instructions = [header]
    payload: List[Dict[str, Any]] = []
    total_chars = len(header)
    
    # Limit questions
    limited_qa = qa_items[:max_questions]
    
    for idx, qa in enumerate(limited_qa, start=1):
        qid = qa.get("id") or f"{persona}_q{idx}"
        question = qa.get("question", "")
        
        # Truncate long questions
        if len(question) > 300:
            question = question[:300] + "..."
        
        question_text = f"{idx}. question_id={qid}\n   Question: {question}"
        
        if total_chars + len(question_text) + 10 > max_chars:
            break
        
        instructions.append(question_text)
        payload.append({"id": qid, "question": question, "expected": qa.get("answer")})
        total_chars += len(question_text) + 2
    
    prompt = "\n".join(instructions)
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


def run_persona_with_memory(
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
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    max_qa_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = 10,
) -> Dict[str, Any]:
    """
    Run benchmark for a single persona using the memory system.
    
    Steps:
    1. Store conversation memories directly via API
    2. Reset short-term session (preserve long-term memory)
    3. Ask QA questions - agent should search memory
    """
    userid = f"{userid_prefix}_{persona}"
    
    # Step 1: Store memories
    print(f"  Storing memories for {persona}...")
    stored_count = store_persona_memories(
        persona=persona,
        conversations=conversations,
        server_url=server_url,
        userid=userid,
        timeout=timeout,
        delay_sec=delay_sec,
        max_content_chars=max_content_chars,
    )
    print(f"  Stored {stored_count} memory items")
    
    # Verify memories were stored
    memories = list_memories(server_url, userid, timeout)
    print(f"  Verified {len(memories)} memories in storage")
    
    # Step 2: Reset short-term session but KEEP memory
    throttle(delay_sec)
    try:
        reset_short_term_session(server_url, userid, timeout, preserve_memory=True)
        print(f"  Reset short-term session (memory preserved)")
    except Exception as exc:
        print(f"  [WARN] Failed to reset session: {exc}")
    
    if not qa_items:
        return {
            "persona": persona,
            "memories_stored": stored_count,
            "qa_results": [],
        }
    
    # Step 3: Ask QA questions
    qa_prompt, qa_payload = build_qa_prompt_with_memory(
        persona, qa_items,
        max_chars=max_qa_chars,
        max_questions=max_questions,
    )
    qa_tokens = estimate_tokens(qa_prompt)
    print(f"  Asking {len(qa_payload)} questions ({len(qa_prompt)} chars, ~{qa_tokens} tokens)...")
    
    throttle(delay_sec)
    qa_response, _ = call_agent(
        [{"role": "user", "content": qa_prompt}],
        server_url=server_url,
        userid=userid,
        system_prompt=system_prompt,
        model=model,
        max_search_results=max_search_results,
        timeout=timeout,
    )
    
    qa_results = evaluate_answers(qa_response, qa_payload)
    
    return {
        "persona": persona,
        "memories_stored": stored_count,
        "qa_response": qa_response,
        "qa_results": qa_results,
    }


def build_persona_map(data: Dict[str, Any]) -> Tuple[Dict[str, List[List[Dict[str, Any]]]], Dict[str, List[Dict[str, Any]]]]:
    """Build mapping from persona to conversations and QA items."""
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
    parser = argparse.ArgumentParser(
        description="LoCoMo Memory Benchmark Runner - Tests cross-session memory"
    )
    parser.add_argument(
        "--benchmark-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark"),
    )
    parser.add_argument(
        "--locomo-file",
        default=None,
        help="Path to locomo1_converted.json",
    )
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant with access to long-term memory. Use your memory tools to recall past conversations when answering questions.",
    )
    parser.add_argument("--max-search-results", type=int, default=10)
    parser.add_argument("--userid-prefix", default="memory_test")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--delay-sec", type=float, default=3.0, help="Delay between API calls (default: 3s to respect rate limits)")
    parser.add_argument("--limit-personas", type=int, default=None)
    parser.add_argument(
        "--output",
        default="locomo_memory_results.json",
        help="Output file for results",
    )
    
    # Token limit arguments
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=DEFAULT_MAX_CONTENT_CHARS,
        help=f"Max chars per memory item (default: {DEFAULT_MAX_CONTENT_CHARS}, ~{DEFAULT_MAX_CONTENT_CHARS//4} tokens)",
    )
    parser.add_argument(
        "--max-qa-chars",
        type=int,
        default=DEFAULT_MAX_QA_CHARS,
        help=f"Max chars for QA prompt (default: {DEFAULT_MAX_QA_CHARS}, ~{DEFAULT_MAX_QA_CHARS//4} tokens)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=10,
        help="Max questions per persona (default: 10)",
    )

    args = parser.parse_args(argv)
    locomo_file = args.locomo_file or os.path.join(args.benchmark_dir, "locomo1_converted.json")
    
    print(f"Loading benchmark from: {locomo_file}")
    data = load_json(locomo_file)

    personas_convos, personas_qa = build_persona_map(data)
    personas = list(personas_convos.keys())
    
    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]
    
    print(f"\nRunning memory benchmark for {len(personas)} personas")
    print(f"Server: {args.server_url}")
    print(f"Token limits:")
    print(f"  Max content per memory: {args.max_content_chars} chars (~{args.max_content_chars//4} tokens)")
    print(f"  Max QA prompt: {args.max_qa_chars} chars (~{args.max_qa_chars//4} tokens)")
    print(f"  Max questions: {args.max_questions}")
    print(f"  Delay between calls: {args.delay_sec}s")
    print("=" * 60)

    all_results: List[Dict[str, Any]] = []
    total_correct = 0
    total_questions = 0
    
    for persona in personas:
        print(f"\n=== Persona: {persona} ===")
        qa_items = personas_qa.get(persona, [])
        
        result = run_persona_with_memory(
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
            max_content_chars=args.max_content_chars,
            max_qa_chars=args.max_qa_chars,
            max_questions=args.max_questions,
        )
        
        correct = sum(1 for r in result["qa_results"] if r["answer_correct"])
        total = len(result["qa_results"])
        total_correct += correct
        total_questions += total
        
        print(f"  QA Accuracy: {correct}/{total}")
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Personas tested: {len(all_results)}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {total_correct}")
    if total_questions > 0:
        print(f"Overall accuracy: {total_correct/total_questions*100:.1f}%")

    summary = {
        "personas_run": len(all_results),
        "qa_accuracy_total": total_correct,
        "qa_questions_total": total_questions,
        "accuracy_percent": (total_correct / total_questions * 100) if total_questions > 0 else 0,
        "details": all_results,
    }

    output_path = os.path.join(os.getcwd(), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()

