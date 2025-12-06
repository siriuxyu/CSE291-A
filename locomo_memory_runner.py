#!/usr/bin/env python
"""
LoCoMo Memory Benchmark Runner (v2 - Full Coverage with Resume Support)

This runner tests cross-session memory by:
1) Storing ALL conversation history in chunks (no truncation)
2) Resetting short-term session while preserving long-term memory
3) Asking QA questions in batches that require recalling from long-term memory

Two-phase approach:
- Phase 1: Store all conversations for all personas (chunked storage)
- Phase 2: Ask QA questions in batches (respecting token limits)

Resume support:
- Automatically saves checkpoint after each persona
- Use --resume to continue from last checkpoint after interruption
- Use --force-restart to ignore checkpoint and start fresh

This properly tests the memory system's ability to persist and retrieve
information across sessions without losing any conversation content.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
import requests
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Set

from benchmark_runner import (
    load_json,
    normalize_text,
    call_agent,
    reset_short_term_session,
)

# Chunking settings
DEFAULT_TURNS_PER_CHUNK = 3  # Number of conversation turns per memory chunk
DEFAULT_MAX_CHUNK_CHARS = 1500  # Max chars per chunk (~375 tokens)

# QA batch settings (for Phase 2)
DEFAULT_MAX_QA_CHARS = 2000  # ~500 tokens per QA batch
DEFAULT_MAX_QUESTIONS_PER_BATCH = 3  # Questions per API call

# Rate limiting
DEFAULT_DELAY_SEC = 30.0  # Increased to avoid rate limiting (agent may call search_memory multiple times)
DEFAULT_MEMORY_STORE_DELAY = 1.0  # Shorter delay for memory storage (no LLM call)
DEFAULT_MAX_RETRIES = 2  # Retries for failed batches

# Checkpoint settings
DEFAULT_CHECKPOINT_FILE = "locomo_checkpoint.json"


# =============================================================================
# Checkpoint Management
# =============================================================================

def load_checkpoint(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file if it exists."""
    if not os.path.exists(checkpoint_file):
        return None
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"[CHECKPOINT] Loaded checkpoint from {checkpoint_file}")
        print(f"  Phase 1 completed: {len(checkpoint.get('phase1_completed', []))} personas")
        print(f"  Phase 2 completed: {len(checkpoint.get('phase2_completed', []))} personas")
        return checkpoint
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {e}")
        return None


def save_checkpoint(
    checkpoint_file: str,
    phase1_completed: List[str],
    phase2_completed: List[str],
    storage_stats: Dict[str, Any],
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    current_persona_progress: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save current progress to checkpoint file.
    
    Args:
        current_persona_progress: Partial progress for persona currently being processed
            {"persona": str, "qa_results": List[...]}
    """
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "phase1_completed": phase1_completed,
        "phase2_completed": phase2_completed,
        "storage_stats": storage_stats,
        "results": results,
        "config": config,
    }
    if current_persona_progress:
        checkpoint["current_persona_progress"] = current_persona_progress
    
    try:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")


def delete_checkpoint(checkpoint_file: str, backup: bool = True) -> None:
    """
    Delete checkpoint file after successful completion.
    
    Args:
        checkpoint_file: Path to checkpoint file
        backup: If True, create a backup copy before deleting
    """
    if os.path.exists(checkpoint_file):
        try:
            if backup:
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = checkpoint_file.replace(".json", f"_final_{timestamp}.json")
                shutil.copy2(checkpoint_file, backup_file)
                print(f"[CHECKPOINT] Backed up to: {backup_file}")
            
            os.remove(checkpoint_file)
            print(f"[CHECKPOINT] Deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"[WARN] Failed to delete checkpoint: {e}")


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def throttle(delay_sec: float) -> None:
    if delay_sec > 0:
        time.sleep(delay_sec)


# =============================================================================
# Memory Storage Functions
# =============================================================================

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


def build_conversation_memories_chunked(
    conversations: List[List[Dict[str, Any]]],
    persona: str,
    turns_per_chunk: int = DEFAULT_TURNS_PER_CHUNK,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[Dict[str, str]]:
    """
    Convert conversations into chunked memory items.
    
    Each chunk contains N turns (or fewer if hitting char limit).
    No truncation - all content is preserved across multiple chunks.
    
    Args:
        conversations: List of conversation sessions
        persona: Persona identifier for key naming
        turns_per_chunk: Target number of turns per memory chunk
        max_chunk_chars: Max characters per chunk (soft limit, will break at turn boundary)
    
    Returns:
        List of memory items with 'key' and 'content'
    """
    memories = []
    global_chunk_idx = 0
    
    for session_idx, convo in enumerate(conversations, start=1):
        if not convo:
            continue
        
        current_chunk_lines = []
        current_chunk_chars = 0
        turns_in_chunk = 0
        chunk_in_session = 0
        
        session_header = f"[Session {session_idx}]"
        
        for turn_idx, turn in enumerate(convo):
            speaker = turn.get("speaker") or turn.get("role") or "unknown"
            content = turn.get("content", "")
            dia_id = turn.get("dia_id", "")
            
            if not content:
                continue
            
            # Format line with optional dia_id
            if dia_id:
                line = f"[{dia_id}] {speaker}: {content}"
            else:
                line = f"{speaker}: {content}"
            
            line_chars = len(line) + 1  # +1 for newline
            
            # Check if we need to start a new chunk
            # Conditions: exceeded turn count OR exceeded char limit
            should_break = (
                turns_in_chunk >= turns_per_chunk or
                (current_chunk_chars + line_chars > max_chunk_chars and turns_in_chunk > 0)
            )
            
            if should_break and current_chunk_lines:
                # Save current chunk
                chunk_content = session_header + "\n" + "\n".join(current_chunk_lines)
                global_chunk_idx += 1
                memories.append({
                    "key": f"{persona}_s{session_idx}_c{chunk_in_session}",
                    "content": chunk_content,
                })
                
                # Reset for new chunk
                current_chunk_lines = []
                current_chunk_chars = 0
                turns_in_chunk = 0
                chunk_in_session += 1
            
            # Add turn to current chunk
            current_chunk_lines.append(line)
            current_chunk_chars += line_chars
            turns_in_chunk += 1
        
        # Don't forget the last chunk of this session
        if current_chunk_lines:
            chunk_content = session_header + "\n" + "\n".join(current_chunk_lines)
            global_chunk_idx += 1
            memories.append({
                "key": f"{persona}_s{session_idx}_c{chunk_in_session}",
                "content": chunk_content,
            })
    
    return memories


def store_all_persona_memories(
    persona: str,
    conversations: List[List[Dict[str, Any]]],
    server_url: str,
    userid: str,
    timeout: float,
    memory_store_delay: float,
    turns_per_chunk: int,
    max_chunk_chars: int,
    skip_clear: bool = False,
) -> Tuple[int, int]:
    """
    Store all conversation memories for a persona (no truncation).
    
    Args:
        skip_clear: If True, don't clear existing memories (for resume)
    
    Returns:
        (stored_count, total_chars)
    """
    # Clear existing memories (unless resuming with existing memories)
    if not skip_clear:
        clear_user_memories(server_url, userid, timeout)
        throttle(memory_store_delay)
    
    # Build chunked memories
    memories = build_conversation_memories_chunked(
        conversations, persona, turns_per_chunk, max_chunk_chars
    )
    
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
        
        throttle(memory_store_delay)
    
    return stored_count, total_chars


# =============================================================================
# QA Functions
# =============================================================================

def build_qa_batch_prompt(
    qa_items: List[Dict[str, Any]],
    start_idx: int = 1,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build QA prompt for a batch of questions.
    
    Returns:
        (prompt_text, payload_with_expected_answers)
    """
    header = (
        "You have access to stored memories about past conversations. "
        "Use your memory search tool to recall relevant information and answer each question. "
        "Return JSON with the schema: {\"answers\": [{\"question_id\": \"...\", \"answer\": \"...\"}]}.\n\n"
        "Questions to answer:"
    )
    
    instructions = [header]
    payload: List[Dict[str, Any]] = []
    
    for idx, qa in enumerate(qa_items, start=start_idx):
        qid = qa.get("id") or f"q{idx}"
        question = qa.get("question", "")
        
        question_text = f"{idx}. question_id={qid}\n   Question: {question}"
        instructions.append(question_text)
        payload.append({
            "id": qid,
            "question": question,
            "expected": qa.get("answer"),
            "category": qa.get("category"),
        })
    
    prompt = "\n".join(instructions)
    return prompt, payload


def chunk_qa_items(
    qa_items: List[Dict[str, Any]],
    max_chars: int = DEFAULT_MAX_QA_CHARS,
    max_questions: int = DEFAULT_MAX_QUESTIONS_PER_BATCH,
) -> List[List[Dict[str, Any]]]:
    """
    Split QA items into batches that fit within token limits.
    
    Args:
        qa_items: All QA items for a persona
        max_chars: Max characters per batch prompt
        max_questions: Max questions per batch
    
    Returns:
        List of QA item batches
    """
    batches = []
    current_batch = []
    current_chars = 300  # Header overhead
    
    for qa in qa_items:
        question = qa.get("question", "")
        question_chars = len(question) + 50  # Overhead for formatting
        
        # Check if we need to start a new batch
        should_break = (
            len(current_batch) >= max_questions or
            (current_chars + question_chars > max_chars and current_batch)
        )
        
        if should_break:
            batches.append(current_batch)
            current_batch = []
            current_chars = 300
        
        current_batch.append(qa)
        current_chars += question_chars
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def parse_json_answers(raw: str) -> Dict[str, str]:
    """Parse JSON answers from agent response, handling markdown code blocks."""
    import re
    
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


def evaluate_batch_answers(
    raw: str,
    qa_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate parsed answers against expected answers."""
    parsed = parse_json_answers(raw)
    results: List[Dict[str, Any]] = []
    raw_norm = normalize_text(raw)
    
    for entry in qa_payload:
        qid = entry["id"]
        expected = entry.get("expected")
        exp_norm = normalize_text(expected) if expected else ""
        answer_text = parsed.get(qid)
        
        if answer_text is None:
            # Fallback: check if expected appears anywhere in response
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
            "category": entry.get("category"),
        })
    
    return results


def ask_qa_batches(
    userid: str,
    qa_items: List[Dict[str, Any]],
    server_url: str,
    system_prompt: Optional[str],
    model: Optional[str],
    max_search_results: Optional[int],
    timeout: float,
    delay_sec: float,
    max_qa_chars: int,
    max_questions_per_batch: int,
    *,
    existing_results: Optional[List[Dict[str, Any]]] = None,
    on_batch_complete: Optional[callable] = None,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Ask QA questions in batches, respecting token limits.
    
    Args:
        existing_results: Results from previous run (for resume)
        on_batch_complete: Callback called after each batch with (batch_idx, all_results)
        max_retries: Number of retries for failed batches
    
    Returns:
        List of all QA results
    """
    if not qa_items:
        return []
    
    batches = chunk_qa_items(
        qa_items,
        max_chars=max_qa_chars,
        max_questions=max_questions_per_batch,
    )
    
    # Resume from existing results
    all_results = list(existing_results) if existing_results else []
    completed_questions = {r["question_id"] for r in all_results}
    
    question_idx = 1
    
    for batch_idx, batch in enumerate(batches, start=1):
        # Check if this batch was already completed
        batch_question_ids = {qa.get("id") or f"q{question_idx + i}" for i, qa in enumerate(batch)}
        if batch_question_ids.issubset(completed_questions):
            print(f"    Batch {batch_idx}/{len(batches)}: Already completed (skipping)")
            question_idx += len(batch)
            continue
        
        qa_prompt, qa_payload = build_qa_batch_prompt(batch, start_idx=question_idx)
        prompt_chars = len(qa_prompt)
        prompt_tokens = estimate_tokens(qa_prompt)
        
        print(f"    Batch {batch_idx}/{len(batches)}: {len(batch)} questions "
              f"({prompt_chars} chars, ~{prompt_tokens} tokens)")
        
        # Retry logic
        batch_results = None
        last_error = None
        
        # Generate unique thread_id for this batch to avoid message history pollution
        # userid is kept the same for memory search (search_memory tool uses userid)
        batch_thread_id = f"{userid}_batch_{batch_idx}_{uuid.uuid4().hex[:8]}"
        
        for retry in range(max_retries + 1):
            if retry > 0:
                retry_delay = delay_sec * (retry + 1)  # Exponential backoff
                print(f"      Retry {retry}/{max_retries} after {retry_delay}s...")
                throttle(retry_delay)
                # Generate new thread_id for retry to ensure clean state
                batch_thread_id = f"{userid}_batch_{batch_idx}_retry{retry}_{uuid.uuid4().hex[:8]}"
            else:
                throttle(delay_sec)
            
            try:
                qa_response, _ = call_agent(
                    [{"role": "user", "content": qa_prompt}],
                    server_url=server_url,
                    userid=userid,  # For memory search
                    system_prompt=system_prompt,
                    model=model,
                    max_search_results=max_search_results,
                    timeout=timeout,
                    thread_id=batch_thread_id,  # Unique per batch to avoid message pollution
                )
                
                batch_results = evaluate_batch_answers(qa_response, qa_payload)
                break  # Success
                
            except Exception as e:
                last_error = e
                print(f"      [ERROR] Attempt {retry + 1} failed: {e}")
        
        if batch_results is not None:
            all_results.extend(batch_results)
            correct = sum(1 for r in batch_results if r["answer_correct"])
            print(f"      -> Batch accuracy: {correct}/{len(batch_results)}")
        else:
            # All retries failed
            print(f"    [ERROR] Batch {batch_idx} failed after {max_retries + 1} attempts: {last_error}")
            for qa in qa_payload:
                all_results.append({
                    "question_id": qa["id"],
                    "question": qa["question"],
                    "expected_answer": qa.get("expected"),
                    "model_answer": f"[Error after {max_retries + 1} attempts: {last_error}]",
                    "answer_correct": False,
                    "category": qa.get("category"),
                })
        
        # Call the callback for checkpoint saving
        if on_batch_complete:
            on_batch_complete(batch_idx, all_results)
        
        question_idx += len(batch)
    
    return all_results


# =============================================================================
# Data Processing
# =============================================================================

def build_persona_map(
    data: Dict[str, Any]
) -> Tuple[Dict[str, List[List[Dict[str, Any]]]], Dict[str, List[Dict[str, Any]]]]:
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


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo Memory Benchmark Runner v2 - Full conversation coverage with resume support"
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
        default="You are a helpful AI assistant with access to long-term memory. "
                "Use your memory tools to recall past conversations when answering questions.",
    )
    parser.add_argument("--max-search-results", type=int, default=10)
    parser.add_argument("--userid-prefix", default="memory_test")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=f"Delay between LLM API calls (default: {DEFAULT_DELAY_SEC}s)",
    )
    parser.add_argument(
        "--memory-store-delay",
        type=float,
        default=DEFAULT_MEMORY_STORE_DELAY,
        help=f"Delay between memory store calls (default: {DEFAULT_MEMORY_STORE_DELAY}s)",
    )
    parser.add_argument("--limit-personas", type=int, default=None)
    parser.add_argument(
        "--output",
        default="locomo_memory_results.json",
        help="Output file for results",
    )
    
    # Chunking arguments
    parser.add_argument(
        "--turns-per-chunk",
        type=int,
        default=DEFAULT_TURNS_PER_CHUNK,
        help=f"Number of conversation turns per memory chunk (default: {DEFAULT_TURNS_PER_CHUNK})",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARS,
        help=f"Max chars per memory chunk (default: {DEFAULT_MAX_CHUNK_CHARS})",
    )
    
    # QA batch arguments
    parser.add_argument(
        "--max-qa-chars",
        type=int,
        default=DEFAULT_MAX_QA_CHARS,
        help=f"Max chars per QA batch prompt (default: {DEFAULT_MAX_QA_CHARS})",
    )
    parser.add_argument(
        "--max-questions-per-batch",
        type=int,
        default=DEFAULT_MAX_QUESTIONS_PER_BATCH,
        help=f"Max questions per QA batch (default: {DEFAULT_MAX_QUESTIONS_PER_BATCH})",
    )
    
    # Resume arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help=f"Checkpoint file path (default: {DEFAULT_CHECKPOINT_FILE})",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for failed QA batches (default: {DEFAULT_MAX_RETRIES})",
    )

    args = parser.parse_args(argv)
    locomo_file = args.locomo_file or os.path.join(args.benchmark_dir, "locomo1_converted.json")
    
    # Configuration for checkpoint
    config = {
        "locomo_file": locomo_file,
        "server_url": args.server_url,
        "userid_prefix": args.userid_prefix,
        "turns_per_chunk": args.turns_per_chunk,
        "max_chunk_chars": args.max_chunk_chars,
        "max_qa_chars": args.max_qa_chars,
        "max_questions_per_batch": args.max_questions_per_batch,
    }
    
    # Load checkpoint if resuming
    checkpoint = None
    phase1_completed: List[str] = []
    phase2_completed: List[str] = []
    storage_stats: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []
    current_persona_progress: Optional[Dict[str, Any]] = None
    
    if args.force_restart:
        print("[CHECKPOINT] Force restart - ignoring any existing checkpoint")
        delete_checkpoint(args.checkpoint_file)
    elif args.resume:
        checkpoint = load_checkpoint(args.checkpoint_file)
        if checkpoint:
            phase1_completed = checkpoint.get("phase1_completed", [])
            phase2_completed = checkpoint.get("phase2_completed", [])
            storage_stats = checkpoint.get("storage_stats", {})
            all_results = checkpoint.get("results", [])
            current_persona_progress = checkpoint.get("current_persona_progress")
            print(f"[CHECKPOINT] Resuming with {len(phase1_completed)} phase1, {len(phase2_completed)} phase2 completed")
            if current_persona_progress:
                persona = current_persona_progress.get("persona")
                partial_results = current_persona_progress.get("qa_results", [])
                print(f"  Partial progress for {persona}: {len(partial_results)} questions answered")
        else:
            print("[CHECKPOINT] No checkpoint found, starting fresh")
    elif os.path.exists(args.checkpoint_file):
        print(f"[CHECKPOINT] Found existing checkpoint: {args.checkpoint_file}")
        print("  Use --resume to continue from checkpoint")
        print("  Use --force-restart to start fresh")
        print("  Proceeding with fresh start...")
    
    print(f"Loading benchmark from: {locomo_file}")
    data = load_json(locomo_file)

    personas_convos, personas_qa = build_persona_map(data)
    personas = list(personas_convos.keys())
    
    if args.limit_personas is not None:
        personas = personas[: args.limit_personas]
    
    phase1_completed_set: Set[str] = set(phase1_completed)
    phase2_completed_set: Set[str] = set(phase2_completed)
    
    print(f"\n{'='*60}")
    print("LoCoMo Memory Benchmark Runner v2 (Full Coverage + Resume)")
    print(f"{'='*60}")
    print(f"Personas to test: {len(personas)}")
    print(f"Server: {args.server_url}")
    print(f"\nChunking settings:")
    print(f"  Turns per chunk: {args.turns_per_chunk}")
    print(f"  Max chunk chars: {args.max_chunk_chars} (~{args.max_chunk_chars//4} tokens)")
    print(f"\nQA batch settings:")
    print(f"  Max QA chars per batch: {args.max_qa_chars} (~{args.max_qa_chars//4} tokens)")
    print(f"  Max questions per batch: {args.max_questions_per_batch}")
    print(f"\nRate limiting:")
    print(f"  LLM call delay: {args.delay_sec}s")
    print(f"  Memory store delay: {args.memory_store_delay}s")
    print(f"\nResume status:")
    print(f"  Checkpoint file: {args.checkpoint_file}")
    print(f"  Phase 1 already completed: {len(phase1_completed_set)} personas")
    print(f"  Phase 2 already completed: {len(phase2_completed_set)} personas")
    print(f"{'='*60}")
    
    # =========================================================================
    # PHASE 1: Store all conversations for all personas
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Storing all conversations")
    print(f"{'='*60}")
    
    total_chunks_stored = sum(s.get("chunks_stored", 0) for s in storage_stats.values())
    total_chars_stored = sum(s.get("chars_stored", 0) for s in storage_stats.values())
    
    for persona in personas:
        # Skip if already completed in phase 1
        if persona in phase1_completed_set:
            print(f"\n[{persona}] Already stored (skipping)")
            continue
        
        userid = f"{args.userid_prefix}_{persona}"
        conversations = personas_convos[persona]
        
        # Count total turns
        total_turns = sum(len(c) for c in conversations)
        print(f"\n[{persona}] {len(conversations)} sessions, {total_turns} turns")
        
        try:
            stored_count, total_chars = store_all_persona_memories(
                persona=persona,
                conversations=conversations,
                server_url=args.server_url,
                userid=userid,
                timeout=args.timeout,
                memory_store_delay=args.memory_store_delay,
                turns_per_chunk=args.turns_per_chunk,
                max_chunk_chars=args.max_chunk_chars,
            )
            
            # Verify storage
            memories = list_memories(args.server_url, userid, args.timeout)
            
            print(f"  -> Stored {stored_count} chunks ({total_chars} chars, ~{total_chars//4} tokens)")
            print(f"  -> Verified {len(memories)} memories in storage")
            
            storage_stats[persona] = {
                "sessions": len(conversations),
                "total_turns": total_turns,
                "chunks_stored": stored_count,
                "chars_stored": total_chars,
                "verified_memories": len(memories),
            }
            
            total_chunks_stored += stored_count
            total_chars_stored += total_chars
            
            # Mark as completed and save checkpoint
            phase1_completed.append(persona)
            phase1_completed_set.add(persona)
            
            save_checkpoint(
                args.checkpoint_file,
                phase1_completed,
                phase2_completed,
                storage_stats,
                all_results,
                config,
            )
            print(f"  [CHECKPOINT] Saved (Phase 1: {len(phase1_completed)}/{len(personas)})")
            
        except Exception as e:
            print(f"  [ERROR] Failed to store memories for {persona}: {e}")
            print(f"  [INFO] Run with --resume to retry this persona")
            # Don't mark as completed, will retry on resume
            continue
    
    print(f"\n{'='*60}")
    print(f"Phase 1 Complete: {total_chunks_stored} total chunks, "
          f"{total_chars_stored} chars (~{total_chars_stored//4} tokens)")
    print(f"{'='*60}")
    
    # =========================================================================
    # PHASE 2: Ask QA questions in batches
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Asking QA questions")
    print(f"{'='*60}")
    
    total_correct = sum(
        sum(1 for r in res.get("qa_results", []) if r.get("answer_correct"))
        for res in all_results
    )
    total_questions = sum(len(res.get("qa_results", [])) for res in all_results)
    
    for persona in personas:
        # Skip if already completed in phase 2
        if persona in phase2_completed_set:
            print(f"\n[{persona}] QA already completed (skipping)")
            continue
        
        # Skip if phase 1 not completed for this persona
        if persona not in phase1_completed_set:
            print(f"\n[{persona}] Phase 1 not completed, skipping QA")
            continue
        
        userid = f"{args.userid_prefix}_{persona}"
        qa_items = personas_qa.get(persona, [])
        
        if not qa_items:
            print(f"\n[{persona}] No QA questions, skipping")
            all_results.append({
                "persona": persona,
                "storage": storage_stats.get(persona, {}),
                "qa_results": [],
            })
            phase2_completed.append(persona)
            phase2_completed_set.add(persona)
            continue
        
        print(f"\n[{persona}] {len(qa_items)} QA questions")
        
        # Check for partial progress from previous run
        existing_qa_results: List[Dict[str, Any]] = []
        if current_persona_progress and current_persona_progress.get("persona") == persona:
            existing_qa_results = current_persona_progress.get("qa_results", [])
            print(f"  Resuming from {len(existing_qa_results)} previously answered questions")
            current_persona_progress = None  # Clear after using
        
        try:
            # Reset short-term session (keep memory) - only if not resuming
            if not existing_qa_results:
                throttle(args.delay_sec)
                try:
                    reset_short_term_session(
                        args.server_url, userid, args.timeout, preserve_memory=True
                    )
                    print(f"  Reset short-term session (memory preserved)")
                except Exception as exc:
                    print(f"  [WARN] Failed to reset session: {exc}")
            
            # Create callback for batch-level checkpoint saving
            def on_batch_complete(batch_idx: int, batch_results: List[Dict[str, Any]]):
                save_checkpoint(
                    args.checkpoint_file,
                    phase1_completed,
                    phase2_completed,
                    storage_stats,
                    all_results,
                    config,
                    current_persona_progress={
                        "persona": persona,
                        "qa_results": batch_results,
                    },
                )
                print(f"      [CHECKPOINT] Batch {batch_idx} saved ({len(batch_results)} total)")
            
            # Ask questions in batches
            qa_results = ask_qa_batches(
                userid=userid,
                qa_items=qa_items,
                server_url=args.server_url,
                system_prompt=args.system_prompt,
                model=args.model,
                max_search_results=args.max_search_results,
                timeout=args.timeout,
                delay_sec=args.delay_sec,
                max_qa_chars=args.max_qa_chars,
                max_questions_per_batch=args.max_questions_per_batch,
                existing_results=existing_qa_results,
                on_batch_complete=on_batch_complete,
                max_retries=args.max_retries,
            )
            
            correct = sum(1 for r in qa_results if r["answer_correct"])
            total = len(qa_results)
            total_correct += correct
            total_questions += total
            
            print(f"  -> Total accuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "  -> No results")
            
            all_results.append({
                "persona": persona,
                "storage": storage_stats.get(persona, {}),
                "qa_results": qa_results,
            })
            
            # Mark as completed and save checkpoint
            phase2_completed.append(persona)
            phase2_completed_set.add(persona)
            
            save_checkpoint(
                args.checkpoint_file,
                phase1_completed,
                phase2_completed,
                storage_stats,
                all_results,
                config,
            )
            print(f"  [CHECKPOINT] Saved (Phase 2: {len(phase2_completed)}/{len(personas)})")
            
        except Exception as e:
            print(f"  [ERROR] QA failed for {persona}: {e}")
            print(f"  [INFO] Run with --resume to retry this persona")
            continue

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Personas tested: {len(all_results)}")
    print(f"Total memory chunks stored: {total_chunks_stored}")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {total_correct}")
    if total_questions > 0:
        print(f"Overall accuracy: {total_correct/total_questions*100:.1f}%")
    
    # Category breakdown
    category_totals: Dict[Any, int] = defaultdict(int)
    category_correct: Dict[Any, int] = defaultdict(int)
    for result in all_results:
        for qa in result.get("qa_results", []):
            cat = qa.get("category")
            category_totals[cat] += 1
            if qa.get("answer_correct"):
                category_correct[cat] += 1
    
    if category_totals:
        print("\nPer-category accuracy:")
        for cat in sorted(category_totals.keys(), key=lambda x: (x is None, x)):
            tot = category_totals[cat]
            corr = category_correct[cat]
            pct = corr / tot * 100 if tot > 0 else 0
            print(f"  Category {cat}: {corr}/{tot} ({pct:.1f}%)")

    summary = {
        "personas_run": len(all_results),
        "total_chunks_stored": total_chunks_stored,
        "total_chars_stored": total_chars_stored,
        "qa_accuracy_total": total_correct,
        "qa_questions_total": total_questions,
        "accuracy_percent": (total_correct / total_questions * 100) if total_questions > 0 else 0,
        "category_totals": dict(category_totals),
        "category_correct": dict(category_correct),
        "details": all_results,
    }

    output_path = os.path.join(os.getcwd(), args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults written to: {output_path}")
    
    # Check if all personas completed successfully
    all_completed = (
        len(phase1_completed_set) == len(personas) and
        len(phase2_completed_set) == len(personas)
    )
    
    if all_completed:
        delete_checkpoint(args.checkpoint_file)
        print("\n[SUCCESS] All personas completed successfully!")
    else:
        print(f"\n[INCOMPLETE] Some personas not completed:")
        print(f"  Phase 1 incomplete: {set(personas) - phase1_completed_set}")
        print(f"  Phase 2 incomplete: {set(personas) - phase2_completed_set}")
        print(f"  Run with --resume to continue")


if __name__ == "__main__":
    main()
