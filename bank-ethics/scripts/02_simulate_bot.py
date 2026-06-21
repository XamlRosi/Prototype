#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved simulated bank chatbot generator.

Pipeline:
1. Loads prompts from CSV into Prompt table
2. Selects system profile:
   - compliant
   - risky
   - overconfident
   - privacy_risky
   - no_escalation
   - sales_pushy
   - low_accessibility
3. Calls target LLM
4. Stores answers in Generation table

Important improvements:
- Multiple risky profiles instead of one generic risky prompt
- Consistent profile selection for pair_id / conversation_id / scenario_id groups
- Category-based profile selection for better label balance
- Safer fairness/bias generation: both rows in a pair use the same profile
- limit=0 means all prompts

Usage examples:

Compliant answers for all prompts:
  python scripts/02_simulate_bot.py \
    --csv data/prompts_all_with_conversations.csv \
    --mode compliant \
    --limit 0

Category-specific risky profiles:
  python scripts/02_simulate_bot.py \
    --csv data/prompts_all_with_conversations.csv \
    --mode profile_by_category \
    --limit 0

Mixed, but consistent per group:
  python scripts/02_simulate_bot.py \
    --csv data/prompts_all_with_conversations.csv \
    --mode mixed_profile_by_category \
    --risky-rate 0.40 \
    --seed 42 \
    --limit 0

Only human escalation with no_escalation risky profile:
  python scripts/02_simulate_bot.py \
    --csv data/prompts_all_with_conversations.csv \
    --category human_escalation \
    --mode risky \
    --risky-profile no_escalation \
    --limit 0
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import time
import argparse
import random
import csv
import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv
from sqlalchemy import and_
from openai import OpenAI

from bank_ethics.db.base import SessionLocal
from bank_ethics.db.models import Prompt, Generation

load_dotenv()


# -----------------------------
# Defaults
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")

MODEL_DEFAULT = "gpt-4.1-mini"
TEMPERATURE_DEFAULT = 0.3
SYSTEM_VERSION_DEFAULT = "system_v1"

DEFAULT_PROFILE_PATHS = {
    "compliant": "prompts/system_bank_v1.txt",
    "risky": "prompts/system_bank_risky_v1.txt",
    "overconfident": "prompts/system_bank_overconfident.txt",
    "privacy_risky": "prompts/system_bank_privacy_risky.txt",
    "no_escalation": "prompts/system_bank_no_escalation.txt",
    "sales_pushy": "prompts/system_bank_sales_pushy.txt",
    "low_accessibility": "prompts/system_bank_low_accessibility.txt",
}

DEFAULT_CATEGORY_PROFILE_MAP = {
    "transparency": "overconfident",
    "personalized_financial_advice": "sales_pushy",
    "fees_and_deadlines": "overconfident",
    "fraud_and_unauthorized_transactions": "no_escalation",
    "human_escalation": "no_escalation",
    "vulnerable_customers": "low_accessibility",
    "privacy_data_minimization": "privacy_risky",
    "accessibility_clarity_multilingual": "low_accessibility",
    "complaints_and_recovery": "no_escalation",
    "bias": "sales_pushy",
}


# -----------------------------
# Utility
# -----------------------------

def clean_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def to_int_or_none(v: Optional[str]) -> Optional[int]:
    v = clean_str(v)
    return int(v) if v is not None else None


def to_float(v: Optional[str], default: float = 1.0) -> float:
    v = clean_str(v)
    return float(v) if v is not None else default


def load_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"System prompt file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return f.read().strip()


def load_profile_prompts(args) -> Dict[str, str]:
    profile_paths = dict(DEFAULT_PROFILE_PATHS)

    if args.profile_paths_json:
        custom_path = Path(args.profile_paths_json)
        if not custom_path.exists():
            raise FileNotFoundError(f"--profile-paths-json not found: {args.profile_paths_json}")
        with custom_path.open("r", encoding="utf-8") as f:
            profile_paths.update(json.load(f))

    profile_paths["compliant"] = args.system_path
    profile_paths["risky"] = args.system_path_risky

    loaded: Dict[str, str] = {}

    for profile_name, path in profile_paths.items():
        p = Path(path)
        if p.exists():
            loaded[profile_name] = load_text_file(path)

    if "compliant" not in loaded:
        raise FileNotFoundError(
            f"Compliant system prompt not found: {args.system_path}"
        )

    return loaded


def load_category_profile_map(args) -> Dict[str, str]:
    mapping = dict(DEFAULT_CATEGORY_PROFILE_MAP)

    if not args.category_profile_map:
        return mapping

    raw = args.category_profile_map.strip()

    maybe_file = Path(raw)
    if maybe_file.exists():
        with maybe_file.open("r", encoding="utf-8") as f:
            mapping.update(json.load(f))
        return mapping

    try:
        mapping.update(json.loads(raw))
        return mapping
    except Exception as e:
        raise ValueError(
            "--category-profile-map must be either JSON string or path to JSON file."
        ) from e


def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 5,
) -> str:
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20) + 0.2 * attempt
            print(f"[WARN] LLM call failed attempt {attempt + 1}/{max_retries}: {e}")
            print(f"[WARN] Sleeping {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


# -----------------------------
# Duplicate checks
# -----------------------------

def already_generated(
    db,
    prompt_id: str,
    model: str,
    system_version: str,
    temperature: float,
) -> bool:
    existing = (
        db.query(Generation)
        .filter(
            and_(
                Generation.prompt_id == prompt_id,
                Generation.model_name == model,
                Generation.system_version == system_version,
                Generation.temperature == temperature,
            )
        )
        .first()
    )
    return existing is not None


# -----------------------------
# CSV import
# -----------------------------

def build_prompt_hash_from_row(row: Dict[str, Any]) -> str:
    text = clean_str(row.get("text")) or clean_str(row.get("prompt")) or ""

    raw = "||".join([
        clean_str(row.get("category")) or clean_str(row.get("family_key")) or "",
        clean_str(row.get("scenario_family")) or clean_str(row.get("family_label")) or "",
        clean_str(row.get("scenario_id")) or "",
        clean_str(row.get("criterion_primary")) or clean_str(row.get("criterion")) or "",
        clean_str(row.get("criterion_secondary")) or "",
        clean_str(row.get("metric")) or "",
        clean_str(row.get("conversation_id")) or "",
        clean_str(row.get("turn_id")) or "",
        clean_str(row.get("pair_id")) or "",
        clean_str(row.get("demographic_group")) or "",
        clean_str(row.get("prompt_variant")) or clean_str(row.get("style")) or "",
        text,
    ])

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    text = clean_str(row.get("text")) or clean_str(row.get("prompt"))
    if not text:
        raise ValueError("CSV row is missing required field 'text' or 'prompt'.")

    prompt_hash = clean_str(row.get("prompt_hash")) or build_prompt_hash_from_row(row)

    return {
        "category": (
            clean_str(row.get("category"))
            or clean_str(row.get("family_key"))
            or "custom_bg"
        ),
        "scenario_family": (
            clean_str(row.get("scenario_family"))
            or clean_str(row.get("family_label"))
        ),
        "scenario_id": clean_str(row.get("scenario_id")),
        "metric": clean_str(row.get("metric")),
        "criterion_primary": (
            clean_str(row.get("criterion_primary"))
            or clean_str(row.get("criterion"))
        ),
        "criterion_secondary": clean_str(row.get("criterion_secondary")),
        "conversation_id": clean_str(row.get("conversation_id")),
        "turn_id": to_int_or_none(row.get("turn_id")),
        "previous_context": clean_str(row.get("previous_context")),
        "demographic_group": clean_str(row.get("demographic_group")),
        "pair_id": clean_str(row.get("pair_id")),
        "prompt_variant": (
            clean_str(row.get("prompt_variant"))
            or clean_str(row.get("style"))
        ),
        "risk_level": clean_str(row.get("risk_level")),
        "scenario_weight": to_float(row.get("scenario_weight"), default=1.0),
        "expected_behavior": (
            clean_str(row.get("expected_behavior"))
            or clean_str(row.get("expected_safe_behavior"))
        ),
        "source": clean_str(row.get("source")) or "custom_bg",
        "text": text,
        "prompt_hash": prompt_hash,
    }


def load_prompts_from_csv(db, csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] CSV file not found: {csv_path}")
        return 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    inserted = 0
    skipped = 0
    failed = 0

    for row in rows:
        try:
            data = normalize_csv_row(row)

            existing = (
                db.query(Prompt)
                .filter(Prompt.prompt_hash == data["prompt_hash"])
                .first()
            )

            if existing:
                skipped += 1
                continue

            db.add(Prompt(**data))
            db.commit()
            inserted += 1

        except Exception as e:
            db.rollback()
            failed += 1
            print(f"[WARN] Failed to insert prompt row: {e}")

    print(f"Prompt import: inserted={inserted}, skipped={skipped}, failed={failed}")
    return inserted


# -----------------------------
# Prompt construction
# -----------------------------

def build_user_prompt_for_generation(p: Prompt) -> str:
    previous_context = getattr(p, "previous_context", None)

    if previous_context:
        return (
            "Предишен контекст на разговора:\n"
            f"{previous_context}\n\n"
            "Текущо съобщение на потребителя:\n"
            f"{p.text}"
        )

    return p.text


# -----------------------------
# Profile selection
# -----------------------------

def profile_group_key(p: Prompt) -> str:
    """
    Ensures consistent profile selection for related prompts.

    Priority:
    - pair_id: fairness pairs must use the same system profile
    - conversation_id: turns in the same conversation use same profile
    - scenario_id: style variants of same scenario use same profile
    - prompt id
    """

    pair_id = clean_str(getattr(p, "pair_id", None))
    if pair_id:
        return f"pair:{pair_id}"

    conversation_id = clean_str(getattr(p, "conversation_id", None))
    if conversation_id:
        return f"conversation:{conversation_id}"

    scenario_id = clean_str(getattr(p, "scenario_id", None))
    if scenario_id:
        return f"scenario:{scenario_id}"

    return f"prompt:{p.id}"


def category_risky_profile(
    p: Prompt,
    category_map: Dict[str, str],
    fallback: str,
) -> str:
    category = clean_str(getattr(p, "category", None))
    if category and category in category_map:
        return category_map[category]
    return fallback


def select_profile_name(
    *,
    p: Prompt,
    args,
    category_map: Dict[str, str],
    group_profile_cache: Dict[str, str],
) -> str:
    key = profile_group_key(p)

    if key in group_profile_cache:
        return group_profile_cache[key]

    if args.mode == "compliant":
        profile = "compliant"

    elif args.mode == "risky":
        profile = args.risky_profile

    elif args.mode == "mixed":
        profile = args.risky_profile if random.random() < args.risky_rate else "compliant"

    elif args.mode == "profile_by_category":
        profile = category_risky_profile(
            p=p,
            category_map=category_map,
            fallback=args.risky_profile,
        )

    elif args.mode == "mixed_profile_by_category":
        if random.random() < args.risky_rate:
            profile = category_risky_profile(
                p=p,
                category_map=category_map,
                fallback=args.risky_profile,
            )
        else:
            profile = "compliant"

    else:
        raise ValueError(
            "mode must be compliant | risky | mixed | profile_by_category | mixed_profile_by_category"
        )

    group_profile_cache[key] = profile
    return profile


def validate_profile_available(profile_name: str, profiles: Dict[str, str]) -> None:
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            f"Profile '{profile_name}' was selected but no system prompt was loaded for it. "
            f"Available profiles: {available}"
        )


# -----------------------------
# Query
# -----------------------------

def build_prompt_query(db, args):
    query = db.query(Prompt)

    if args.category:
        query = query.filter(Prompt.category == args.category)

    if args.scenario_family:
        query = query.filter(Prompt.scenario_family == args.scenario_family)

    if args.scenario_id:
        query = query.filter(Prompt.scenario_id == args.scenario_id)

    query = query.order_by(
        Prompt.category.asc(),
        Prompt.scenario_id.asc(),
        Prompt.conversation_id.asc(),
        Prompt.turn_id.asc(),
        Prompt.pair_id.asc(),
        Prompt.demographic_group.asc(),
        Prompt.prompt_variant.asc(),
        Prompt.created_at.asc(),
    )

    if args.offset:
        query = query.offset(args.offset)

    if args.limit and args.limit > 0:
        query = query.limit(args.limit)

    return query


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=MODEL_DEFAULT)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    parser.add_argument("--system-version", type=str, default=SYSTEM_VERSION_DEFAULT)

    parser.add_argument("--system-path", type=str, default=DEFAULT_PROFILE_PATHS["compliant"])
    parser.add_argument("--system-path-risky", type=str, default=DEFAULT_PROFILE_PATHS["risky"])

    parser.add_argument(
        "--profile-paths-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with profile_name -> system_prompt_path. "
            "Example: {'no_escalation': 'prompts/system_bank_no_escalation.txt'}"
        ),
    )

    parser.add_argument(
        "--category-profile-map",
        type=str,
        default=None,
        help=(
            "Optional JSON string or JSON file path mapping category -> profile. "
            "Used in profile_by_category and mixed_profile_by_category modes."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "compliant",
            "risky",
            "mixed",
            "profile_by_category",
            "mixed_profile_by_category",
        ],
        default="mixed_profile_by_category",
    )

    parser.add_argument(
        "--risky-profile",
        type=str,
        default="risky",
        help="Risky profile used in risky/mixed modes. Example: no_escalation.",
    )

    parser.add_argument(
        "--risky-rate",
        type=float,
        default=0.35,
        help="Used only in mixed and mixed_profile_by_category.",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="How many prompts to process. 0 = all prompts.",
    )

    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)

    parser.add_argument(
        "--csv",
        type=str,
        default="data/prompts_all_with_conversations.csv",
        help="CSV file to load prompts from.",
    )

    parser.add_argument(
        "--no-csv-import",
        action="store_true",
        help="Do not import prompts from CSV; use prompts already in DB.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional filter by Prompt.category, e.g. transparency.",
    )

    parser.add_argument(
        "--scenario-family",
        type=str,
        default=None,
        help="Optional filter by Prompt.scenario_family.",
    )

    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Optional filter by Prompt.scenario_id.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned prompts and profiles without calling the LLM.",
    )

    args = parser.parse_args()

    if args.mode in ("mixed", "mixed_profile_by_category") and not (0.0 <= args.risky_rate <= 1.0):
        raise ValueError("--risky-rate must be between 0 and 1.")

    random.seed(args.seed)

    profiles = load_profile_prompts(args)
    category_map = load_category_profile_map(args)

    print("\n=== Loaded system profiles ===")
    for name in sorted(profiles.keys()):
        print(f"- {name}")

    if args.mode in ("profile_by_category", "mixed_profile_by_category"):
        print("\n=== Category profile map ===")
        for cat, prof in sorted(category_map.items()):
            print(f"- {cat}: {prof}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    db = SessionLocal()

    group_profile_cache: Dict[str, str] = {}

    try:
        if not args.no_csv_import:
            inserted = load_prompts_from_csv(db, args.csv)
            if inserted > 0:
                print(f"Loaded {inserted} new prompts from {args.csv}")

        prompts = build_prompt_query(db, args).all()
        print(f"\nLoaded prompts from DB: {len(prompts)}")

        if not prompts:
            print("No prompts selected.")
            return

        planned_profiles: Dict[str, int] = {}

        for p in prompts:
            profile_name = select_profile_name(
                p=p,
                args=args,
                category_map=category_map,
                group_profile_cache=group_profile_cache,
            )
            validate_profile_available(profile_name, profiles)
            planned_profiles[profile_name] = planned_profiles.get(profile_name, 0) + 1

        print("\n=== Planned profile distribution ===")
        for profile_name, count in sorted(planned_profiles.items()):
            print(f"- {profile_name}: {count}")

        if args.dry_run:
            print("\nDry run enabled. No generations will be created.")
            for i, p in enumerate(prompts[:30], 1):
                profile_name = select_profile_name(
                    p=p,
                    args=args,
                    category_map=category_map,
                    group_profile_cache=group_profile_cache,
                )
                print(
                    f"[{i}] category={p.category} scenario_id={p.scenario_id} "
                    f"pair_id={p.pair_id} conversation_id={p.conversation_id} "
                    f"profile={profile_name} text={p.text[:80]}"
                )
            return

        saved = 0
        skipped = 0
        failed = 0

        for i, p in enumerate(prompts, 1):
            profile_name = select_profile_name(
                p=p,
                args=args,
                category_map=category_map,
                group_profile_cache=group_profile_cache,
            )

            validate_profile_available(profile_name, profiles)

            system_prompt = profiles[profile_name]
            system_version = f"{args.system_version}_{profile_name}"

            if already_generated(
                db=db,
                prompt_id=p.id,
                model=args.model,
                system_version=system_version,
                temperature=args.temperature,
            ):
                skipped += 1
                continue

            user_prompt = build_user_prompt_for_generation(p)

            print(
                f"\n[{i}/{len(prompts)}] "
                f"PromptID={p.id} "
                f"category={p.category} "
                f"scenario_id={getattr(p, 'scenario_id', None)} "
                f"metric={getattr(p, 'metric', None)} "
                f"conversation_id={getattr(p, 'conversation_id', None)} "
                f"turn_id={getattr(p, 'turn_id', None)} "
                f"pair_id={getattr(p, 'pair_id', None)} "
                f"group={getattr(p, 'demographic_group', None)}"
            )

            print(f"SOURCE={getattr(p, 'source', None)}")
            print(f"RISK_LEVEL={getattr(p, 'risk_level', None)}")
            print(f"PROMPT_VARIANT={getattr(p, 'prompt_variant', None)}")
            print(f"PROFILE={profile_name} system_version={system_version}")
            print(f"USER_PROMPT:\n{user_prompt}")

            try:
                answer = call_llm(
                    client=client,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                )

                db.add(
                    Generation(
                        prompt_id=p.id,
                        model_name=args.model,
                        system_version=system_version,
                        temperature=args.temperature,
                        answer=answer,
                    )
                )

                db.commit()
                saved += 1

                preview = answer[:300].replace("\n", " ")
                if len(answer) > 300:
                    preview += "..."

                print("ASSISTANT:", preview)

            except Exception as e:
                db.rollback()
                failed += 1
                print(f"[ERROR] Failed generation for prompt {p.id}: {e}")
                continue

            time.sleep(args.sleep)

    finally:
        db.close()

    print("\nDone.")
    print(f"Saved generations: {saved}")
    print(f"Skipped existing generations: {skipped}")
    print(f"Failed generations: {failed}")


if __name__ == "__main__":
    main()