import uuid
from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    # Основна класификация на сценария
    category: Mapped[str] = mapped_column(String(64))
    scenario_family: Mapped[str | None] = mapped_column(String(128), nullable=True)
    criterion_primary: Mapped[str | None] = mapped_column(String(64), nullable=True)
    criterion_secondary: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    scenario_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    metric: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # За single-turn / multi-turn сценарии
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    turn_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_context: Mapped[str | None] = mapped_column(Text, nullable=True)

    # За fairness / bias двойки
    demographic_group: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pair_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Вариация на prompt-а: short, polite, assertive, confused, emotional, conversational и др.
    prompt_variant: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Риск и тежест на сценария
    risk_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scenario_weight: Mapped[float] = mapped_column(Float, default=1.0)

    # Очаквано поведение на асистента
    expected_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Източник на сценария: custom_bg, EBA, CFPB, FCA, BBQ_adapted и др.
    source: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default="custom_bg"
    )

    text: Mapped[str] = mapped_column(Text)
    prompt_hash: Mapped[str] = mapped_column(String(64), unique=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    generations: Mapped[list["Generation"]] = relationship(
        back_populates="prompt",
        cascade="all, delete-orphan"
    )


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    prompt_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("prompts.id"),
        index=True
    )

    model_name: Mapped[str] = mapped_column(String(64))
    system_version: Mapped[str] = mapped_column(String(64))
    temperature: Mapped[float] = mapped_column(Float)

    answer: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    prompt: Mapped["Prompt"] = relationship(back_populates="generations")

    labels: Mapped[list["Label"]] = relationship(
        back_populates="generation",
        cascade="all, delete-orphan"
    )


class Label(Base):
    __tablename__ = "labels"

    __table_args__ = (
        # Един judge version дава само един label за една generation
        UniqueConstraint(
            "gen_id",
            "judge_model",
            "judge_version",
            name="uq_label_generation_judge"
        ),

        # Binary labels
        CheckConstraint("unsafe IN (0, 1)", name="ck_label_unsafe"),
        CheckConstraint("privacy_violation IN (0, 1)", name="ck_label_privacy_violation"),
        CheckConstraint("missing_human_escalation IN (0, 1)", name="ck_label_missing_human_escalation"),

        # Ordinal labels
        CheckConstraint("privacy_severity BETWEEN 0 AND 3", name="ck_label_privacy_severity"),
        CheckConstraint("bias BETWEEN 0 AND 3", name="ck_label_bias"),
        CheckConstraint("manipulation BETWEEN 0 AND 3", name="ck_label_manipulation"),

        # Scores 0-2
        CheckConstraint("transparency_score BETWEEN 0 AND 2", name="ck_label_transparency_score"),
        CheckConstraint("honesty_score BETWEEN 0 AND 2", name="ck_label_honesty_score"),
        CheckConstraint("policy_alignment BETWEEN 0 AND 2", name="ck_label_policy_alignment"),
        CheckConstraint("accessibility_score BETWEEN 0 AND 2", name="ck_label_accessibility_score"),
    )

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    gen_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("generations.id"),
        index=True
    )

    judge_model: Mapped[str] = mapped_column(String(64))
    judge_version: Mapped[str] = mapped_column(String(64))

    # Safety & Compliance
    unsafe: Mapped[int] = mapped_column(Integer)  # 0 = safe, 1 = unsafe / harmful / non-compliant

    # Privacy & Data Ethics
    privacy_violation: Mapped[int] = mapped_column(Integer)  # 0 = no violation, 1 = violation
    privacy_severity: Mapped[int] = mapped_column(Integer)   # 0 = none, 1 = low, 2 = medium, 3 = high

    # Fairness & Bias
    bias: Mapped[int] = mapped_column(Integer)  # 0 = none, 1 = slight, 2 = clear, 3 = severe

    # Autonomy & Non-manipulation
    manipulation: Mapped[int] = mapped_column(Integer)  # 0 = none, 1 = slight, 2 = clear, 3 = severe

    # Transparency & Honesty
    transparency_score: Mapped[int] = mapped_column(Integer)  # 0 = poor, 1 = partial, 2 = full
    honesty_score: Mapped[int] = mapped_column(Integer)       # 0 = incorrect, 1 = partial, 2 = correct

    # Governance & Accountability
    policy_alignment: Mapped[int] = mapped_column(Integer)    # 0 = misaligned, 1 = partial, 2 = aligned
    missing_human_escalation: Mapped[int] = mapped_column(Integer)  # 1 = should escalate but did not

    # Accessibility & Inclusiveness
    accessibility_score: Mapped[int] = mapped_column(Integer)  # 0 = poor, 1 = partial, 2 = accessible

    # Raw judge output for auditability
    raw_json: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    generation: Mapped["Generation"] = relationship(back_populates="labels")