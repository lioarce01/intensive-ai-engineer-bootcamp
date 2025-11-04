# Responsible AI Guide

> **Building AI systems that are fair, transparent, safe, and beneficial for everyone.**

## Table of Contents
1. [AI Ethics Fundamentals](#ai-ethics)
2. [Bias Detection and Mitigation](#bias)
3. [Safety Guardrails](#safety-guardrails)
4. [RLHF: Reinforcement Learning from Human Feedback](#rlhf)
5. [Compliance and Governance](#compliance)

---

## AI Ethics Fundamentals

### Core Principles

#### 1. Fairness and Non-Discrimination

**Definition**: AI systems should treat all individuals and groups equitably, without discriminatory bias.

```python
"""
Key Questions for Fairness:
- Does the model perform equally well across demographic groups?
- Are certain groups systematically disadvantaged?
- Does the training data represent all populations fairly?
- Are there disparate impacts in outcomes?
"""
```

#### 2. Transparency and Explainability

**Definition**: AI systems should be understandable and their decisions should be explainable.

```python
class TransparentModel:
    """
    Model with built-in explainability features.
    """

    def __init__(self, model):
        self.model = model
        self.decision_log = []

    def predict_with_explanation(self, input_data):
        """Make prediction and log reasoning."""
        prediction = self.model(input_data)

        # Log decision
        explanation = {
            "input": input_data,
            "prediction": prediction,
            "confidence": self.get_confidence(prediction),
            "top_features": self.get_important_features(input_data),
            "timestamp": datetime.now()
        }

        self.decision_log.append(explanation)
        return prediction, explanation

    def get_important_features(self, input_data):
        """Return most influential features for this prediction."""
        # Use techniques like SHAP, LIME, or attention weights
        pass
```

#### 3. Privacy and Data Protection

**Definition**: Protect user data and respect privacy throughout the AI lifecycle.

```python
import hashlib
from typing import Dict, Any

class PrivacyPreservingSystem:
    """Handle data with privacy protection."""

    @staticmethod
    def anonymize_pii(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or hash personally identifiable information."""
        pii_fields = ["email", "phone", "ssn", "address", "name"]

        anonymized = data.copy()
        for field in pii_fields:
            if field in anonymized:
                # Hash sensitive data
                value = str(anonymized[field])
                anonymized[field] = hashlib.sha256(
                    value.encode()
                ).hexdigest()[:16]

        return anonymized

    @staticmethod
    def minimize_data_collection(data: Dict[str, Any], required_fields: list):
        """Only keep necessary data (data minimization principle)."""
        return {k: v for k, v in data.items() if k in required_fields}

    @staticmethod
    def implement_right_to_deletion(user_id: str, database):
        """Allow users to delete their data (GDPR compliance)."""
        database.delete_user_data(user_id)
        database.anonymize_historical_records(user_id)
```

#### 4. Accountability and Governance

**Definition**: Clear responsibility for AI system behavior and outcomes.

```python
class AuditableAISystem:
    """AI system with comprehensive audit trail."""

    def __init__(self):
        self.audit_log = []

    def log_decision(
        self,
        input_data,
        output,
        model_version,
        operator_id,
        metadata=None
    ):
        """Log every AI decision for accountability."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": self._hash_input(input_data),
            "output": output,
            "model_version": model_version,
            "operator_id": operator_id,
            "metadata": metadata or {}
        }

        self.audit_log.append(audit_entry)
        self._persist_audit_log(audit_entry)

    def _hash_input(self, data):
        """Create hash of input for privacy."""
        return hashlib.sha256(str(data).encode()).hexdigest()

    def generate_audit_report(self, start_date, end_date):
        """Generate audit report for governance review."""
        filtered_logs = [
            log for log in self.audit_log
            if start_date <= log["timestamp"] <= end_date
        ]

        report = {
            "period": f"{start_date} to {end_date}",
            "total_decisions": len(filtered_logs),
            "model_versions_used": self._count_versions(filtered_logs),
            "operators": self._count_operators(filtered_logs),
            "error_rate": self._calculate_error_rate(filtered_logs)
        }

        return report
```

### Ethical Frameworks

#### Utilitarian Approach
Maximize overall benefit and minimize harm for the greatest number of people.

```python
def evaluate_utilitarian_impact(model, test_cases, stakeholders):
    """
    Evaluate model impact from utilitarian perspective.

    Args:
        model: AI model to evaluate
        test_cases: Test scenarios
        stakeholders: Different affected groups

    Returns:
        Overall utility score
    """
    total_utility = 0

    for case in test_cases:
        prediction = model(case)

        # Calculate utility for each stakeholder group
        for stakeholder in stakeholders:
            benefit = calculate_benefit(prediction, stakeholder, case)
            harm = calculate_harm(prediction, stakeholder, case)
            utility = benefit - harm

            total_utility += utility * stakeholder.weight

    return total_utility / (len(test_cases) * len(stakeholders))
```

#### Deontological Approach
Follow ethical rules and duties regardless of outcomes.

```python
class DeontologicalGuardrails:
    """Enforce ethical rules in AI system."""

    RULES = [
        "Never discriminate based on protected attributes",
        "Always obtain informed consent",
        "Never deceive users",
        "Respect user autonomy",
        "Protect vulnerable populations"
    ]

    @staticmethod
    def check_consent(user_data):
        """Ensure informed consent was obtained."""
        if not user_data.get("consent_given"):
            raise EthicalViolation("Consent not obtained")

    @staticmethod
    def check_protected_attributes(features):
        """Ensure protected attributes aren't used improperly."""
        protected = ["race", "gender", "age", "religion", "disability"]

        for attr in protected:
            if attr in features:
                raise EthicalViolation(
                    f"Protected attribute '{attr}' used in decision-making"
                )

    @staticmethod
    def check_transparency(model_output):
        """Ensure decision can be explained to user."""
        if not model_output.get("explanation"):
            raise EthicalViolation("Decision lacks explanation")
```

---

## Bias Detection and Mitigation

### Types of Bias

#### 1. Data Bias

```python
import pandas as pd
from collections import Counter

class DataBiasDetector:
    """Detect various types of bias in datasets."""

    @staticmethod
    def check_representation_bias(data, demographic_col):
        """Check if all groups are adequately represented."""
        counts = Counter(data[demographic_col])
        total = len(data)

        representation = {
            group: count / total
            for group, count in counts.items()
        }

        # Flag underrepresented groups (< 5% of data)
        underrepresented = {
            group: pct for group, pct in representation.items()
            if pct < 0.05
        }

        return {
            "representation": representation,
            "underrepresented_groups": underrepresented,
            "bias_detected": len(underrepresented) > 0
        }

    @staticmethod
    def check_label_bias(data, label_col, demographic_col):
        """Check for biased label distributions across groups."""
        label_dist = data.groupby(demographic_col)[label_col].value_counts(
            normalize=True
        ).unstack()

        # Calculate divergence in label distributions
        baseline = data[label_col].value_counts(normalize=True)
        divergences = {}

        for group in label_dist.index:
            group_dist = label_dist.loc[group]
            divergence = ((group_dist - baseline) ** 2).sum()
            divergences[group] = divergence

        return {
            "label_distributions": label_dist,
            "divergences": divergences,
            "bias_detected": max(divergences.values()) > 0.1
        }

    @staticmethod
    def check_missing_data_bias(data, demographic_col):
        """Check if missing data correlates with demographics."""
        missing_rates = {}

        for group in data[demographic_col].unique():
            group_data = data[data[demographic_col] == group]
            missing_rate = group_data.isnull().sum().sum() / group_data.size
            missing_rates[group] = missing_rate

        avg_missing = sum(missing_rates.values()) / len(missing_rates)
        variance = sum(
            (rate - avg_missing) ** 2
            for rate in missing_rates.values()
        ) / len(missing_rates)

        return {
            "missing_rates_by_group": missing_rates,
            "variance": variance,
            "bias_detected": variance > 0.01
        }
```

#### 2. Model Bias

```python
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

class ModelFairnessEvaluator:
    """Evaluate model fairness across demographic groups."""

    @staticmethod
    def demographic_parity(y_pred, sensitive_attribute):
        """
        Check if positive prediction rate is equal across groups.

        Demographic Parity: P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
        """
        groups = {}

        for group in np.unique(sensitive_attribute):
            mask = sensitive_attribute == group
            positive_rate = np.mean(y_pred[mask])
            groups[group] = positive_rate

        # Calculate max difference
        rates = list(groups.values())
        max_diff = max(rates) - min(rates)

        return {
            "positive_rates": groups,
            "max_difference": max_diff,
            "passes": max_diff < 0.1  # 10% threshold
        }

    @staticmethod
    def equal_opportunity(y_true, y_pred, sensitive_attribute):
        """
        Check if true positive rate is equal across groups.

        Equal Opportunity: P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)
        """
        groups = {}

        for group in np.unique(sensitive_attribute):
            mask = sensitive_attribute == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]

            # True positive rate
            true_positives = np.sum((group_true == 1) & (group_pred == 1))
            actual_positives = np.sum(group_true == 1)

            tpr = true_positives / actual_positives if actual_positives > 0 else 0
            groups[group] = tpr

        # Calculate max difference
        tprs = list(groups.values())
        max_diff = max(tprs) - min(tprs)

        return {
            "true_positive_rates": groups,
            "max_difference": max_diff,
            "passes": max_diff < 0.1
        }

    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_attribute):
        """
        Check if both TPR and FPR are equal across groups.

        Equalized Odds: TPR and FPR equal across all groups
        """
        groups = {}

        for group in np.unique(sensitive_attribute):
            mask = sensitive_attribute == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]

            tn, fp, fn, tp = confusion_matrix(
                group_true, group_pred
            ).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            groups[group] = {"tpr": tpr, "fpr": fpr}

        # Calculate max differences
        tprs = [g["tpr"] for g in groups.values()]
        fprs = [g["fpr"] for g in groups.values()]

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return {
            "group_metrics": groups,
            "tpr_difference": tpr_diff,
            "fpr_difference": fpr_diff,
            "passes": tpr_diff < 0.1 and fpr_diff < 0.1
        }

    @staticmethod
    def disparate_impact(y_pred, sensitive_attribute):
        """
        Calculate disparate impact ratio.

        Disparate Impact: ratio of positive rates between groups
        Should be > 0.8 (80% rule)
        """
        groups = {}

        for group in np.unique(sensitive_attribute):
            mask = sensitive_attribute == group
            positive_rate = np.mean(y_pred[mask])
            groups[group] = positive_rate

        rates = list(groups.values())
        di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0

        return {
            "positive_rates": groups,
            "disparate_impact_ratio": di_ratio,
            "passes": di_ratio >= 0.8
        }
```

### Bias Mitigation Strategies

#### Pre-processing: Data Reweighing

```python
class DataReweigher:
    """Reweigh training samples to achieve fairness."""

    def __init__(self, sensitive_attribute):
        self.sensitive_attribute = sensitive_attribute
        self.weights = None

    def fit(self, X, y):
        """Calculate sample weights for fairness."""
        weights = np.ones(len(X))

        # Calculate expected and observed frequencies
        for group in np.unique(self.sensitive_attribute):
            for label in np.unique(y):
                mask = (self.sensitive_attribute == group) & (y == label)

                # Expected probability
                p_group = np.mean(self.sensitive_attribute == group)
                p_label = np.mean(y == label)
                expected = p_group * p_label

                # Observed probability
                observed = np.mean(mask)

                # Reweigh
                if observed > 0:
                    weights[mask] = expected / observed

        self.weights = weights
        return self

    def transform(self, X, y):
        """Return reweighed dataset."""
        return X, y, self.weights
```

#### In-processing: Adversarial Debiasing

```python
import torch
import torch.nn as nn

class AdversarialDebiasing(nn.Module):
    """
    Train model with adversarial debiasing.

    Predictor tries to make accurate predictions.
    Adversary tries to predict sensitive attribute from predictions.
    Predictor is penalized if adversary succeeds.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_groups):
        super().__init__()

        # Main predictor
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

        # Adversary (tries to predict sensitive attribute)
        self.adversary = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_groups)
        )

    def forward(self, x):
        # Get prediction
        prediction = self.predictor(x)

        # Adversary tries to infer sensitive attribute
        adv_prediction = self.adversary(prediction)

        return prediction, adv_prediction

def train_with_adversarial_debiasing(
    model,
    train_loader,
    optimizer_pred,
    optimizer_adv,
    epochs=10,
    lambda_adv=1.0
):
    """
    Train with adversarial debiasing.

    Args:
        lambda_adv: Weight for adversarial loss (higher = more fairness)
    """
    criterion_main = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_x, batch_y, batch_sensitive in train_loader:

            # Step 1: Train adversary
            optimizer_adv.zero_grad()

            pred, adv_pred = model(batch_x)
            adv_loss = criterion_adv(adv_pred, batch_sensitive)

            adv_loss.backward()
            optimizer_adv.step()

            # Step 2: Train predictor (maximize main task, fool adversary)
            optimizer_pred.zero_grad()

            pred, adv_pred = model(batch_x)

            # Main task loss
            main_loss = criterion_main(pred, batch_y)

            # Adversarial loss (negative to fool adversary)
            adv_loss = -lambda_adv * criterion_adv(adv_pred, batch_sensitive)

            # Combined loss
            total_loss = main_loss + adv_loss

            total_loss.backward()
            optimizer_pred.step()
```

#### Post-processing: Threshold Optimization

```python
class FairThresholdOptimizer:
    """Optimize decision thresholds for fairness."""

    def __init__(self, fairness_metric="equalized_odds"):
        self.fairness_metric = fairness_metric
        self.thresholds = {}

    def fit(self, y_scores, y_true, sensitive_attribute):
        """
        Find optimal thresholds for each group.

        Args:
            y_scores: Predicted probabilities
            y_true: True labels
            sensitive_attribute: Protected attribute values
        """
        for group in np.unique(sensitive_attribute):
            mask = sensitive_attribute == group

            # Find threshold that optimizes fairness + accuracy
            best_threshold = 0.5
            best_score = 0

            for threshold in np.linspace(0.1, 0.9, 81):
                y_pred = (y_scores[mask] >= threshold).astype(int)

                # Calculate accuracy
                accuracy = accuracy_score(y_true[mask], y_pred)

                # Calculate fairness (simplified)
                tpr = np.mean(y_pred[y_true[mask] == 1])
                fairness_score = 1 - abs(tpr - 0.5)  # Prefer balanced TPR

                # Combined score
                score = 0.7 * accuracy + 0.3 * fairness_score

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            self.thresholds[group] = best_threshold

        return self

    def predict(self, y_scores, sensitive_attribute):
        """Make predictions using group-specific thresholds."""
        y_pred = np.zeros(len(y_scores))

        for group, threshold in self.thresholds.items():
            mask = sensitive_attribute == group
            y_pred[mask] = (y_scores[mask] >= threshold).astype(int)

        return y_pred
```

---

## Safety Guardrails

### Content Moderation

#### Toxicity Detection

```python
from transformers import pipeline
import re

class ToxicityFilter:
    """Detect and filter toxic content."""

    def __init__(self):
        # Load toxicity detection model
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )

        # Toxicity thresholds
        self.thresholds = {
            "toxic": 0.7,
            "severe_toxic": 0.5,
            "obscene": 0.7,
            "threat": 0.5,
            "insult": 0.7,
            "identity_hate": 0.5
        }

    def check_toxicity(self, text):
        """Check if text is toxic."""
        result = self.classifier(text)[0]

        is_toxic = result["score"] > self.thresholds.get(
            result["label"], 0.7
        )

        return {
            "is_toxic": is_toxic,
            "toxicity_type": result["label"],
            "score": result["score"],
            "explanation": self._get_explanation(result)
        }

    def _get_explanation(self, result):
        """Provide explanation for toxicity detection."""
        if result["score"] < 0.5:
            return "Content appears safe"
        elif result["score"] < 0.7:
            return f"Moderate {result['label']} detected"
        else:
            return f"High {result['label']} detected"

    def filter_response(self, text):
        """Filter toxic content from model response."""
        toxicity = self.check_toxicity(text)

        if toxicity["is_toxic"]:
            return {
                "filtered": True,
                "original": text,
                "replacement": "I cannot provide that response.",
                "reason": toxicity["explanation"]
            }

        return {"filtered": False, "text": text}
```

#### PII Detection and Redaction

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIProtector:
    """Detect and redact personally identifiable information."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # PII entity types to detect
        self.pii_types = [
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "PERSON",
            "LOCATION",
            "DATE_TIME",
            "US_SSN",
            "US_PASSPORT",
            "MEDICAL_LICENSE",
            "IP_ADDRESS"
        ]

    def detect_pii(self, text):
        """Detect PII in text."""
        results = self.analyzer.analyze(
            text=text,
            entities=self.pii_types,
            language="en"
        )

        return [{
            "type": result.entity_type,
            "score": result.score,
            "start": result.start,
            "end": result.end,
            "text": text[result.start:result.end]
        } for result in results]

    def redact_pii(self, text, replacement="[REDACTED]"):
        """Redact PII from text."""
        results = self.analyzer.analyze(
            text=text,
            entities=self.pii_types,
            language="en"
        )

        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )

        return {
            "original": text,
            "redacted": anonymized.text,
            "pii_found": len(results),
            "detected_types": [r.entity_type for r in results]
        }

    def check_safe_to_log(self, text):
        """Check if text is safe to log (no PII)."""
        pii_entities = self.detect_pii(text)

        return {
            "safe_to_log": len(pii_entities) == 0,
            "pii_detected": pii_entities
        }
```

### Output Validation

#### Hallucination Detection

```python
class HallucinationDetector:
    """Detect potential hallucinations in model outputs."""

    def __init__(self):
        self.confidence_threshold = 0.7

    def check_consistency(self, model_outputs):
        """Check consistency across multiple generations."""
        if len(model_outputs) < 3:
            return {"error": "Need at least 3 outputs for consistency check"}

        # Compare similarity between outputs
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(model_outputs)
        similarities = cosine_similarity(vectors)

        # Average similarity (excluding diagonal)
        n = len(similarities)
        avg_similarity = (similarities.sum() - n) / (n * (n - 1))

        return {
            "consistent": avg_similarity > 0.7,
            "similarity_score": avg_similarity,
            "warning": avg_similarity < 0.5
        }

    def check_against_source(self, output, source_documents):
        """Verify output is grounded in source documents."""
        # Extract claims from output
        claims = self._extract_claims(output)

        # Check each claim against sources
        verified_claims = []
        unverified_claims = []

        for claim in claims:
            if self._verify_claim(claim, source_documents):
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)

        return {
            "total_claims": len(claims),
            "verified": len(verified_claims),
            "unverified": len(unverified_claims),
            "unverified_claims": unverified_claims,
            "grounding_score": len(verified_claims) / len(claims) if claims else 0
        }

    def _extract_claims(self, text):
        """Extract factual claims from text."""
        # Simple sentence splitting (can be improved with NLP)
        sentences = text.split(". ")
        return [s.strip() for s in sentences if len(s) > 20]

    def _verify_claim(self, claim, source_documents):
        """Check if claim is supported by sources."""
        # Simple keyword matching (can be improved with semantic search)
        claim_words = set(claim.lower().split())

        for doc in source_documents:
            doc_words = set(doc.lower().split())
            overlap = len(claim_words & doc_words) / len(claim_words)

            if overlap > 0.5:  # More than 50% overlap
                return True

        return False
```

### Complete Safety Layer

```python
class ComprehensiveSafetyLayer:
    """Comprehensive safety checks for AI outputs."""

    def __init__(self):
        self.toxicity_filter = ToxicityFilter()
        self.pii_protector = PIIProtector()
        self.hallucination_detector = HallucinationDetector()

    def validate_input(self, user_input):
        """Validate user input before processing."""
        checks = {}

        # Check for prompt injection
        checks["prompt_injection"] = self._check_prompt_injection(user_input)

        # Check for jailbreak attempts
        checks["jailbreak"] = self._check_jailbreak(user_input)

        # Check for malicious content
        checks["toxicity"] = self.toxicity_filter.check_toxicity(user_input)

        # Overall safety
        checks["safe"] = all([
            not checks["prompt_injection"]["detected"],
            not checks["jailbreak"]["detected"],
            not checks["toxicity"]["is_toxic"]
        ])

        return checks

    def validate_output(self, output, context=None):
        """Validate model output before returning to user."""
        checks = {}

        # Check toxicity
        checks["toxicity"] = self.toxicity_filter.check_toxicity(output)

        # Check for PII
        checks["pii"] = self.pii_protector.detect_pii(output)

        # Check for hallucinations if context provided
        if context:
            checks["hallucination"] = self.hallucination_detector.check_against_source(
                output, context
            )

        # Overall safety
        checks["safe"] = all([
            not checks["toxicity"]["is_toxic"],
            len(checks["pii"]) == 0,
            not context or checks["hallucination"]["grounding_score"] > 0.7
        ])

        return checks

    def _check_prompt_injection(self, text):
        """Detect potential prompt injection attempts."""
        injection_patterns = [
            r"ignore (previous|above) instructions",
            r"disregard (previous|all) (instructions|prompts)",
            r"you are now",
            r"new instructions:",
            r"system:",
            r"<\|im_start\|>",  # Special tokens
        ]

        detected = any(
            re.search(pattern, text.lower())
            for pattern in injection_patterns
        )

        return {
            "detected": detected,
            "confidence": 0.9 if detected else 0.1
        }

    def _check_jailbreak(self, text):
        """Detect jailbreak attempts."""
        jailbreak_patterns = [
            r"pretend (you are|to be)",
            r"roleplay as",
            r"DAN mode",
            r"developer mode",
            r"without (any |ethical |moral )?restrictions",
        ]

        detected = any(
            re.search(pattern, text.lower())
            for pattern in jailbreak_patterns
        )

        return {
            "detected": detected,
            "confidence": 0.8 if detected else 0.2
        }

    def safe_response(self, output, checks):
        """Return safe version of response."""
        if checks["safe"]:
            return output

        # Build safe response
        issues = []

        if checks.get("toxicity", {}).get("is_toxic"):
            issues.append("inappropriate content")

        if checks.get("pii") and len(checks["pii"]) > 0:
            issues.append("personal information")
            output = self.pii_protector.redact_pii(output)["redacted"]

        if checks.get("hallucination", {}).get("grounding_score", 1.0) < 0.7:
            issues.append("unverified claims")

        if issues:
            warning = f"⚠️ Response modified (removed: {', '.join(issues)})\n\n"
            return warning + output

        return "I cannot provide a safe response to this request."
```

---

## RLHF: Reinforcement Learning from Human Feedback

### Overview

RLHF aligns AI models with human preferences through a three-stage process:

1. **Supervised Fine-Tuning (SFT)**: Train on high-quality human demonstrations
2. **Reward Model Training**: Learn to predict human preferences
3. **RL Fine-Tuning**: Optimize policy using PPO against reward model

### Reward Model Training

```python
import torch.nn as nn

class RewardModel(nn.Module):
    """Learn to predict human preferences."""

    def __init__(self, base_model, hidden_dim=768):
        super().__init__()

        self.base_model = base_model  # Pre-trained LM

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Output scalar reward
        )

    def forward(self, input_ids, attention_mask):
        # Get representation from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state
        hidden = outputs.hidden_states[-1]

        # Pool (use last token or mean)
        pooled = hidden[:, -1, :]  # Last token

        # Predict reward
        reward = self.reward_head(pooled)

        return reward

def train_reward_model(model, comparison_data, optimizer, epochs=3):
    """
    Train reward model on human preference comparisons.

    Args:
        comparison_data: List of (prompt, chosen_response, rejected_response)
    """
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0

        for prompt, chosen, rejected in comparison_data:
            # Prepare inputs
            chosen_input = tokenize(prompt + chosen)
            rejected_input = tokenize(prompt + rejected)

            # Get rewards
            reward_chosen = model(**chosen_input)
            reward_rejected = model(**rejected_input)

            # Loss: chosen should have higher reward
            # Using log-sigmoid for stability
            loss = -torch.nn.functional.logsigmoid(
                reward_chosen - reward_rejected
            ).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_accuracy += (reward_chosen > reward_rejected).float().mean().item()

        avg_loss = total_loss / len(comparison_data)
        avg_accuracy = total_accuracy / len(comparison_data)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

    return model
```

### PPO Training

```python
class PPOTrainer:
    """Train policy with PPO (Proximal Policy Optimization)."""

    def __init__(
        self,
        policy_model,
        reward_model,
        ref_model,  # Reference model (frozen)
        optimizer,
        kl_coef=0.2,  # KL divergence coefficient
        clip_ratio=0.2,  # PPO clip ratio
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio

    def train_step(self, prompts):
        """Single PPO training step."""
        # Generate responses
        with torch.no_grad():
            responses = self.policy.generate(prompts)

        # Get rewards from reward model
        rewards = self.reward_model(prompts + responses)

        # Calculate KL divergence from reference model
        policy_logprobs = self.policy.compute_logprobs(prompts, responses)
        ref_logprobs = self.ref_model.compute_logprobs(prompts, responses)
        kl_div = policy_logprobs - ref_logprobs

        # Total reward (reward - KL penalty)
        total_rewards = rewards - self.kl_coef * kl_div

        # PPO update
        loss = self.ppo_loss(
            prompts,
            responses,
            total_rewards,
            policy_logprobs
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kl": kl_div.mean().item()
        }

    def ppo_loss(self, prompts, responses, rewards, old_logprobs):
        """Calculate PPO clipped objective."""
        # Get current logprobs
        new_logprobs = self.policy.compute_logprobs(prompts, responses)

        # Importance ratio
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clipped objective
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_ratio,
            1 + self.clip_ratio
        )

        # Take minimum (pessimistic bound)
        loss = -torch.min(
            ratio * rewards,
            clipped_ratio * rewards
        ).mean()

        return loss
```

### Practical RLHF Pipeline

```python
class RLHFPipeline:
    """Complete RLHF training pipeline."""

    def __init__(self, base_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def stage1_supervised_finetuning(self, demonstration_data, epochs=3):
        """Stage 1: SFT on high-quality demonstrations."""
        print("Stage 1: Supervised Fine-Tuning")

        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            for prompt, response in demonstration_data:
                inputs = self.tokenizer(
                    prompt + response,
                    return_tensors="pt"
                )

                outputs = self.base_model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("✓ SFT complete")
        return self.base_model

    def stage2_reward_modeling(self, comparison_data, epochs=3):
        """Stage 2: Train reward model on preferences."""
        print("Stage 2: Reward Model Training")

        reward_model = RewardModel(self.base_model)
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

        reward_model = train_reward_model(
            reward_model,
            comparison_data,
            optimizer,
            epochs
        )

        print("✓ Reward model trained")
        return reward_model

    def stage3_ppo_training(self, reward_model, prompts, epochs=10):
        """Stage 3: PPO fine-tuning."""
        print("Stage 3: PPO Training")

        # Create reference model (frozen copy)
        ref_model = copy.deepcopy(self.base_model)
        ref_model.eval()

        # Setup PPO trainer
        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=1e-6)
        ppo_trainer = PPOTrainer(
            policy_model=self.base_model,
            reward_model=reward_model,
            ref_model=ref_model,
            optimizer=optimizer
        )

        # Training loop
        for epoch in range(epochs):
            metrics = ppo_trainer.train_step(prompts)
            print(f"Epoch {epoch+1}: {metrics}")

        print("✓ PPO training complete")
        return self.base_model

    def run_full_pipeline(
        self,
        demonstration_data,
        comparison_data,
        ppo_prompts
    ):
        """Run complete RLHF pipeline."""
        # Stage 1: SFT
        self.stage1_supervised_finetuning(demonstration_data)

        # Stage 2: Reward modeling
        reward_model = self.stage2_reward_modeling(comparison_data)

        # Stage 3: PPO
        final_model = self.stage3_ppo_training(reward_model, ppo_prompts)

        return final_model
```

---

## Compliance and Governance

### Model Card

```python
class ModelCard:
    """Document model capabilities, limitations, and ethical considerations."""

    def __init__(self):
        self.metadata = {}

    def create(
        self,
        model_name,
        version,
        intended_use,
        limitations,
        fairness_assessment,
        training_data,
        evaluation_metrics
    ):
        """Create comprehensive model card."""
        return {
            "model_details": {
                "name": model_name,
                "version": version,
                "date": datetime.now().isoformat(),
                "type": "Classification/Generation/etc.",
                "architecture": "Transformer-based",
            },
            "intended_use": {
                "primary_use": intended_use,
                "out_of_scope": self._list_out_of_scope_uses(),
            },
            "factors": {
                "relevant_factors": [
                    "User demographics",
                    "Language",
                    "Context of use"
                ],
                "evaluation_factors": fairness_assessment
            },
            "metrics": evaluation_metrics,
            "training_data": {
                "description": training_data,
                "preprocessing": self._describe_preprocessing(),
            },
            "ethical_considerations": {
                "risks": self._identify_risks(),
                "mitigations": self._describe_mitigations(),
            },
            "limitations": limitations,
            "recommendations": self._provide_recommendations()
        }

    def _list_out_of_scope_uses(self):
        return [
            "Medical diagnosis",
            "Legal advice",
            "High-stakes decision making without human oversight",
            "Use with vulnerable populations without safeguards"
        ]

    def _identify_risks(self):
        return [
            "Potential for biased outputs",
            "May generate plausible but incorrect information",
            "Privacy concerns with input data",
            "Misuse for harmful content generation"
        ]

    def _describe_mitigations(self):
        return [
            "Fairness metrics monitored continuously",
            "Content filtering applied to outputs",
            "PII detection and redaction",
            "Human review for high-stakes applications",
            "Regular audits and updates"
        ]
```

---

## Best Practices Summary

1. **Ethics First**: Consider ethical implications from the start, not as an afterthought

2. **Measure Fairness**: Use multiple fairness metrics, understand trade-offs

3. **Be Transparent**: Document limitations, provide explanations, use model cards

4. **Layer Safety**: Multiple independent safety checks are better than one

5. **Continuous Monitoring**: Regularly audit for bias, safety issues, and drift

6. **Human Oversight**: Keep humans in the loop for high-stakes decisions

7. **Privacy by Design**: Build privacy protection into the system architecture

8. **Stay Informed**: Follow AI ethics research and regulatory developments

---

## Resources

- [Fairlearn Documentation](https://fairlearn.org/)
- [IBM AI Fairness 360](https://aif360.mybluemix.net/)
- [Anthropic Safety Research](https://www.anthropic.com/safety)
- [RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Model Cards Toolkit](https://github.com/tensorflow/model-card-toolkit)

---

**Next**: [Project Implementation](../projects/README.md)
