import os
import uuid
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')


# ── Connection ───────────────────────────────────────────────────────────────
def get_db():
    """Returns the loan_approval database instance."""
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "loan_approval")

    if not uri:
        raise ValueError("MONGODB_URI not found in .env file")

    client = MongoClient(uri)
    return client[db_name]


# ── Create ───────────────────────────────────────────────────────────────────
def save_application(
    applicant: dict,
    prediction: dict,
    email_result: dict,
    guardrail_result: dict,
    bias_result: dict,
    nbo_result: dict,
    final_status: str
) -> str:
    """
    Saves a complete loan application journey to MongoDB.
    Returns the generated application_id.
    """
    db = get_db()
    application_id = str(uuid.uuid4())

    # Clean applicant dict — remove internal fields before storing
    applicant_clean = {
        k: v for k, v in applicant.items()
        if k not in ['loan_status']
    }

    document = {
        "application_id":   application_id,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0",
        "applicant":        applicant_clean,
        "ml_prediction": {
            "decision":      prediction.get("decision"),
            "approval_prob": prediction.get("approval_prob"),
            "default_prob":  prediction.get("default_prob"),
            "confidence":    prediction.get("confidence"),
            "risk_tier":     prediction.get("risk_tier"),
            "top_factors":   prediction.get("top_factors"),
        },
        "generated_email": {
            "body":         email_result.get("email_body"),
            "tokens_used":  email_result.get("tokens_used"),
        },
        "guardrail_result": {
            "safe_to_send":    guardrail_result.get("safe_to_send"),
            "issues_found":    guardrail_result.get("issues_found", []),
            "severity":        guardrail_result.get("severity"),
            "recommendation":  guardrail_result.get("recommendation"),
        },
        "bias_check": {
            "bias_detected":    bias_result.get("bias_detected"),
            "disparate_impact": bias_result.get("disparate_impact"),
            "approval_rates":   bias_result.get("approval_rates"),
            "action":           bias_result.get("action"),
        },
        "next_best_offer": {
            "triggered": nbo_result.get("triggered", False),
            "offer":     nbo_result.get("offer") if nbo_result.get("triggered") else None,
        },
        "final_status": final_status,
    }

    db.applications.insert_one(document)
    return application_id


# ── Read ─────────────────────────────────────────────────────────────────────
def get_application(application_id: str) -> dict:
    """Fetch a single application by ID."""
    db = get_db()
    result = db.applications.find_one(
        {"application_id": application_id},
        {"_id": 0}   # exclude MongoDB internal _id field
    )
    return result


def get_recent_applications(limit: int = 10) -> list:
    """Fetch most recent applications."""
    db = get_db()
    results = db.applications.find(
        {},
        {"_id": 0}
    ).sort("timestamp", DESCENDING).limit(limit)
    return list(results)


def get_applications_by_status(status: str) -> list:
    """Fetch all applications with a specific final_status."""
    db = get_db()
    results = db.applications.find(
        {"final_status": status},
        {"_id": 0}
    ).sort("timestamp", DESCENDING)
    return list(results)


def get_flagged_applications() -> list:
    """Fetch all applications flagged for human review."""
    db = get_db()
    results = db.applications.find(
        {"final_status": {"$in": ["FLAGGED_BIAS", "FLAGGED_GUARDRAIL", "HUMAN_REVIEW"]}},
        {"_id": 0}
    ).sort("timestamp", DESCENDING)
    return list(results)


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_approval_stats() -> dict:
    """Returns overall approval/denial stats from the database."""
    db = get_db()
    total       = db.applications.count_documents({})
    approved    = db.applications.count_documents({"ml_prediction.decision": "APPROVED"})
    denied      = db.applications.count_documents({"ml_prediction.decision": "DENIED"})
    flagged     = db.applications.count_documents({"guardrail_result.safe_to_send": False})
    bias_flags  = db.applications.count_documents({"bias_check.bias_detected": True})

    return {
        "total_applications": total,
        "approved":           approved,
        "denied":             denied,
        "approval_rate":      round(approved / total, 4) if total > 0 else 0,
        "flagged_by_guardrail": flagged,
        "bias_flags":         bias_flags,
    }


# ── Indexes ───────────────────────────────────────────────────────────────────
def create_indexes():
    """Create indexes for common query patterns. Run once on setup."""
    db = get_db()
    db.applications.create_index("application_id", unique=True)
    db.applications.create_index("timestamp")
    db.applications.create_index("final_status")
    db.applications.create_index("ml_prediction.decision")
    print("Indexes created successfully.")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing MongoDB connection...")
    db = get_db()
    db.client.admin.command('ping')
    print("Connection successful.")

    create_indexes()

    # Insert a test document
    test_id = save_application(
        applicant        = {"name": "Test User", "loan_amnt": 5000},
        prediction       = {"decision": "APPROVED", "approval_prob": 0.95,
                            "default_prob": 0.05, "confidence": "95%",
                            "risk_tier": "Low Risk", "top_factors": ["credit_score"]},
        email_result     = {"email_body": "Test email body", "tokens_used": 100},
        guardrail_result = {"safe_to_send": True, "issues_found": [],
                            "severity": "none", "recommendation": "SEND"},
        bias_result      = {"bias_detected": False, "disparate_impact": 0.95,
                            "approval_rates": {"male": 0.80, "female": 0.76},
                            "action": "NONE"},
        nbo_result       = {"triggered": False},
        final_status     = "EMAIL_SENT"
    )
    print(f"Test document saved with ID: {test_id}")

    # Fetch it back
    doc = get_application(test_id)
    print(f"Fetched back: {doc['applicant']['name']} — {doc['final_status']}")

    # Stats
    stats = get_approval_stats()
    print(f"DB Stats: {stats}")