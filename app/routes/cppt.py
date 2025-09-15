from flask import Blueprint, request, jsonify
from app.model import db, CPPT, Patient
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

cppt_bp = Blueprint("cppt_bp", __name__, url_prefix="/cppt")

@cppt_bp.route("/", methods=["GET"])

def get_cppts():
    cppts = CPPT.query.all()
    data = [
        {
            "id": c.id,
            "patient_id": c.patient_id,
            "tanggal": c.tanggal.isoformat(),
            "user_id": c.user_id,
            "subjective": c.subjective,
            "objective": c.objective,
            "assessment": c.assessment,
            "plan": c.plan,
            "keterangan": c.keterangan,
            "dokter": c.dokter,
            "signature": c.signature,
        }
        for c in cppts
    ]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@cppt_bp.route("/<int:cppt_id>", methods=["GET"])
def get_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt:
        return jsonify({"status": 404, "message": "CPPT not found", "data": None}), 404

    data = {
        "id": cppt.id,
        "patient_id": cppt.patient_id,
        "tanggal": cppt.tanggal.isoformat(),
        "user_id": cppt.user_id,
        "subjective": cppt.subjective,
        "objective": cppt.objective,
        "assessment": cppt.assessment,
        "plan": cppt.plan,
        "keterangan": cppt.keterangan,
        "dokter": cppt.dokter,
        "signature": cppt.signature,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@cppt_bp.route("/", methods=["POST"])

def create_cppt():
    payload = request.get_json()

    if not payload or not payload.get("query") or not payload.get("patient_id") or not payload.get("perawat_id"):
        return jsonify({"status": 400, "message": "Fields required: query, patient_id, perawat_id", "data": None}), 400

    query = payload["query"]
    patient_id = payload["patient_id"]
    perawat_id = payload["perawat_id"]

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system",
                    "content": """Kamu adalah asisten medis. Rapikan teks mentah menjadi JSON sesuai struktur CPPT.
Isi JSON dengan field berikut:
- subjective
- objective
- assessment
- plan
- keterangan
Jika ada dokter atau tanda tangan sebutkan juga, jika tidak biarkan kosong.
Jangan ubah makna medis, hanya rapikan struktur.""",
                },
                {"role": "user", "content": query},
            ],
        )
        ai_json = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}", "data": None}), 500

    try:
        parsed = json.loads(ai_json)
    except Exception:
        parsed = {
            "subjective": None,
            "objective": None,
            "assessment": None,
            "plan": None,
            "keterangan": ai_json,
            "dokter": None,
            "signature": None,
        }

    new_cppt = CPPT(
        patient_id=patient_id,
        user_id=perawat_id,
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective"),
        objective=parsed.get("objective"),
        assessment=parsed.get("assessment"),
        plan=parsed.get("plan"),
        keterangan=parsed.get("keterangan"),
        dokter=parsed.get("dokter"),
        signature=parsed.get("signature"),
    )
    db.session.add(new_cppt)
    db.session.commit()

    return jsonify(
        {
            "status": 201,
            "message": "CPPT created successfully",
            "data": {
                "id": new_cppt.id,
                "patient_id": patient_id,
                "user_id": perawat_id,
                "tanggal": new_cppt.tanggal.isoformat(),
                "subjective": new_cppt.subjective,
                "objective": new_cppt.objective,
                "assessment": new_cppt.assessment,
                "plan": new_cppt.plan,
                "keterangan": new_cppt.keterangan,
                "dokter": new_cppt.dokter,
                "signature": new_cppt.signature,
            },
        }
    ), 201


@cppt_bp.route("/<int:cppt_id>", methods=["PUT"])
def update_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt:
        return jsonify({"status": 404, "message": "CPPT not found", "data": None}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "No data provided", "data": None}), 400

    for field in [
        "user_id",
        "subjective",
        "objective",
        "assessment",
        "plan",
        "keterangan",
        "dokter",
        "signature",
    ]:
        if field in payload:
            setattr(cppt, field, payload[field])

    db.session.commit()
    return jsonify({"status": 200, "message": "CPPT updated", "data": {"id": cppt.id}}), 200


@cppt_bp.route("/<int:cppt_id>", methods=["DELETE"])
def delete_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt:
        return jsonify({"status": 404, "message": "CPPT not found", "data": None}), 404

    db.session.delete(cppt)
    db.session.commit()

    return jsonify({"status": 200, "message": "CPPT deleted", "data": {"id": cppt.id}}), 200


@cppt_bp.route("/search", methods=["POST"])
def search_cppts():
    payload = request.get_json()
    query_string = payload.get("query") if payload else None

    if not query_string:
        return jsonify({"status": 400, "message": "Missing field: query", "data": None}), 400

    results = CPPT.query.filter(
        (CPPT.subjective.ilike(f"%{query_string}%")) |
        (CPPT.objective.ilike(f"%{query_string}%")) |
        (CPPT.assessment.ilike(f"%{query_string}%")) |
        (CPPT.plan.ilike(f"%{query_string}%")) |
        (CPPT.keterangan.ilike(f"%{query_string}%"))
    ).all()

    data = [
        {
            "id": cppt.id,
            "patient_id": cppt.patient_id,
            "tanggal": cppt.tanggal.isoformat(),
            "user_id": cppt.user_id,
            "subjective": cppt.subjective,
            "objective": cppt.objective,
            "assessment": cppt.assessment,
            "plan": cppt.plan,
            "keterangan": cppt.keterangan,
            "dokter": cppt.dokter,
            "signature": cppt.signature,
        }
        for cppt in results
    ]

    return jsonify({"status": 200, "message": "Success", "data": data}), 200
