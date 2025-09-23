from flask import Blueprint, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from app.model import db, Laporan, CPPT, Patient
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from flask_jwt_extended import jwt_required
from app.utils import role_required

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

EMBEDDING_DIM = 384
siki_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SIKI_FILE = "app/siki-slki-sdki/siki-slki-sdki.faiss"
PERMENKES_FILE = "app/permenkes/permenkes.faiss"

FAISS_SIKI = None
FAISS_PERMENKES = None


def load_faiss_index(path):
    if os.path.exists(path):
        return faiss.read_index(path)
    return faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))


def initialize_indexes():
    global FAISS_SIKI, FAISS_PERMENKES
    if FAISS_SIKI is None:
        FAISS_SIKI = load_faiss_index(SIKI_FILE)
    if FAISS_PERMENKES is None:
        FAISS_PERMENKES = load_faiss_index(PERMENKES_FILE)


def search_index(index, query_vector, k=3):
    """Cari embedding terdekat di FAISS index"""
    if index.ntotal == 0:
        return []
    distances, indices = index.search(query_vector, k)
    return [{"id": int(idx), "score": float(1 / (1 + dist))}
            for idx, dist in zip(indices[0], distances[0]) if idx != -1]


@laporan_bp.route("/", methods=["GET"])
def get_laporans():
    laporans = Laporan.query.all()
    data = [
        {
            "id": l.id,
            "patient_id": l.patient_id,
            "cppt_id": l.cppt_id,
            "tanggal": l.tanggal.isoformat(),
            "user_id": l.user_id,
            "subjective": l.subjective,
            "objective": l.objective,
            "assessment": l.assessment,
            "plan": l.plan,
            "keterangan": l.keterangan,
            "dokter": l.dokter,
            "signature": l.signature,
            "tindakan_lanjutan": l.tindakan_lanjutan,
            "SDKI": l.SDKI,
            "SLKI": l.SLKI,
            "SIKI": l.SIKI,
        }
        for l in laporans
    ]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@laporan_bp.route("/", methods=["POST"])
@jwt_required()
@role_required("admin", "user")
def create_laporan():
    payload = request.get_json()
    if not payload or not payload.get("query") or not payload.get("cppt_id") or not payload.get("patient_id") or not payload.get("perawat_id"):
        return jsonify({"status": 400, "message": "Fields required: query, cppt_id, patient_id, perawat_id", "data": None}), 400

    query = payload["query"]
    cppt_id = payload["cppt_id"]
    patient_id = payload["patient_id"]
    perawat_id = payload["perawat_id"]

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404
    cppt = CPPT.query.get(cppt_id)
    if not cppt:
        return jsonify({"status": 404, "message": "CPPT not found", "data": None}), 404

    initialize_indexes()
    query_vector = siki_model.encode([query], convert_to_numpy=True).astype("float32")

    siki_refs = search_index(FAISS_SIKI, query_vector, k=5)
    permenkes_refs = search_index(FAISS_PERMENKES, query_vector, k=5)

    references = {
        "siki": siki_refs,
        "permenkes": permenkes_refs
    }

    try:
        import json
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    Kamu adalah asisten medis.
                    Buat JSON dengan field:
                    - tindakan_lanjutan
                    - SDKI 
                    - SLKI
                    - SIKI

                    Gunakan referensi standar dari PERMENKES & SIKI-SLKI-SDKI berikut:
                    {json.dumps(references, indent=2)}

                    Jawab hanya dengan JSON valid.
                    """,
                },
                {"role": "user", "content": query},
            ],
        )
        ai_json = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}", "data": None}), 500

    try:
        parsed = ai_json
        if isinstance(parsed, str):
            import json
            if parsed.startswith("```json"):
                parsed = parsed[len("```json"):].strip()
            if parsed.endswith("```"):
                parsed = parsed[:-3].strip()
            parsed = json.loads(parsed)
    except Exception:
        parsed = {"tindakan_lanjutan": None, "SLKI": None, "SIKI": None}

    new_laporan = Laporan(
        patient_id=patient_id,
        cppt_id=cppt_id,
        user_id=perawat_id,
        tanggal=datetime.utcnow(),
        subjective=cppt.subjective,
        objective=cppt.objective,
        assessment=cppt.assessment,
        plan=cppt.plan,
        keterangan=cppt.keterangan,
        dokter=cppt.dokter,
        signature=cppt.signature,
        tindakan_lanjutan=parsed.get("tindakan_lanjutan"),
        SDKI=parsed.get("SDKI"),
        SLKI=parsed.get("SLKI"),
        SIKI=parsed.get("SIKI"),
    )
    db.session.add(new_laporan)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Laporan created successfully",
        "data": {
            "id": new_laporan.id,
            "tindakan_lanjutan": new_laporan.tindakan_lanjutan,
            "SDKI": new_laporan.SDKI,
            "SLKI": new_laporan.SLKI,
            "SIKI": new_laporan.SIKI,
        }
    }), 201

@laporan_bp.route("/search", methods=["POST"])
def search_laporan():
    payload = request.get_json()
    if not payload or not payload.get("query"):
        return jsonify({"status": 400, "message": "Field required: query", "data": None}), 400

    query = payload["query"]

    initialize_indexes()
    query_vector = siki_model.encode([query], convert_to_numpy=True).astype("float32")

    siki_refs = search_index(FAISS_SIKI, query_vector, k=5)
    permenkes_refs = search_index(FAISS_PERMENKES, query_vector, k=5)

    references = {
        "siki": siki_refs,
        "permenkes": permenkes_refs
    }

    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system",
                    "content": f"""
Kamu adalah asisten medis.
Buat JSON dengan field:
- tindakan_lanjutan
- SDKI
- SLKI
- SIKI

Gunakan referensi standar dari PERMENKES & SIKI-SLKI-SDKI berikut:
{json.dumps(references, indent=2)}

Jawab hanya dengan JSON valid.
""",
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
        parsed = {"tindakan_lanjutan": None, "SLKI": None, "SIKI": None}

    return jsonify({
        "status": 200,
        "message": "Search result generated",
        "data": {
            "query": query,
            "tindakan_lanjutan": parsed.get("tindakan_lanjutan"),
            "SDKI": parsed.get("SDKI"),
            "SLKI": parsed.get("SLKI"),
            "SIKI": parsed.get("SIKI"),
        }
    }), 200

@laporan_bp.route("/<int:id>", methods=["GET"])
def get_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    data = {
        "id": laporan.id,
        "patient_id": laporan.patient_id,
        "cppt_id": laporan.cppt_id,
        "tanggal": laporan.tanggal.isoformat(),
        "user_id": laporan.user_id,
        "subjective": laporan.subjective,
        "objective": laporan.objective,
        "assessment": laporan.assessment,
        "plan": laporan.plan,
        "keterangan": laporan.keterangan,
        "dokter": laporan.dokter,
        "signature": laporan.signature,
        "tindakan_lanjutan": laporan.tindakan_lanjutan,
        "SDKI": laporan.SLKI,
        "SLKI": laporan.SLKI,
        "SIKI": laporan.SIKI,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@laporan_bp.route("/<int:id>", methods=["PUT"])
@jwt_required()
@role_required("admin", "user")
def update_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "Request body required", "data": None}), 400

    for field in [
        "subjective", "objective", "assessment", "plan", "keterangan",
        "dokter", "signature", "tindakan_lanjutan", "SLKI", "SIKI"
    ]:
        if field in payload:
            setattr(laporan, field, payload[field])

    db.session.commit()

    return jsonify({"status": 200, "message": "Laporan updated successfully", "data": {"id": laporan.id}}), 200


@laporan_bp.route("/<int:id>", methods=["DELETE"])
@jwt_required()
@role_required("admin", "user")
def delete_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    db.session.delete(laporan)
    db.session.commit()

    return jsonify({"status": 200, "message": "Laporan deleted successfully", "data": {"id": id}}), 200
