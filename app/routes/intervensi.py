from zoneinfo import ZoneInfo
from flask import Blueprint, request, jsonify
from app.model import db, Intervensi, Patient
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import pickle
import re

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"

def search_siki(query, id_to_text, k=3, index=None):
    """Cari intervensi SIKI dari FAISS berdasarkan input query."""
    if index is None or not id_to_text:
        return []

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_text[i] for i in I[0] if i != -1 and i in id_to_text]


# --- Blueprint ---
intervensi_bp = Blueprint("intervensi_bp", __name__, url_prefix="/intervensi")


@intervensi_bp.route("/", methods=["GET"])
def get_intervensis():
    intervensis = Intervensi.query.all()
    data = [
        {
            "id": i.id,
            "patient_id": i.patient_id,
            "tanggal": i.tanggal.isoformat(),
            "user_id": i.user_id,
            "implementasi": i.implementasi,
            "evaluasi": i.evaluasi,
        }
        for i in intervensis
    ]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@intervensi_bp.route("/<int:intervensi_id>", methods=["GET"])
def get_intervensi(intervensi_id):
    intervensi = Intervensi.query.get(intervensi_id)
    if not intervensi:
        return jsonify({"status": 404, "message": "Intervensi not found", "data": None}), 404

    data = {
        "id": intervensi.id,
        "patient_id": intervensi.patient_id,
        "tanggal": intervensi.tanggal.isoformat(),
        "user_id": intervensi.user_id,
        "implementasi": intervensi.implementasi,
        "evaluasi": intervensi.evaluasi,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@intervensi_bp.route("/", methods=["POST"])
def create_intervensi():
    payload = request.get_json()

    if not payload or not payload.get("implementasi") or not payload.get("evaluasi") or not payload.get("patient_id") or not payload.get("user_id"):
        return jsonify({"status": 400, "message": "Fields required: implementasi, evaluasi, patient_id, user_id", "data": None}), 400

    patient_id = payload["patient_id"]
    user_id = payload["user_id"]
    implementasi = payload["implementasi"]
    evaluasi = payload["evaluasi"]

    # cek pasien
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    # buat intervensi baru
    new_intervensi = Intervensi(
        tanggal=datetime.now(ZoneInfo("Asia/Makassar")),
        implementasi=implementasi,
        evaluasi=evaluasi,
        patient_id=patient_id,
        user_id=user_id,
    )

    db.session.add(new_intervensi)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Intervensi created successfully",
        "data": {
            "id": new_intervensi.id,
            "tanggal": new_intervensi.tanggal.isoformat(),
            "implementasi": new_intervensi.implementasi,
            "evaluasi": new_intervensi.evaluasi,
            "patient_id": new_intervensi.patient_id,
            "user_id": new_intervensi.user_id,
        }
    }), 201



@intervensi_bp.route("/<int:intervensi_id>", methods=["PUT"])
def update_intervensi(intervensi_id):
    intervensi = Intervensi.query.get(intervensi_id)
    if not intervensi:
        return jsonify({"status": 404, "message": "Intervensi not found", "data": None}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "No data provided", "data": None}), 400

    for field in ["user_id", "implementasi", "evaluasi"]:
        if field in payload:
            setattr(intervensi, field, payload[field])

    db.session.commit()
    return jsonify({"status": 200, "message": "Intervensi updated", "data": {"id": intervensi.id}}), 200


@intervensi_bp.route("/<int:intervensi_id>", methods=["DELETE"])
def delete_intervensi(intervensi_id):
    intervensi = Intervensi.query.get(intervensi_id)
    if not intervensi:
        return jsonify({"status": 404, "message": "Intervensi tidak ditemukan", "data": None}), 404

    try:
        db.session.delete(intervensi)
        db.session.commit()
        return jsonify({"status": 200, "message": "Intervensi berhasil dihapus", "data": {"id": intervensi_id}}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Gagal menghapus Intervensi: {str(e)}", "data": None}), 500


@intervensi_bp.route("/search", methods=["POST"])
def search_intervensi():
    payload = request.get_json()
    query_string = payload.get("query") if payload else None

    if not query_string:
        return jsonify({"status": 400, "message": "Missing field: query", "data": None}), 400

    results = Intervensi.query.filter(
        (Intervensi.implementasi.ilike(f"%{query_string}%")) |
        (Intervensi.evaluasi.ilike(f"%{query_string}%"))
    ).all()

    data = [
        {
            "id": i.id,
            "patient_id": i.patient_id,
            "tanggal": i.tanggal.isoformat(),
            "user_id": i.user_id,
            "implementasi": i.implementasi,
            "evaluasi": i.evaluasi,
        }
        for i in results
    ]

    return jsonify({"status": 200, "message": "Success", "data": data}), 200
