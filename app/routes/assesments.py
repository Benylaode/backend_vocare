from flask import Blueprint, request, jsonify
from app.model import db, Assesment, Patient
import faiss
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

assesment_bp = Blueprint("assesment_bp", __name__, url_prefix="/assesments")

FAISS_INDEX = None
EMBEDDING_DIM = 384
FAISS_INDEX_FILE = "app/faisses/assesment/assesmen.faiss"


def initialize_faiss_index():
    """Load FAISS index from file or create new one if not exist"""
    global FAISS_INDEX
    if FAISS_INDEX is None:
        if os.path.exists(FAISS_INDEX_FILE):
            FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
        else:
            FAISS_INDEX = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))


def save_faiss_index():
    """Save FAISS index to file"""
    if FAISS_INDEX is not None:
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_FILE)


@assesment_bp.route("/", methods=["GET"])
def get_assesments():
    assesments = Assesment.query.all()
    data = [
        {
            "id": a.id,
            "tanggal": a.tanggal.isoformat(),
            "perawat": a.perawat,
            "data": a.data,
        }
        for a in assesments
    ]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@assesment_bp.route("/<int:assesment_id>", methods=["GET"])
def get_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404
    try:
        data = assesment.data
        if isinstance(data, str):
            import json
            if data.startswith("```json"):
                data = data[len("```json"):].strip()
            if data.endswith("```"):
                data = data[:-3].strip()
            json_ku = json.loads(data)
    except Exception:
        return jsonify({"status": 500, "message": "Invalid JSON in assesment", "data": data}), 500

    data = {
        "id": assesment.id,
        "tanggal": assesment.tanggal.isoformat(),
        "perawat": assesment.perawat,
        "data": json_ku,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@assesment_bp.route("/", methods=["POST"])
def create_patient():
    payload = request.get_json()

    if not payload or not payload.get("id_assesment") or not payload.get("nama"):
        return jsonify({"status": 400, "message": "Fields required: id_assesment, nama"}), 400

    id_assesment = payload["id_assesment"]
    nama = payload["nama"]

    assesment = Assesment.query.filter_by(id=id_assesment).first()
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    try:
        data = assesment.data
        if isinstance(data, str):
            import json
            if data.startswith("```json"):
                data = data[len("```json"):].strip()
            if data.endswith("```"):
                data = data[:-3].strip()
            data = json.loads(data)
    except Exception:
        return jsonify({"status": 500, "message": "Invalid JSON in assesment"}), 500

    if Patient.query.filter_by(assesment_id=id_assesment).first():
        return jsonify({"status": 400, "message": "Patient with this id_assesment already exists"}), 400

    info_umum = data.get("asesmen_awal_keperawatan", {}).get("informasi_umum", {})

    nama_pasien = info_umum.get("nama") or info_umum.get("nama_pasien") or nama

    no_rm = info_umum.get("no_rm") or info_umum.get("kode_rm")

    tgl_lahir = info_umum.get("tanggal_lahir")
    if tgl_lahir and isinstance(tgl_lahir, str):
        try:
            from datetime import datetime
            tgl_lahir = datetime.strptime(tgl_lahir, "%d %B %Y").date()
        except Exception:
            tgl_lahir = None

    pj = info_umum.get("penanggung_jawab")
    if isinstance(pj, dict):
        nama_pj = pj.get("nama")
        hubungan_pj = pj.get("hubungan")
        kontak_pj = pj.get("kontak")
    else:
        nama_pj = pj
        hubungan_pj = info_umum.get("hubungan_penanggung_jawab")
        kontak_pj = info_umum.get("kontak_penanggung_jawab")

    new_patient = Patient(
        assesment_id=id_assesment,
        no_rekam_medis=no_rm,
        nama=nama_pasien,
        tgl_lahir=tgl_lahir,
        jenis_kelamin=info_umum.get("jenis_kelamin"),
        alamat=info_umum.get("alamat"),
        agama=info_umum.get("agama"),
        pekerjaan=info_umum.get("pekerjaan"),
        status_perkawinan=info_umum.get("status_perkawinan"),
        penanggung_jawab=nama_pj,
        hubungan_penanggung_jawab=hubungan_pj,
        kontak_penanggung_jawab=kontak_pj,
        status_rawat="rawat_inap"  # default untuk pasien baru
    )

    db.session.add(new_patient)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Patient created from assesment",
        "data": {"id": new_patient.id}
    }), 201




@assesment_bp.route("/<int:assesment_id>", methods=["PUT"])
def update_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "No data provided"}), 400

    if "perawat" in payload:
        assesment.perawat = payload["perawat"]

    if "data" in payload:
        assesment.data = payload["data"]

    db.session.commit()

    return jsonify({"status": 200, "message": "Assesment updated", "data": {"id": assesment.id}}), 200


@assesment_bp.route("/<int:assesment_id>", methods=["DELETE"])
def delete_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    db.session.delete(assesment)
    db.session.commit()


    return jsonify({"status": 200, "message": "Assesment deleted"}), 200


@assesment_bp.route("/search", methods=["POST"])
def search_assesments():
    payload = request.get_json()
    query_string = payload.get("query") if payload else None

    if not query_string:
        return jsonify({"status": 400, "message": "Missing field: query"}), 400

    initialize_faiss_index()
    if FAISS_INDEX.ntotal == 0:
        return jsonify({"status": 200, "message": "No assessments in index", "data": []}), 200

    query_vector = model.encode([query_string], convert_to_numpy=True).astype("float32")

    k = 10
    distances, indices = FAISS_INDEX.search(np.array([query_vector]), k)

    results = []
    for i, assesment_id in enumerate(indices[0]):
        assesment = Assesment.query.get(int(assesment_id))
        if assesment:
            results.append(
                {
                    "id": assesment.id,
                    "patient_id": assesment.patient_id,
                    "tanggal": assesment.tanggal.isoformat(),
                    "perawat": assesment.perawat,
                    "data": assesment.data,
                    "relevance_score": float(1 / (1 + distances[0][i])),
                }
            )

    return jsonify({"status": 200, "message": "Success", "data": results}), 200
