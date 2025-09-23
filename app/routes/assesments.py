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
def create_assesment():
    payload = request.get_json()
    if not payload or not payload.get("query") or not payload.get("perawat"):
        return jsonify({"status": 400, "message": "Fields required: query, patient_id, perawat"}), 400

    # Cek apakah FAISS index sudah ada
    if FAISS_INDEX_FILE and not os.path.exists(FAISS_INDEX_FILE):
        return jsonify({"status": 210, "message": "belum mengupload data assesmen"}), 400

    query = payload["query"]
    perawat = payload["perawat"]
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")

    # --- Bagian Retrieval ---
    retrieved_data = []
    try:
        initialize_faiss_index()

        if FAISS_INDEX.ntotal > 0: 
            k = 3  # ambil 3 terdekat
            D, I = FAISS_INDEX.search(query_vector, k)

            for idx in I[0]:
                if idx != -1:
                    asses = Assesment.query.get(int(idx))
                    if asses:
                        retrieved_data.append(asses.data)
    except Exception as e:
        return jsonify({"status": 500, "message": f"FAISS retrieval failed: {str(e)}"}), 500

    context = "\n".join(retrieved_data) if retrieved_data else "Tidak ada data relevan yang ditemukan."

    try:
        prompt = f"""
        Berikut adalah konteks dari data assesmen sebelumnya:
        {context}

        Sekarang, susun JSON terstruktur yang mengisi semua field ASESMEN AWAL KEPERAWATAN RAWAT INAP.
        Tambahkan field: alamat, pekerjaan, status_perkawinan, penanggung_jawab, hubungan_penanggung_jawab, 
        kontak_penanggung_jawab. Masukkan ke bagian informasi umum.
        Jika field agama sudah ada di tempat lain maka tambahkan saja berdasarkan teks berikut.

        Query:
        {query}
        """

        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": "Anda adalah asisten medis yang menyusun data asesmen."},
                {"role": "user", "content": prompt},
            ],
        )
        ai_json = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}"}), 500

    # --- Simpan ke DB ---
    new_assesment = Assesment(
        perawat=perawat,
        tanggal=datetime.utcnow(),
        data=ai_json,
    )
    db.session.add(new_assesment)
    db.session.commit()

    # --- Update FAISS Index ---
    try:
        FAISS_INDEX.add_with_ids(query_vector, np.array([new_assesment.id]))
        save_faiss_index()
    except Exception as e:
        return jsonify({"status": 500, "message": f"FAISS update failed: {str(e)}"}), 500

    return jsonify(
        {
            "status": 201,
            "message": "Assesment created successfully",
            "data": {
                "id": new_assesment.id,
                "perawat": perawat,
                "tanggal": new_assesment.tanggal.isoformat(),
                "data": ai_json,
            },
        }
    ), 201



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
