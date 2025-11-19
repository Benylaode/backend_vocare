from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, Intervensi, CPPT
import os, json, pickle
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# --- Load Embedding Model ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- File FAISS & Mapping ---
FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"

def search_with_faiss(query, index, id_to_text, k=3):
    """Search di FAISS dan balikan teks mapping."""
    if index is None or not id_to_text:
        return []
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_text[i] for i in I[0] if i != -1 and i in id_to_text]

# --- Blueprint ---
laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

@laporan_bp.route("/", methods=["GET"])
def get_laporans():
    laporans = Laporan.query.all()
    data = [
        {
            "id": l.id,
            "patient_id": l.patient_id,
            "cppt_id": l.cppt_id,
            "intervensi_id": l.intervensi_id,
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


# ================== CREATE ==================
@laporan_bp.route("/", methods=["POST"])
def create_laporan():
    payload = request.get_json()

    if not payload or not payload.get("query") or not payload.get("patient_id") or not payload.get("perawat_id") or not payload.get("intevensi_id"):
        return jsonify({"status": 400, "message": "Fields required: query, patient_id, perawat_id, intevensi_id", "data": None}), 400

    query = payload["query"] + ", "
    patient_id = payload["patient_id"]
    perawat_id = payload["perawat_id"]
    cppt_id = payload["cppt_id"]
    intervensi_id = payload["intevensi_id"]

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404

    cppt = CPPT.query.get(cppt_id)
    query = query + f", assesment : {cppt.assessment}, plan : {cppt.plan}"
    if not cppt:
        return jsonify({"status": 404, "message": "Intervensi not found", "data": None}), 404

    # cek file FAISS
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "rb") as f:
            id_to_text = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        matches = search_with_faiss(query, index=index, id_to_text=id_to_text, k=5)
    else:
        return jsonify({"status": 210, "message": "belum mengupload data sdki-siki-slki", "data": None}), 404

    context_text = "\n\n".join(matches)

    try:
        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "system",
                    "content": """Kamu adalah asisten medis yang membantu menyusun Laporan Keperawatan.
        Output HARUS berupa JSON valid dengan struktur pasti tanpa sesuai struktur berikut (tanpa ```json, tanpa teks tambahan):

        {
        "subjective": "string",
        "objective": "string",
        "plan": "string",
        "tindakan_lanjutan": "string",
        "keterangan": "string",
        "SDKI": ["string", "string"],
        "SIKI": ["string", "string"],
        "SLKI": ["string", "string"]
        }

        Aturan penting:
        1. Semua value wajib berupa STRING tunggal atau NULL, kecuali SDKI, SIKI, SLKI yang berupa ARRAY of STRING.
        2. Gunakan teks "query" dari perawat + referensi SDKI–SIKI–SLKI yang disediakan (jangan ambil dari luar).
        3. Mapping:
        - "subjective" → keluhan pasien dari query.
        - "objective" → hasil pemeriksaan fisik, tanda vital, observasi dari query.
        - "plan" → rencana tindakan dari query, hubungkan dengan intervensi SIKI bila ada kecocokan.
        - "tindakan_lanjutan" → follow-up tambahan dari query.
        - "keterangan" → catatan tambahan dari query.
        - "SDKI" → Ambil semua keterangan dan penjelasan dari assesment.
        - "SIKI" → ambil semua keterangna dan penjelasan dari plan
        - "SLKI" → untuk setiap diagnosa di assessment, pilih satu atau lebih intervensi SLKI yang relevan. Tulis dalam bentuk narasi yang menjelaskan hubungan diagnosa dengan rencana perawatan.
        4. Jangan menukar SDKI ↔ SIKI ↔ SLKI. Semua harus konsisten sesuai baris yang sama dalam referensi.
        5. Rapikan hasil dalam bentuk narasi singkat, tapi tetap dalam format array string.
        """
                },
                {
                    "role": "user",
                    "content": f"Teks catatan:\n{query}\n\nReferensi SDKI–SIKI–SLKI yang boleh dipakai (hanya dari sini):\n{context_text}\n\nTolong hasilkan JSON sesuai aturan di atas."
                }
            ],
        )


        ai_json = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}", "data": None}), 500

    # parse hasil AI
    try:
        parsed = json.loads(ai_json)
    except Exception:
        data = ai_json
        if isinstance(data, str):
            import json,re
            ai_json = re.sub(r"^```(?:json)?|```$", "", ai_json.strip(), flags=re.MULTILINE)
            data = re.sub(r"<\｜.*?？\｜>|<\｜.*?▁of▁sentence｜>", "", ai_json).strip()

            parsed = json.loads(data)
        else :
            parsed = {
                "subjective": query,
                "objective": None,
                "plan": None,
                "tindakan_lanjutan": None,
                "keterangan": ai_json,
                "SDKI": None,
                "SIKI": None,
                "SLKI": None
            }

    # simpan ke database
    laporan = Laporan(
        patient_id=patient_id,
        user_id=perawat_id,
        cppt_id=cppt_id,  # simpan relasi
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective"),
        objective=parsed.get("objective"),
        assessment=cppt.assessment,  # ambil dari CPPT, bukan AI
        plan=parsed.get("plan"),
        tindakan_lanjutan=parsed.get("tindakan_lanjutan"),
        keterangan=parsed.get("keterangan"),
        SDKI=parsed.get("SDKI"),
        SIKI=parsed.get("SIKI"),
        SLKI=parsed.get("SLKI"),
        intervensi_id = intervensi_id
    )
    db.session.add(laporan)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Laporan created successfully",
        "data": laporan_to_dict(laporan)
    }), 201

# ================== READ ==================
@laporan_bp.route("/<int:id>", methods=["GET"])
def get_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404
    return jsonify({"status": 200, "message": "success", "data": laporan_to_dict(laporan)}), 200

# ================== UPDATE ==================
@laporan_bp.route("/<int:id>", methods=["PUT"])
def update_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    payload = request.get_json()
    for field in ["subjective", "objective", "assessment", "plan", "tindakan_lanjutan", "keterangan", "SDKI", "SIKI", "SLKI"]:
        if field in payload:
            setattr(laporan, field, payload[field])

    db.session.commit()
    return jsonify({"status": 200, "message": "Laporan updated successfully", "data": laporan_to_dict(laporan)}), 200

# ================== DELETE ==================
@laporan_bp.route("/<int:id>", methods=["DELETE"])
def delete_laporan(id):
    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404
    db.session.delete(laporan)
    db.session.commit()
    return jsonify({"status": 200, "message": "Laporan deleted successfully", "data": None}), 200

# ================== SEARCH ==================
@laporan_bp.route("/search", methods=["GET"])
def search_laporan():
    keyword = request.args.get("q")
    if not keyword:
        return jsonify({"status": 400, "message": "Query parameter 'q' required", "data": None}), 400

    results = Laporan.query.filter(
        (Laporan.subjective.ilike(f"%{keyword}%")) |
        (Laporan.objective.ilike(f"%{keyword}%")) |
        (Laporan.assessment.ilike(f"%{keyword}%")) |
        (Laporan.plan.ilike(f"%{keyword}%")) |
        (Laporan.keterangan.ilike(f"%{keyword}%"))
    ).all()

    return jsonify({
        "status": 200,
        "message": "success",
        "data": [laporan_to_dict(l) for l in results]
    }), 200

# ================== HELPER ==================
def laporan_to_dict(laporan):
    return {
        "id": laporan.id,
        "patient_id": laporan.patient_id,
        "cppt_id": laporan.cppt_id,
        "user_id": laporan.user_id,
        "tanggal": laporan.tanggal.isoformat() if laporan.tanggal else None,
        "subjective": laporan.subjective,
        "objective": laporan.objective,
        "assessment": laporan.assessment,
        "plan": laporan.plan,
        "tindakan_lanjutan": laporan.tindakan_lanjutan,
        "keterangan": laporan.keterangan,
        "SDKI": laporan.SDKI,
        "SIKI": laporan.SIKI,
        "SLKI": laporan.SLKI,
        "intervensi_id": laporan.intervensi_id
    }
