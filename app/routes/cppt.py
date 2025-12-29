from flask import Blueprint, request, jsonify
from app.model import db, CPPT, Patient, Assesment
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import pickle

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

# --- Load FAISS Index & Mapping (SDKI-SLKI-SIKI sudah diekstrak) ---
FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"


def load_faiss_and_mapping():
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        berhasil = True
    else:
        berhasil = False
    return berhasil



def search_diagnosa_sdki(query,id_to_text, k=3, index=None):
    """Cari diagnosa SDKI dari index FAISS berdasarkan input query."""
    if index is None or not id_to_text:
        # fallback: tidak ada index, balikan kosong
        return []

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_text[i] for i in I[0] if i != -1 and i in id_to_text]

# --- Blueprint ---
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
    assessment_id = payload["assessment_id"]
    patient_id = payload["patient_id"]
    perawat_id = payload["perawat_id"]

    patient = Patient.query.get(patient_id)
    assessment = Assesment.query.get(assessment_id)
    try:
        data = assessment.data
        if isinstance(data, str):
            import json, re
            data = re.sub(r"^```json\s*|\s*```$", "", data.strip(), flags=re.DOTALL)
            json_ku = json.loads(data)
    except Exception:
        return jsonify({"status": 500, "message": "Invalid JSON in assesment", "data": data}), 500

    query = query + f", assesment : {json_ku.get("asesmen_awal_keperawatan", {}).get("masalah_keperawatan", [])}"
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found", "data": None}), 404
    
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "rb") as f:
            id_to_text = pickle.load(f)
        sdki_matches = search_diagnosa_sdki(query, k=3, index= faiss.read_index(FAISS_INDEX_FILE), id_to_text=id_to_text)
    else:
        return jsonify({"status": 210, "message": "belum mengupload data sdki-siki-slki", "data": None}), 404
    # --- Cari diagnosa SDKI dari FAISS ---

    context_text = "\n\n".join(sdki_matches)

    try:
        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {
                    "role": "system",
                    "content": """Kamu adalah asisten medis.
        Tugasmu menyusun CPPT (Catatan Perkembangan Pasien Terintegrasi) dari teks catatan pasien dan referensi SDKI.

        Aturan penting:
        1. Output HANYA JSON valid, tanpa ```json, catatan tambahan, atau teks di luar JSON.
        2. Semua value WAJIB berupa STRING tunggal (bukan list/array).
        3. Struktur JSON:
        {
        "subjective": "string",
        "objective": "string",
        "assessment": "string",
        "plan": "string",
        "keterangan": "string",
        "dokter": "string",
        "signature": null
        }
        4. Mapping konten:
        - subjective → keluhan pasien dari query.
        - objective → hasil pemeriksaan fisik, tanda vital, observasi.
        - assessment → gabungkan semua diagnosa SDKI dari query assessment menjadi satu narasi (dipisahkan dengan koma atau kalimat) dimana diagnosa itu di dasarkan dari context.
        - plan → untuk setiap diagnosa di assessment, pilih satu atau lebih intervensi SIKI yang relevan. Tulis dalam bentuk narasi yang menjelaskan hubungan diagnosa dengan rencana perawatan.
        Contoh format narasi:
        "Untuk diagnosa [nama diagnosa]: [intervensi SIKI]. Untuk diagnosa [nama diagnosa lain]: [intervensi SIKI]."
        - keterangan → catatan tambahan dari query.
        - dokter → nama dokter jika ada, jika tidak "".
        - signature → null jika tidak ada tanda tangan.
        5. Jangan ubah makna medis dari query. Rapikan menjadi narasi medis yang utuh dan masuk akal.
        """
                },
                {
                    "role": "user",
                    "content": f"Teks catatan:\n{query}\n\nReferensi Diagnosa & Intervensi SDKI/SIKI:\n{context_text}"
                }
            ],
        )

        ai_json = completion.choices[0].message.content
    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}", "data": None}), 500

    try:
        import re
        ai_json = re.sub(r"^```(?:json)?|```$", "", ai_json.strip(), flags=re.MULTILINE)
        ai_json = re.sub(r"<\｜.*?？\｜>|<\｜.*?▁of▁sentence｜>", "", ai_json).strip()
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

    # --- Simpan ke database ---
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
        return jsonify({
            "status": 404,
            "message": "CPPT tidak ditemukan",
            "data": None
        }), 404

    try:
        db.session.delete(cppt)
        db.session.commit()
        return jsonify({
            "status": 200,
            "message": "CPPT berhasil dihapus",
            "data": {"id": cppt_id}
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "status": 500,
            "message": f"Gagal menghapus CPPT: {str(e)}",
            "data": None
        }), 500



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
