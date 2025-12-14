from flask import Blueprint, request, jsonify
from app.model import db, Assesment, Patient
import faiss
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import pickle

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

assesment_bp = Blueprint("assesment_bp", __name__, url_prefix="/assesments")

FAISS_INDEX = None
EMBEDDING_DIM = 384
FAISS_INDEX_FILE = "app/faisses/assesment/assesmen.faiss"
PICKLE_FILE = "app/faisses/assesment/assesmen.pkl"

import re

def extract_medical_record_number(text):
    """
    Mencari nomor rekam medis dari text.
    Pola umum: 6–12 digit berturut-turut.
    """
    # Cari angka panjang (6-12 digit)
    match = re.search(r"\b(\d{6,12})\b", text)
    return match.group(1) if match else None


def query_contains_rm_keyword(text):
    """
    Mengecek apakah query mengandung kata kunci RM.
    """
    keywords = [
        "nomor rekam medis",
        "no rekam medis",
        "no rm",
        "norm",
        "no. rm",
        "no.rm",
        "rm",
        "rekam medis"
    ]

    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def initialize_faiss_index():
    global FAISS_INDEX
    if FAISS_INDEX is None:
        if os.path.exists(FAISS_INDEX_FILE):
            FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
        else:
            FAISS_INDEX = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))
    return FAISS_INDEX

def save_faiss_index():
    if FAISS_INDEX is not None:
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_FILE)

def load_mapping():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def split_pasien_perawat(questions):
    pasien_q = []
    perawat_q = []
    for q in questions:
        q_lower = q.lower()
        if "pasien" in q_lower or "?" in q_lower:
            pasien_q.append(q)
        else:
            perawat_q.append(q)
    return pasien_q, perawat_q


# === Utility Functions ===

def initialize_faiss_index():
    """Load FAISS index from file or create new one if not exist"""
    global FAISS_INDEX
    if FAISS_INDEX is None:
        if os.path.exists(FAISS_INDEX_FILE):
            FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
        else:
            FAISS_INDEX = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIM))
    return FAISS_INDEX


def save_faiss_index():
    """Save FAISS index to file"""
    if FAISS_INDEX is not None:
        faiss.write_index(FAISS_INDEX, FAISS_INDEX_FILE)


def load_mapping():
    """Load mapping id→text dari pickle"""
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


# === Routes ===

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
            import json,re
            data = re.sub(r"^```json\s*|\s*```$", "", data.strip(), flags=re.DOTALL)
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
        return jsonify({"status": 400, "message": "Fields required: query, perawat"}), 400

    query = payload["query"]
    perawat = payload["perawat"]

    rm_in_query = extract_medical_record_number(query)
    keyword_found = query_contains_rm_keyword(query)

    if not keyword_found or not rm_in_query:
        return jsonify({
            "status": 400,
            "message": (
                "Query harus mencantumkan nomor rekam medis pasien. "
                "Pastikan ada kata kunci seperti 'nomor rekam medis', 'no rm', atau angka RM."
            )
        }), 400
    # ------------------------------------------------------

    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")

    # --- Load FAISS index & mapping ---
    index = initialize_faiss_index()
    mapping = load_mapping()

    if not mapping:
        return jsonify({"status": 210, "message": "Belum ada data historis assesmen untuk referensi"}), 400

    # --- Retrieval top-k relevan ---
    retrieved_data = []
    try:
        if index.ntotal > 0:
            k = min(3, index.ntotal)
            D, I = index.search(query_vector, k)
            for idx in I[0]:
                if idx != -1 and idx in mapping:
                    retrieved_data.append(mapping[idx])
    except Exception as e:
        return jsonify({"status": 500, "message": f"FAISS retrieval failed: {str(e)}"}), 500

    context = "\n".join(retrieved_data) if retrieved_data else "Tidak ada data relevan yang ditemukan."

    # --- Prompt ke AI ---
    try:
        prompt = f"""
        Berdasarkan data historis asesmen sebelumnya:
        {context}

        Susun JSON terstruktur untuk ASESMEN AWAL KEPERAWATAN RAWAT INAP.
        Root key: "asesmen_awal_keperawatan".
        Pastikan semua field:
        informasi_umum, data_kunjungan, keluhan_utama, pemeriksaan_fisik, 
        tanda_vital, pemeriksaan_sistem, alergi, asesmen_nyeri, 
        skrining_gizi (berat_badan, tinggi_badan, IMT, status_gizi), 
        skrining_risiko_jatuh, status_psikososial, rencana_perawatan, 
        masalah_keperawatan (analisis SDKI), edukasi, serta 
        field kosong wajib bernama rencana_asuhan_keperawatan.

        Pastikan JSON tidak ada bagian <｜begin▁of▁sentence｜>.

        Nomor rekam medis pasien: {rm_in_query}
        Query baru pasien:
        {query}
        """

        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten medis yang menyusun data asesmen berdasarkan referensi historis."},
                {"role": "user", "content": prompt},
            ],
        )
        import re
        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"<\｜begin▁of▁sentence｜>", "", ai_json).strip()

    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}"}), 500

    # --- Save to DB ---
    new_assesment = Assesment(
        perawat=perawat,
        tanggal=datetime.utcnow(),
        data=ai_json,
    )
    db.session.add(new_assesment)
    db.session.commit()

    # --- Update FAISS ---
    try:
        index.add_with_ids(query_vector, np.array([new_assesment.id]))
        save_faiss_index()
    except Exception as e:
        return jsonify({"status": 500, "message": f"FAISS update failed: {str(e)}"}), 500

    return jsonify({
        "status": 201,
        "message": "Assesment created successfully",
        "data": {
            "id": new_assesment.id,
            "perawat": perawat,
            "tanggal": new_assesment.tanggal.isoformat(),
            "data": ai_json,
        },
    }), 201



@assesment_bp.route("/search", methods=["POST"])
def search_assesments():
    payload = request.get_json()
    query_string = payload.get("query") if payload else None

    if not query_string:
        return jsonify({"status": 400, "message": "Missing field: query"}), 400

    index = initialize_faiss_index()
    mapping = load_mapping()

    if index.ntotal == 0:
        return jsonify({"status": 200, "message": "No assessments in index", "data": []}), 200

    query_vector = model.encode([query_string], convert_to_numpy=True).astype("float32")

    k = 10
    distances, indices = index.search(query_vector, k)

    results = []
    for i, assesment_id in enumerate(indices[0]):
        if assesment_id in mapping:
            results.append(
                {
                    "id": int(assesment_id),
                    "text_chunk": mapping[assesment_id],
                    "relevance_score": float(1 / (1 + distances[0][i])),
                }
            )

    return jsonify({"status": 200, "message": "Success", "data": results}), 200


@assesment_bp.route("/questions", methods=["GET"])
def get_assesmen_questions():
    # --- Pastikan file FAISS dan mapping ada ---
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(PICKLE_FILE):
        return jsonify({
            "status": 404,
            "message": "File assesmen.faiss atau assesmen.pkl belum tersedia",
            "data": None
        }), 404

    # --- Load FAISS index dan mapping ---
    mapping = load_mapping()

    # --- Ambil semua chunk dari mapping ---
    all_chunks = list(mapping.values())

    if not all_chunks:
        return jsonify({
            "status": 200,
            "message": "Tidak ada data di file assesmen",
            "data": {"pasien": [], "perawat": []}
        }), 200

    # --- Prompt ke AI untuk generate pertanyaan ---
    context_text = "\n".join(all_chunks)
    try:
        prompt = f"""
        Berdasarkan semua data asesmen yang tersedia:
        {context_text}

        Buat JSON pertanyaan yang akan diajukan:
        1. "pasien" → pertanyaan untuk pasien yang di dasarakan pada data assesmen yang tersedia
        2. "perawat" → pertanyaan yang diisi oleh perawat berdasarkan data assesmen yang tersedia yang menjadi objectif observasi perawat terhadap pasien

        Pastikan:
        - Semua value berupa string pertanyaan
        - Struktur pertanyaan menyesuaikan yang hanya ada pada kategori assesmen secara lengkap sesuai konteks di atas tanpa ada yang tertinggal dan pastikan struktur json yang dihasilkan itu sama seperti contoh.
        - JSON valid sebagai string

        Contoh output (jangan terlalu mengcopy contoh ini, buat pertanyaan baru):
        {{
          "pasien": [{{"list_pertanyaan": ["Apa alamat pasien?,..] "}}, ...],
          "perawat": [{{"list_pertanyaan": ["Hasil tanda vital pasien?,..]"}}, ...]
        }}
        """

        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten medis yang membuat daftar pertanyaan asesmen."},
                {"role": "user", "content": prompt}
            ],
        )
        ai_json = completion.choices[0].message.content

    except Exception as e:
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}", "data": None}), 500

    # --- Parsing JSON dari AI ---
    try:
        import json, re
        ai_json = re.sub(r"^```(?:json)?|```$", "", ai_json.strip(), flags=re.MULTILINE)
        ai_json = re.sub(r"<\｜.*?？\｜>|<\｜.*?▁of▁sentence｜>", "", ai_json).strip()
        parsed_questions = json.loads(ai_json)
    except Exception:
        parsed_questions = {
            "pasien": [{"informasi_umum": "Pertanyaan tidak tersedia"}],
            "perawat": [{"observasi": "Pertanyaan tidak tersedia"}],
            "debug_ai_json": ai_json
        }

    return jsonify({
        "status": 200,
        "message": "Pertanyaan assesmen berhasil diambil",
        "data": parsed_questions
    }), 200

# === UPDATE ===
@assesment_bp.route("/<int:assesment_id>", methods=["PUT", "PATCH"])
def update_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    payload = request.get_json()
    if not payload:
        return jsonify({"status": 400, "message": "No data provided"}), 400

    data = payload.get("data", assesment.perawat)
    perawat = payload.get("perawat", None)

    # Update perawat
    assesment.perawat = perawat
    assesment.tanggal = datetime.utcnow()
    data_str = data if isinstance(data, str) else str(data)
    assesment.data = data_str
    perawat = perawat if perawat else "Unknown"
    assesment.perawat = perawat

    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Assesment updated successfully",
        "data": {
            "id": assesment.id,
            "perawat": assesment.perawat,
            "tanggal": assesment.tanggal.isoformat(),
            "data": assesment.data,
        }
    }), 200


# === DELETE ===
@assesment_bp.route("/<int:assesment_id>", methods=["DELETE"])
def delete_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    try:
        # Hapus dari DB
        db.session.delete(assesment)
        db.session.commit()

        # Hapus dari FAISS index
        index = initialize_faiss_index()
        try:
            index.remove_ids(np.array([assesment.id], dtype=np.int64))
            save_faiss_index()
        except Exception:
            pass

        # Hapus dari mapping
        mapping = load_mapping()
        if assesment.id in mapping:
            del mapping[assesment.id]
            with open(PICKLE_FILE, "wb") as f:
                pickle.dump(mapping, f)

    except Exception as e:
        return jsonify({"status": 500, "message": f"Delete failed: {str(e)}"}), 500

    return jsonify({"status": 200, "message": "Assesment deleted successfully"}), 200



