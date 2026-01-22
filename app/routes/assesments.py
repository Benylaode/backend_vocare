from flask import Blueprint, request, jsonify
from app.model import db, Assesment, Patient, User
from flask_jwt_extended import jwt_required, get_jwt_identity
import faiss
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import pickle
import re
import json

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

# === Utility Functions ===

def extract_medical_record_number(text):
    """Mencari nomor rekam medis (6-12 digit) dari text."""
    match = re.search(r"\b(\d{6,12})\b", text)
    return match.group(1) if match else None

def query_contains_rm_keyword(text):
    """Mengecek apakah query mengandung kata kunci RM."""
    keywords = ["nomor rekam medis", "no rekam medis", "no rm", "norm", "no. rm", "no.rm", "rm", "rekam medis"]
    return any(kw in text.lower() for kw in keywords)

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


@assesment_bp.route("/", methods=["GET"])
@jwt_required()
def get_assesments():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    query = Assesment.query
    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    assesments = query.order_by(Assesment.tanggal.desc()).all()
    
    data = []
    for a in assesments:
        konten_data = a.data
        if isinstance(konten_data, str):
            try:
                konten_data = json.loads(konten_data)
            except:
                pass 

        data.append({
            "id": a.id,
            "tanggal": a.tanggal.isoformat(),
            "perawat": a.user.username if a.user else (a.perawat if hasattr(a, 'perawat') else "Unknown"),
            "data": konten_data,
            "patient_rm": a.patient.no_rekam_medis if a.patient else None
        })
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@assesment_bp.route("/<int:assesment_id>", methods=["GET"])
@jwt_required()
def get_assesment(assesment_id):
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404
    
    konten = assesment.data
    if isinstance(konten, str):
        try:
            konten = re.sub(r"^```json\s*|\s*```$", "", konten.strip(), flags=re.DOTALL)
            konten = json.loads(konten)
        except Exception:
            pass

    data = {
        "id": assesment.id,
        "tanggal": assesment.tanggal.isoformat(),
        "perawat": assesment.user.username if assesment.user else "Unknown",
        "data": konten,
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


@assesment_bp.route("/", methods=["POST"])
@jwt_required()  
def create_assesment():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    payload = request.get_json()
    
    if not payload or not payload.get("query") or not payload.get("patient_id"):
        return jsonify({"status": 400, "message": "Fields required: query, patient_id"}), 400

    query = payload["query"]
    perawat_name = user.username 
    patient_id = payload["patient_id"]

    rm_in_query = extract_medical_record_number(query)
    keyword_found = query_contains_rm_keyword(query)

    if not keyword_found and not rm_in_query:
         return jsonify({
            "status": 400,
            "message": "Query harus mencantumkan nomor rekam medis pasien. Pastikan ada kata kunci seperti 'nomor rekam medis', 'no rm', atau angka RM."
        }), 400

    # 2. Cari Pasien & Validasi Ruangan
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404

    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
         return jsonify({"status": 403, "message": f"Akses Ditolak. Pasien di ruangan {patient.ruangan}."}), 403

    # 3. FAISS Retrieval (PROMPT ASLI ANDA)
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    index = initialize_faiss_index()
    mapping = load_mapping()

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

    # --- PROMPT (SESUAI PERMINTAAN ANDA) ---
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
        ai_resp = completion.choices[0].message.content
        # Bersihkan response AI
        import re
        ai_json = re.sub(r"<\｜begin▁of▁sentence｜>", "", ai_resp).strip()
        # Bersihkan markdown json jika ada
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json, flags=re.MULTILINE)
        
        # Validasi JSON agar bisa disimpan di kolom type JSON
        parsed_data = json.loads(ai_json)

    except Exception as e:
        # Fallback jika AI error
        parsed_data = {"error": str(e), "raw_response": ai_json if 'ai_json' in locals() else ""}
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}"}), 500

    # 4. Simpan ke Database
    new_assesment = Assesment(
        patient_id=patient.id,
        user_id=user.id, 
        tanggal=datetime.utcnow(),
        data=parsed_data 
    )
    db.session.add(new_assesment)
    db.session.commit()

    # 5. Update FAISS
    try:
        summary_text = f"RM:{patient.no_rekam_medis} | {query}"
        new_id = new_assesment.id 
        index.add_with_ids(query_vector, np.array([new_id]).astype('int64'))
        save_faiss_index()
        
        mapping[new_id] = summary_text
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(mapping, f)
    except Exception:
        pass

    return jsonify({
        "status": 201,
        "message": "Assesment created successfully",
        "data": {
            "id": new_assesment.id,
            "perawat": perawat_name,
            "tanggal": new_assesment.tanggal.isoformat(),
            "data": parsed_data,
        },
    }), 201

# === SEARCH & QUESTIONS (Tetap sama seperti logika Anda) ===
@assesment_bp.route("/search", methods=["POST"])
@jwt_required()
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
            results.append({
                "id": int(assesment_id),
                "text_chunk": mapping[assesment_id],
                "relevance_score": float(1 / (1 + distances[0][i])),
            })

    return jsonify({"status": 200, "message": "Success", "data": results}), 200

@assesment_bp.route("/questions", methods=["GET"])
@jwt_required()
def get_assesmen_questions():
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(PICKLE_FILE):
        return jsonify({"status": 404, "message": "File assesmen belum tersedia", "data": None}), 404

    mapping = load_mapping()
    all_chunks = list(mapping.values())

    if not all_chunks:
        return jsonify({"status": 200, "message": "Tidak ada data", "data": {"pasien": [], "perawat": []}}), 200

    # Ambil sample 15 terakhir agar tidak overload
    context_text = "\n".join(all_chunks[-15:])
    
    try:
        prompt = f"""
        Berdasarkan semua data asesmen yang tersedia:
        {context_text}

        Buat JSON pertanyaan yang akan diajukan:
        1. "pasien" → pertanyaan untuk pasien
        2. "perawat" → pertanyaan observasi perawat

        Contoh output:
        {{
          "pasien": [{{"list_pertanyaan": ["..."]}}],
          "perawat": [{{"list_pertanyaan": ["..."]}}]
        }}
        """

        completion = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "system", "content": "Assistant medis."}, {"role": "user", "content": prompt}]
        )
        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```(?:json)?|```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed_questions = json.loads(ai_json)
        
        return jsonify({"status": 200, "message": "Pertanyaan berhasil diambil", "data": parsed_questions}), 200

    except Exception as e:
        return jsonify({"status": 500, "message": f"AI Error: {str(e)}"}), 500