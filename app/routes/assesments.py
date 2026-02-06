from zoneinfo import ZoneInfo
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
from sqlalchemy import or_

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

def delete_from_faiss(assesment_id):
    """Menghapus ID dari FAISS index dan Mapping Pickle"""
    global FAISS_INDEX
    index = initialize_faiss_index()
    mapping = load_mapping()

    # Hapus dari Index FAISS
    try:
        index.remove_ids(np.array([assesment_id], dtype=np.int64))
        save_faiss_index()
    except Exception as e:
        print(f"Warning: Failed to remove ID {assesment_id} from FAISS: {e}")

    # Hapus dari Mapping Pickle
    if assesment_id in mapping:
        del mapping[assesment_id]
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(mapping, f)


# === CRUD ENDPOINTS ===

# 1. READ ALL (GET)
@assesment_bp.route("/", methods=["GET"])
@jwt_required()
def get_assesments():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    # Gunakan outerjoin agar assesment tanpa pasien (standalone) tetap muncul
    query = Assesment.query.outerjoin(Patient)

    if user.role.name != 'admin' and user.ruangan:
        # Filter: Assesment di ruangan user ATAU Assesment yang dibuat user itu sendiri
        query = query.filter(
            or_(
                Patient.ruangan == user.ruangan,
                Assesment.user_id == user.id
            )
        )

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
            "patient_id": a.patient_id,  # ← PASTI ADA (None / int)
            "patient_rm": a.patient.no_rekam_medis if a.patient else "Draft / Belum Ditentukan"
        })

    return jsonify({"status": 200, "message": "Success", "data": data}), 200


# 2. READ ONE (GET)
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
        "patient_id": assesment.patient_id,
        "patient_rm": assesment.patient.no_rekam_medis if assesment.patient else None
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


# 3. CREATE (POST)
@assesment_bp.route("/", methods=["POST"])
@jwt_required()  
def create_assesment():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    payload = request.get_json()
    
    if not payload or not payload.get("query"):
        return jsonify({"status": 400, "message": "Field required: query"}), 400

    query = payload["query"]
    perawat_name = user.username 
    
    # Patient ID Opsional (Standalone mode)
    patient_id = payload.get("patient_id")
    
    rm_in_query = extract_medical_record_number(query)
    keyword_found = query_contains_rm_keyword(query)
    patient = None

    # Validasi jika patient_id disertakan
    if patient_id:
        if not keyword_found and not rm_in_query:
             return jsonify({
                "status": 400,
                "message": "Query harus mencantumkan nomor rekam medis pasien jika ingin mengaitkan data secara langsung."
            }), 400

        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404

        if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
             return jsonify({"status": 403, "message": f"Akses Ditolak. Pasien di ruangan {patient.ruangan}."}), 403
    
    # RAG: Search context
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
    rm_info = rm_in_query if rm_in_query else (patient.no_rekam_medis if patient else "Belum Ditentukan")
    
    # AI Generation
    try:
        prompt = f"""
        Berdasarkan data historis asesmen sebelumnya:
        {context}

        Susun JSON terstruktur untuk ASESMEN AWAL KEPERAWATAN RAWAT INAP.
        Root key: "asesmen_awal_keperawatan".
        Pastikan semua field terisi: informasi_umum, tanda_vital, keluhan_utama, pemeriksaan_fisik, rencana_asuhan_keperawatan, dll.
        Nomor rekam medis pasien: {rm_info}
        Query baru pasien: {query}
        """

        completion = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten medis yang menyusun data asesmen."},
                {"role": "user", "content": prompt},
            ],
        )
        ai_resp = completion.choices[0].message.content
        ai_json = re.sub(r"<\｜begin▁of▁sentence｜>", "", ai_resp).strip()
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json, flags=re.MULTILINE)
        parsed_data = json.loads(ai_json)

    except Exception as e:
        parsed_data = {"error": str(e), "raw_response": ai_json if 'ai_json' in locals() else ""}
        return jsonify({"status": 500, "message": f"AI processing failed: {str(e)}"}), 500

    # Simpan ke DB
    new_assesment = Assesment(
        patient_id=patient.id if patient else None,
        user_id=user.id, 
        tanggal=datetime.now(ZoneInfo("Asia/Makassar")).date(),
        data=parsed_data 
    )
    db.session.add(new_assesment)
    db.session.commit()

    # Update FAISS jika ada pasien (agar relevan untuk pencarian masa depan)
    if patient:
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
            "patient_id": new_assesment.patient_id,
            "data": parsed_data,
        },
    }), 201


# 4. UPDATE (PUT) - NEW!
@assesment_bp.route("/<int:assesment_id>", methods=["PUT"])
@jwt_required()
def update_assesment(assesment_id):
    """
    Update data asesmen (hasil JSON) atau menautkan ke pasien (patient_id).
    """
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"status": 400, "message": "No data provided"}), 400

    # A. Update Data JSON (Manual Correction dari Frontend)
    if "data" in data:
        assesment.data = data["data"]

    # B. Update/Link Patient ID
    if "patient_id" in data:
        new_patient_id = data["patient_id"]
        # Jika user mengirim null, berarti unlink
        if new_patient_id is None:
            assesment.patient_id = None
        else:
            # Validasi pasien ada
            patient = Patient.query.get(new_patient_id)
            if not patient:
                 return jsonify({"status": 404, "message": "Patient ID not found"}), 404
            assesment.patient_id = new_patient_id

    try:
        db.session.commit()
        return jsonify({"status": 200, "message": "Assesment updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Database Error: {str(e)}"}), 500


# 5. DELETE (DELETE) - NEW!
@assesment_bp.route("/<int:assesment_id>", methods=["DELETE"])
@jwt_required()
def delete_assesment(assesment_id):
    """
    Hapus asesmen dari Database DAN dari Index FAISS (agar tidak muncul di RAG lagi).
    """
    assesment = Assesment.query.get(assesment_id)
    if not assesment:
        return jsonify({"status": 404, "message": "Assesment not found"}), 404

    try:
        # 1. Hapus dari Database
        db.session.delete(assesment)
        db.session.commit()

        # 2. Hapus dari FAISS (Pembersihan Data)
        delete_from_faiss(assesment_id)

        return jsonify({"status": 200, "message": "Assesment deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Delete Failed: {str(e)}"}), 500


# === SEARCH & QUESTIONS ===

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
    # GENERAL FIELDS FIXED (Hardcoded sesuai permintaan sebelumnya)
    FIXED_GENERAL_FIELDS = [
        "Berapa nomor rekam medis pasien?",
        "Siapa nama lengkap pasien?",
        "Apa jenis kelamin pasien?",
        "Kapan tanggal lahir pasien?",
        "Apa status perkawinan pasien?",
        "Dimana alamat pasien?",
        "Apa pekerjaan pasien?",
        "Siapa nama penanggung jawab pasien?",
        "Apa hubungan penanggung jawab dengan pasien?",
        "Bagaimana kontak penanggung jawab?",
        "Tanggal/Jam berapa pasien tiba?",
        "Cara masuk (Jalan/Kursi Roda/Brankar)?",
        "Pasien masuk ke poliklinik mana?",
        "Apakah pasien datang dengan rujukan?",
        "Siapa pendamping pasien saat datang?",
        "Kelas pelayanan apa yang digunakan?",
        "Apa sumber data anamnesa?",
        "Keluhan yang dirasakan pasien?"
    ]

    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(PICKLE_FILE):
        return jsonify({
            "status": 200,
            "message": "Belum ada data historis.",
            "data": {
                "general_fields": FIXED_GENERAL_FIELDS,
                "pasien": [],
                "perawat": []
            }
        }), 200

    mapping = load_mapping()
    all_chunks = list(mapping.values())

    if not all_chunks:
        return jsonify({
            "status": 200, 
            "message": "Data historis kosong", 
            "data": {"general_fields": FIXED_GENERAL_FIELDS, "pasien": [], "perawat": []}
        }), 200

    context_text = "\n".join(all_chunks[-20:])

    try:
        prompt = f"""
        Analisis konteks data klinis historis berikut:
        {context_text}

        Tugas: Buat daftar pertanyaan asesmen keperawatan berbasis RAG.
        Output WAJIB berupa JSON dengan 2 key utama (general_fields sudah ada, tidak perlu dibuat):
        1. "pasien" → ARRAY berisi TEPAT 10 pertanyaan wawancara pasien.
        2. "perawat" → ARRAY berisi TEPAT 10 instruksi observasi/pemeriksaan fisik.
        """

        completion = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}]
        )

        ai_resp = completion.choices[0].message.content
        ai_json = re.sub(r"^```(?:json)?|```$", "", ai_resp.strip(), flags=re.MULTILINE)
        parsed_questions = json.loads(ai_json)

        # Safety check
        if "pasien" not in parsed_questions: parsed_questions["pasien"] = []
        if "perawat" not in parsed_questions: parsed_questions["perawat"] = []

        return jsonify({
            "status": 200,
            "message": "Daftar pertanyaan berhasil dibuat",
            "data": {
                "general_fields": FIXED_GENERAL_FIELDS,
                "pasien": parsed_questions["pasien"],
                "perawat": parsed_questions["perawat"]
            }
        }), 200

    except Exception as e:
        return jsonify({"status": 500, "message": f"AI Error: {str(e)}"}), 500