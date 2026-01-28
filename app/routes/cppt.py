from flask import Blueprint, request, jsonify
from app.model import db, CPPT, Patient, Laporan, User
from flask_jwt_extended import jwt_required, get_jwt_identity
import os, json, pickle
from datetime import datetime, time
from dotenv import load_dotenv
from openai import OpenAI
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY_KU")
api_model = os.getenv("API_MODEL")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# Inisialisasi Model Embedding untuk RAG
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Failed to load SentenceTransformer: {e}")
    model = None

# Lokasi File FAISS SDKI/SIKI (Sesuaikan path folder Anda)
FAISS_INDEX_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.faiss"
MAPPING_FILE = "app/faisses/siki-slki-sdki/siki-slki-sdki.pkl"

cppt_bp = Blueprint("cppt_bp", __name__, url_prefix="/cppt")

# --- Utility Functions ---

def determine_shift(dt):
    t = dt.time()
    if time(7, 0) <= t < time(14, 0): return "Pagi"
    elif time(14, 0) <= t < time(21, 0): return "Sore"
    else: return "Malam"

def ensure_string(value):
    """
    CRITICAL FIX: Mengubah Dict/List menjadi JSON String.
    Mencegah error: (psycopg2.ProgrammingError) can't adapt type 'dict'
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def search_sdki_siki(query):
    """
    RAG Logic: Mencari referensi SDKI/SIKI relevan dari FAISS 
    berdasarkan keluhan/input perawat.
    """
    if not model or not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(MAPPING_FILE):
        return "(Sistem RAG belum siap/file index tidak ditemukan)"
    
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(MAPPING_FILE, "rb") as f:
            mapping = pickle.load(f)
        
        query_vector = model.encode([query]).astype("float32")
        k = 5 # Ambil 5 referensi teratas
        D, I = index.search(query_vector, k)
        
        results = []
        for idx in I[0]:
            if idx in mapping:
                results.append(mapping[idx])
        
        if not results:
            return "Tidak ditemukan referensi spesifik."
            
        return "\n- ".join(results)
    except Exception as e:
        print(f"RAG Search Error: {e}")
        return ""

# --- CRUD Routes ---

# 1. READ ALL (List)
@cppt_bp.route("/", methods=["GET"])
@jwt_required()
def get_cppts():
    cppts = CPPT.query.order_by(CPPT.tanggal.desc()).all()
    data = [{
        "id": c.id, 
        "patient_id": c.patient_id, 
        "patient_name": c.patient.nama if c.patient else "Unknown",
        "tanggal": c.tanggal.isoformat(),
        "shift": c.shift, 
        "subjective": c.subjective, 
        "objective": c.objective,
        "assessment": c.assessment, 
        "plan": c.plan
    } for c in cppts]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


# 2. READ ONE (Detail)
@cppt_bp.route("/<int:cppt_id>", methods=["GET"])
@jwt_required()
def get_cppt_detail(cppt_id):
    c = CPPT.query.get(cppt_id)
    if not c:
        return jsonify({"status": 404, "message": "CPPT not found"}), 404

    data = {
        "id": c.id, 
        "patient_id": c.patient_id, 
        "patient_name": c.patient.nama if c.patient else "Unknown",
        "tanggal": c.tanggal.isoformat(),
        "shift": c.shift, 
        "subjective": c.subjective, 
        "objective": c.objective,
        "assessment": c.assessment, 
        "plan": c.plan,
        "keterangan": c.keterangan,
        "dokter": c.dokter
    }
    return jsonify({"status": 200, "message": "Success", "data": data}), 200


# 3. CREATE (POST) with RAG & AI
@cppt_bp.route("/", methods=["POST"])
@jwt_required()
def create_cppt():
    user_id = int(get_jwt_identity())
    data = request.get_json()
    
    patient_id = data.get("patient_id")
    query = data.get("query")
    
    if not patient_id or not query:
        return jsonify({"status": 400, "message": "Patient ID and Query required"}), 400

    # 1. Ambil Data Pasien
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found"}), 404

    # 2. Ambil Konteks Laporan Terakhir
    laporan = Laporan.query.filter_by(patient_id=patient_id).order_by(Laporan.tanggal.desc()).first()
    laporan_context = "Belum ada Askeb sebelumnya."
    if laporan:
        laporan_context = f"Diagnosa Sebelumnya: {laporan.assessment}\nPlan Sebelumnya: {laporan.plan}"
    
    # 3. Pencarian RAG
    referensi_text = search_sdki_siki(query) 
    
    # 4. Prompt Engineering (Strict Row Mapping + Clinical Relevance Filter)
    system_prompt = """
    Anda adalah Perawat Senior & Validator Data Klinis.
    Tugas: Menyusun CPPT (SOAP) dari narasi perawat, HANYA menggunakan data yang relevan secara medis.

    STRUKTUR REFERENSI:
    [NO] | [SDKI] | [SIKI] | [Data Subjektif] | [Data Objektif]

    TAHAP 1: FILTERISASI DATA (CRITICAL)
    - Input perawat mungkin bercampur dengan cerita non-medis.
    - **ATURAN FILTER:** Buang semua kalimat yang tidak berhubungan dengan kondisi fisik/psikologis pasien.
      - *Contoh Dibuang:* "Pasien sedang menonton TV", "Keluarga datang menjenguk", "Pasien minta ganti channel".
      - *Contoh Diambil:* "Pasien mengeluh pusing", "Tampak gelisah", "TD 130/80", "Tidak mau makan".
    - Pisahkan hasil filter menjadi:
      - **Data S (Subjektif):** Apa yang pasien katakan tentang keluhannya.
      - **Data O (Objektif):** Apa yang terukur/terlihat (TTV, Ekspresi, Hasil Lab).

    TAHAP 2: PENGUNCIAN BARIS (ROW LOCKING)
    1. Bandingkan Data S & O yang sudah difilter dengan kolom "Data Subjektif/Objektif" di Tabel Referensi.
    2. Pilih **SATU NOMOR (NO)** penyakit yang gejalanya paling cocok (Match > 80%).
    3. Ambil SDKI dan SIKI **HANYA** dari baris Nomor tersebut.

    FORMAT OUTPUT JSON (Strict):
    {
        "subjective": "Ringkasan Data S yang sudah dibersihkan (Hanya keluhan klinis)",
        "objective": "Ringkasan Data O yang sudah dibersihkan (Hanya tanda klinis)",
        "assessment": "Diagnosis SDKI (Copy persis dari baris terpilih)",
        "plan": "Intervensi SIKI (Copy persis dari baris terpilih)",
        "keterangan": "Validasi: 'Memilih No.[X] karena data S:[...] & O:[...] sesuai.'"
    }
    """

    user_prompt_content = f"""
    DATA PASIEN: {patient.nama}
    RIWAYAT: {laporan_context}
    
    INPUT MENTAH PERAWAT:
    "{query}"

    TABEL REFERENSI (PDF):
    {referensi_text}
    """

    try:
        completion = client.chat.completions.create(
            model=api_model,
            temperature=0.1, # Sangat rendah untuk meminimalisir 'mengarang'
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ]
        )
        
        ai_resp = completion.choices[0].message.content
        ai_clean = re.sub(r"^```json\s*|\s*```$", "", ai_resp.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_clean)
        
    except Exception as e:
        print(f"AI/Parsing Error: {e}")
        parsed = {
            "subjective": query, # Fallback ke raw query jika gagal
            "objective": "-",
            "assessment": "Gagal Verifikasi AI",
            "plan": "-",
            "keterangan": "Error System"
        }

    # 5. Simpan
    now = datetime.utcnow()
    shift = determine_shift(now)

    try:
        new_cppt = CPPT(
            patient_id=patient_id, 
            user_id=user_id, 
            tanggal=now, 
            shift=shift,
            subjective=parsed.get("subjective", "-"),
            objective=parsed.get("objective", "-"),
            assessment=parsed.get("assessment", "-"),
            plan=parsed.get("plan", "-"),
            keterangan=parsed.get("keterangan", ""),
            laporan_id=laporan.id if laporan else None 
        )
        
        db.session.add(new_cppt)
        db.session.commit()
    except Exception as db_err:
        db.session.rollback()
        return jsonify({"status": 500, "message": str(db_err)}), 500

    return jsonify({
        "status": 201, 
        "message": "CPPT Tervalidasi", 
        "data": {
            "id": new_cppt.id,
            "s": new_cppt.subjective,
            "o": new_cppt.objective,
            "a": new_cppt.assessment,
            "p": new_cppt.plan
        }
    }), 201


# 4. UPDATE (PUT)
@cppt_bp.route("/<int:cppt_id>", methods=["PUT"])
@jwt_required()
def update_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404

    data = request.get_json()
    fields = ["subjective", "objective", "assessment", "plan", "keterangan", "dokter", "shift"]
    
    try:
        for f in fields:
            if f in data: 
                # PENTING: Gunakan ensure_string juga di sini
                # Jika frontend mengirim JSON Object untuk update, harus distringify
                val = ensure_string(data[f])
                setattr(cppt, f, val)

        db.session.commit()
        return jsonify({"status": 200, "message": "CPPT updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Update Failed: {str(e)}"}), 500


# 5. DELETE (Delete)
@cppt_bp.route("/<int:cppt_id>", methods=["DELETE"])
@jwt_required()
def delete_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404
    
    try:
        db.session.delete(cppt)
        db.session.commit()
        return jsonify({"status": 200, "message": "CPPT deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Delete Failed: {str(e)}"}), 500