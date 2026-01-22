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
    FIX ERROR: (psycopg2.ProgrammingError) can't adapt type 'dict'
    Mengubah Dict/List menjadi JSON String agar bisa disimpan di kolom Text Database.
    """
    if value is None:
        return ""
    # Jika value adalah Dictionary atau List (JSON Object), dump jadi String
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    # Jika sudah string atau angka, biarkan
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
        k = 5 # Ambil 5 referensi teratas (SDKI/SIKI paling mirip)
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

# --- Routes ---

@cppt_bp.route("/", methods=["GET"])
@jwt_required()
def get_cppts():
    cppts = CPPT.query.order_by(CPPT.tanggal.desc()).all()
    data = [{
        "id": c.id, 
        "patient_id": c.patient_id, 
        "tanggal": c.tanggal.isoformat(),
        "shift": c.shift, 
        "subjective": c.subjective, 
        "objective": c.objective,
        "assessment": c.assessment, 
        "plan": c.plan
    } for c in cppts]
    return jsonify({"status": 200, "message": "Success", "data": data}), 200

@cppt_bp.route("/", methods=["POST"])
@jwt_required()
def create_cppt():
    user_id = int(get_jwt_identity())
    data = request.get_json()
    
    patient_id = data.get("patient_id")
    query = data.get("query") # Input narasi perkembangan dari perawat
    
    if not patient_id or not query:
        return jsonify({"status": 400, "message": "Patient ID and Query required"}), 400

    # 1. Ambil Data Pasien & Laporan Terakhir (Untuk konteks diagnosa awal)
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Patient not found"}), 404

    laporan = Laporan.query.filter_by(patient_id=patient_id).order_by(Laporan.tanggal.desc()).first()
    
    laporan_context = "Belum ada Askeb sebelumnya."
    if laporan:
        laporan_context = f"Diagnosa Medis/Keperawatan Sebelumnya: {laporan.SDKI}\nRencana Sebelumnya: {laporan.SIKI}"
    
    # 2. Lakukan Pencarian RAG (SDKI & SIKI) berdasarkan Query Perawat
    # Ini memastikan AI mendapatkan referensi buku standar, bukan halusinasi
    referensi_standar = search_sdki_siki(query)
    
    # 3. Susun Prompt Spesifik (SOAP dengan Standar SDKI/SIKI)
    prompt = f"""
    Anda adalah Perawat Profesional & Ahli Dokumentasi Medis (Ners).
    Tugas: Buat catatan perkembangan pasien (CPPT) dalam format SOAP berdasarkan input perawat.
    
    KONTEKS PASIEN:
    Nama: {patient.nama}
    {laporan_context}
    
    REFERENSI STANDAR KEPERAWATAN (SDKI/SIKI) YANG DITEMUKAN DARI SISTEM:
    {referensi_standar}
    
    UPDATE KONDISI DARI PERAWAT (INPUT):
    "{query}"
    
    INSTRUKSI PENULISAN SOAP (WAJIB DIPATUHI):
    1. **Subjective (S)**: Ringkas keluhan pasien dari input.
    2. **Objective (O)**: Ringkas data observasi/TTV dari input.
    3. **Assessment (A)**: WAJIB MENGGUNAKAN LABEL DIAGNOSIS DARI SDKI (Standar Diagnosis Keperawatan Indonesia) berdasarkan referensi di atas. Jika masalah membaik, tulis "Masalah [Nama Diagnosis SDKI] teratasi sebagian/penuh".
    4. **Plan (P)**: WAJIB MENGGUNAKAN LABEL INTERVENSI DARI SIKI (Standar Intervensi Keperawatan Indonesia) yang relevan dari referensi.
    
    Format Output WAJIB JSON (tanpa markdown):
    {{
        "subjective": "...",
        "objective": "...",
        "assessment": "...",
        "plan": "...",
        "keterangan": "..."
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}]
        )
        ai_resp = completion.choices[0].message.content
        
        # Bersihkan Markdown JSON jika ada
        ai_clean = re.sub(r"^```json\s*|\s*```$", "", ai_resp.strip(), flags=re.MULTILINE)
        
        parsed = json.loads(ai_clean)
        
    except Exception as e:
        # Fallback jika AI gagal parsing, kembalikan error agar frontend tahu
        return jsonify({"status": 500, "message": f"AI Processing Error: {str(e)}"}), 500

    now = datetime.utcnow()
    shift = determine_shift(now)

    # 4. SIMPAN KE DB DENGAN SAFETY CHECK (ensure_string)
    # Ini mencegah error "can't adapt type 'dict'" jika AI mengembalikan nested JSON
    try:
        new_cppt = CPPT(
            patient_id=patient_id, 
            user_id=user_id, 
            tanggal=now, 
            shift=shift,
            subjective=ensure_string(parsed.get("subjective")),
            objective=ensure_string(parsed.get("objective")),
            assessment=ensure_string(parsed.get("assessment")),
            plan=ensure_string(parsed.get("plan")),
            keterangan=ensure_string(parsed.get("keterangan")),
            laporan_id=laporan.id if laporan else None 
        )
        
        db.session.add(new_cppt)
        db.session.commit()
    except Exception as db_err:
        db.session.rollback()
        return jsonify({"status": 500, "message": f"Database Save Error: {str(db_err)}"}), 500

    return jsonify({
        "status": 201, 
        "message": f"CPPT berhasil dibuat (Shift {shift}) dengan standar SDKI/SIKI", 
        "data": {
            "id": new_cppt.id,
            "assessment": new_cppt.assessment,
            "plan": new_cppt.plan
        }
    }), 201

@cppt_bp.route("/<int:cppt_id>", methods=["PUT"])
@jwt_required()
def update_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404

    data = request.get_json()
    fields = ["subjective", "objective", "assessment", "plan", "keterangan", "dokter", "shift"]
    
    for f in fields:
        if f in data: 
            val = data[f]
            # Pastikan tipe data string saat update manual juga
            setattr(cppt, f, ensure_string(val))

    db.session.commit()
    return jsonify({"status": 200, "message": "CPPT updated"}), 200

@cppt_bp.route("/<int:cppt_id>", methods=["DELETE"])
@jwt_required()
def delete_cppt(cppt_id):
    cppt = CPPT.query.get(cppt_id)
    if not cppt: return jsonify({"status": 404, "message": "CPPT not found"}), 404
    
    db.session.delete(cppt)
    db.session.commit()
    return jsonify({"status": 200, "message": "CPPT deleted"}), 200