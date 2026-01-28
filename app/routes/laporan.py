from flask import Blueprint, request, jsonify
from app.model import db, Laporan, Patient, Intervensi, CPPT, User
import os, json, pickle, re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np

# --- Load API Key ---
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

def search_with_faiss(query, index, id_to_text, k=3):
    """Search di FAISS dan balikan teks mapping."""
    if index is None or not id_to_text:
        return []
    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    D, I = index.search(q_emb, k)
    return [id_to_text[i] for i in I[0] if i != -1 and i in id_to_text]

laporan_bp = Blueprint("laporan_bp", __name__, url_prefix="/laporan")

# ================== HELPERS ==================
def safe_json(val):
    try:
        return json.loads(val) if val else []
    except:
        return []

def laporan_to_dict(l):
    return {
        "id": l.id,
        "patient_id": l.patient_id,
        "tanggal": l.tanggal.isoformat(),
        "subjective": l.subjective,
        "objective": l.objective,
        "assessment": l.assessment,
        "plan": l.plan,
        "tindakan_lanjutan": l.tindakan_lanjutan,
        "keterangan": l.keterangan,
        "SDKI": safe_json(l.SDKI),
        "SIKI": safe_json(l.SIKI),
        "SLKI": safe_json(l.SLKI),
        "user_id": l.user_id
    }

# ================== GET ALL ==================
@laporan_bp.route("/", methods=["GET"])
@jwt_required()
def get_laporans():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    query = Laporan.query

    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    laporans = query.order_by(Laporan.tanggal.desc()).all()
    data = [laporan_to_dict(l) for l in laporans]

    return jsonify({"status": 200, "message": "Success", "data": data}), 200

# ================== CREATE ==================
@laporan_bp.route("/", methods=["POST"])
@jwt_required()
def create_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    payload = request.get_json()

    if not payload or not payload.get("query") or not payload.get("patient_id"):
        return jsonify({"status": 400, "message": "Harap isi query dan patient_id"}), 400

    query_text = payload["query"]
    patient_id = payload["patient_id"]

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"status": 404, "message": "Pasien tidak ditemukan"}), 404

    # Validasi Ruangan (Kecuali Admin)
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": f"Anda hanya boleh menangani pasien di ruangan {user.ruangan}"}), 403

    # ==== 1. RETRIEVAL (RAG) ====
    matches = []
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "rb") as f:
                id_to_text = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_FILE)
            # Ambil 3 chunk teratas untuk mendapatkan konteks baris tabel yang utuh
            matches = search_with_faiss(query_text, index, id_to_text, k=3)
        except Exception as e:
            print(f"FAISS Error: {e}")
            pass

    context_text = "\n\n".join(matches) if matches else "Tidak ada referensi ditemukan."

    # ==== 2. PROMPT STRICT ROW MAPPING (KHUSUS FORMAT PDF 10 PENYAKIT) ====
    
    system_prompt = """
    Anda adalah Sistem Pakar Dokumentasi Keperawatan (Ners).

    Tugas Anda adalah menyusun CPPT (Format SOAP) berdasarkan "Tabel 10 Penyakit Terbanyak".

    STRUKTUR DATA REFERENSI (WAJIB DIPAHAMI):
    Referensi berbentuk tabel dengan kolom:
    [NO] | [SDKI/Diagnosis] | [SIKI (Intervensi) & SLKI (Luaran)] | [Data Subjektif] | [Data Objektif]

    ATURAN ROW LOCKING (WAJIB PATUH):
    1. Cocokkan input perawat dengan kolom "Data Subjektif" dan "Data Objektif" di tabel referensi.
    2. Tentukan satu Nomor (NO) penyakit yang paling cocok.
    3. Kunci baris tersebut.
    4. Semua data berikut HARUS diambil HANYA dari baris yang terkunci:
    - SDKI
    - SIKI
    - SLKI
    5. DILARANG mengambil SDKI dari satu baris dan SIKI/SLKI dari baris lain.

    ATURAN KOLOM UTUH (WAJIB):
    - SDKI harus berisi SELURUH isi kolom [SDKI/Diagnosis] dari baris terkunci, tanpa diringkas, tanpa dipotong, tanpa diubah.
    - SIKI harus berisi SELURUH isi kolom [SIKI (Intervensi)] dari baris terkunci, walaupun panjang dan terdiri dari banyak poin.
    - SLKI harus berisi SELURUH isi kolom [SLKI (Luaran)] dari baris terkunci, tanpa menghilangkan kriteria hasil.
    - Jika dalam satu kolom terdapat banyak poin dalam satu sel, pecah menjadi array/list TANPA mengubah teks aslinya.

    ATURAN PEMISAHAN SIKI & SLKI:
    Jika kolom "SIKI & SLKI" tercampur:
    - SIKI biasanya diawali kata kerja: "Manajemen", "Pantau", "Berikan", "Ajarkan", "Observasi", "Identifikasi"
    - SLKI biasanya berisi target/hasil: "Menurun", "Meningkat", "Membaik", "Stabil", "Dalam batas normal"

    FORMAT OUTPUT (JSON RFC 8259 — WAJIB VALID):
    {
    "subjective": "Ringkasan keluhan pasien dari input perawat",
    "objective": "Ringkasan data objektif pasien dari input perawat",
    "assessment": "SALIN UTUH isi kolom SDKI dari baris terkunci",
    "plan": "SALIN UTUH isi kolom SIKI dari baris terkunci",
    "tindakan_lanjutan": "Saran operasional singkat berbasis kondisi pasien",
    "keterangan": "Validasi: 'Cocok dengan Penyakit No. [X] karena gejala [Y] sesuai referensi.'",
    "SDKI": [
        "SALIN semua item dari kolom SDKI baris terkunci, tanpa dipotong"
    ],
    "SIKI": [
        "SALIN semua item dari kolom SIKI baris terkunci, tanpa dipotong"
    ],
    "SLKI": [
        "SALIN semua item dari kolom SLKI baris terkunci, tanpa dipotong"
    ]
    }

    LARANGAN KERAS:
    - DILARANG menambah diagnosis, intervensi, atau luaran di luar tabel referensi.
    - DILARANG menyederhanakan, meringkas, atau memparafrase isi kolom SDKI, SIKI, dan SLKI.
    - Jika tidak ada kecocokan yang jelas, pilih baris dengan kecocokan tertinggi dan jelaskan alasannya di kolom "keterangan".

    RESPON HARUS:
    - Hanya dalam format JSON
    - Valid secara struktur
    - Tidak boleh ada teks di luar JSON
    """

    user_prompt_content = f"""
    DATA PASIEN:
    Nama: {patient.nama}
    Input Perawat: "{query_text}"
    
    REFERENSI (TABEL PDF):
    {context_text}
    """

    try:
        completion = client.chat.completions.create(
            model=api_model, 
            temperature=0.2, # Rendah untuk presisi data tabel
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content}
            ],
        )

        ai_json = completion.choices[0].message.content
        ai_json = re.sub(r"^```json\s*|\s*```$", "", ai_json.strip(), flags=re.MULTILINE)
        parsed = json.loads(ai_json)

    except Exception as e:
        print(f"LLM/Parsing Error: {e}")
        parsed = {
            "subjective": query_text,
            "objective": "-",
            "assessment": "Gagal memproses otomatis",
            "plan": "-",
            "tindakan_lanjutan": "-",
            "keterangan": "Error sistem AI",
            "SDKI": [], "SIKI": [], "SLKI": []
        }

    # ==== 3. SIMPAN KE DATABASE ====
    assess_str = parsed.get("assessment", "")
    if not assess_str and parsed.get("SDKI"):
        assess_str = ", ".join(parsed["SDKI"])

    laporan = Laporan(
        patient=patient,
        user_id=user_id,
        tanggal=datetime.utcnow(),
        subjective=parsed.get("subjective", "-"),
        objective=parsed.get("objective", "-"),
        assessment=assess_str,
        plan=parsed.get("plan", "-"),
        tindakan_lanjutan=parsed.get("tindakan_lanjutan", "-"),
        keterangan=parsed.get("keterangan", ""),
        SDKI=json.dumps(parsed.get("SDKI", [])),
        SIKI=json.dumps(parsed.get("SIKI", [])),
        SLKI=json.dumps(parsed.get("SLKI", []))
    )

    db.session.add(laporan)
    db.session.commit()

    return jsonify({
        "status": 201,
        "message": "Laporan (Askeb) berhasil dibuat",
        "data": laporan_to_dict(laporan)
    }), 201

# ================== GET BY ID ==================
@laporan_bp.route("/<int:id>", methods=["GET"])
@jwt_required()
def get_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient

    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    return jsonify({
        "status": 200,
        "message": "success",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== UPDATE ==================
@laporan_bp.route("/<int:id>", methods=["PUT"])
@jwt_required()
def update_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    payload = request.get_json()

    for field in [
        "subjective",
        "objective",
        "assessment",
        "plan",
        "tindakan_lanjutan",
        "keterangan"
    ]:
        if field in payload:
            setattr(laporan, field, payload[field])

    for field in ["SDKI", "SIKI", "SLKI"]:
        if field in payload:
            setattr(laporan, field, json.dumps(payload[field]))

    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan updated successfully",
        "data": laporan_to_dict(laporan)
    }), 200

# ================== DELETE ==================
@laporan_bp.route("/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_laporan(id):
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    laporan = Laporan.query.get(id)
    if not laporan:
        return jsonify({"status": 404, "message": "Laporan not found", "data": None}), 404

    patient = laporan.patient
    if user.role.name != 'admin' and user.ruangan and patient.ruangan != user.ruangan:
        return jsonify({"status": 403, "message": "Akses ditolak"}), 403

    db.session.delete(laporan)
    db.session.commit()

    return jsonify({
        "status": 200,
        "message": "Laporan deleted successfully",
        "data": None
    }), 200

# ================== SEARCH ==================
@laporan_bp.route("/search", methods=["GET"])
@jwt_required()
def search_laporan():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)

    keyword = request.args.get("q")
    if not keyword:
        return jsonify({"status": 400, "message": "Query parameter 'q' required", "data": None}), 400

    query = Laporan.query.filter(
        (Laporan.subjective.ilike(f"%{keyword}%")) |
        (Laporan.objective.ilike(f"%{keyword}%")) |
        (Laporan.assessment.ilike(f"%{keyword}%")) |
        (Laporan.plan.ilike(f"%{keyword}%")) |
        (Laporan.keterangan.ilike(f"%{keyword}%"))
    )

    if user.role.name != 'admin' and user.ruangan:
        query = query.join(Patient).filter(Patient.ruangan == user.ruangan)

    results = query.all()

    return jsonify({
        "status": 200,
        "message": "success",
        "data": [laporan_to_dict(l) for l in results]
    }), 200
